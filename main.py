import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import webbrowser
import numpy as np

# 1. Configuration de la Caméra
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Largeur
cap.set(4, 720)  # Hauteur

# 2. Configuration du Détecteur de main
detector = HandDetector(detectionCon=0.8, maxHands=2)

# 3. Variables de contrôle
angle = 0
scale = 150 # Taille de base
cx, cy = 500, 500
url_opened = False
last_x_right = None

# Charger l'image de la terre (PNG transparent requis)
imgEarth = cv2.imread("earth.png", cv2.IMREAD_UNCHANGED)

if imgEarth is None:
    print("Erreur: earth.png non trouvé. Assurez-vous que l'image est dans le dossier.")
    exit()

def rotate_image(img, angle):
    """Fait pivoter l'image selon l'angle donné."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.flip(img, 1) # Effet miroir

    # Détection des mains
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        left_hand = None
        right_hand = None

        # Identifier les mains
        for hand in hands:
            if hand['type'] == "Left":
                left_hand = hand
            if hand['type'] == "Right":
                right_hand = hand

        # --- LOGIQUE MAIN GAUCHE (Porte la Terre + Lien) ---
        if left_hand:
            lmList = left_hand['lmList']
            # Position sur la paume (landmark 9)
            cx, cy = lmList[9][0], lmList[9][1]
            
            # Vérifier si la main est ouverte
            fingers = detector.fingersUp(left_hand)
            if fingers == [1, 1, 1, 1, 1]: 
                if not url_opened:
                    webbrowser.open('https://earth.google.com/web/')
                    url_opened = True
            else:
                url_opened = False

        # --- LOGIQUE MAIN DROITE (Zoom + Rotation) ---
        if right_hand:
            lmListR = right_hand['lmList']
            
            # 1. Rotation (basée sur le mouvement horizontal de l'index)
            curr_x = lmListR[8][0]
            if last_x_right is not None:
                diff = curr_x - last_x_right
                angle += diff * 0.5 # Sensibilité de rotation
            last_x_right = curr_x

            # 2. Zoom (distance entre pouce et index)
            p1 = lmListR[4][0:2]
            p2 = lmListR[8][0:2]
            length, info, img = detector.findDistance(p1, p2, img)
            
            # Ajustement dynamique du scale
            new_scale = int(np.interp(length, [20, 250], [50, 600]))
            scale = new_scale
        else:
            last_x_right = None

    # --- AFFICHAGE DE LA TERRE ---
    try:
        # 1. Redimensionnement
        imgResized = cv2.resize(imgEarth, (scale, scale))
        
        # 2. Rotation
        imgRotated = rotate_image(imgResized, angle)

        # 3. Superposition
        # On centre l'image sur cx, cy
        img = cvzone.overlayPNG(img, imgRotated, [cx - scale // 2, cy - scale // 2])
        
    except Exception as e:
        # Sécurité si l'image dépasse les bords
        pass

    if url_opened:
        cv2.putText(img, "Google Earth Link Active!", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Earth Hand Tracker", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
