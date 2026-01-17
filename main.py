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

def crop_white_borders(img):
    """Supprime les bordures blanches autour de l'image."""
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]
    return img

# Charger la carte plate de la terre pour la projection 3D
imgEarthMap = cv2.imread("earth_map.png")
imgEarthMap = crop_white_borders(imgEarthMap)

if imgEarthMap is None:
    print("Erreur: earth_map.png non trouvé. Utilisation de l'image par défaut.")
    imgEarthMap = cv2.imread("earth.png")
    imgEarthMap = crop_white_borders(imgEarthMap)

def get_3d_globe(img_map, angle_deg, size):
    """Génère une vue 3D sphérique de la Terre à partir d'une carte plate."""
    if size < 10: size = 10
    
    # Créer les coordonnées de la grille de sortie
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xv, yv = np.meshgrid(x, y)
    
    # Masque circulaire
    dist = xv**2 + yv**2
    mask = dist <= 1
    
    # Coordonnées 3D (z = profondeur)
    zv = np.sqrt(np.maximum(0, 1 - dist))
    
    # Mapping sphérique
    # phi = latitude, theta = longitude
    phi = np.arcsin(np.clip(yv, -1, 1))
    theta = np.arctan2(xv, zv) + np.radians(angle_deg)
    
    # Conversion en coordonnées normalisées (0-1) pour la carte équirectangulaire
    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi
    u = np.mod(u, 1.0) # Rotation infinie
    
    # Conversion en pixels et Remap (plus rapide que des boucles)
    map_h, map_w = img_map.shape[:2]
    map_x = (u * map_w).astype(np.float32)
    map_y = (v * map_h).astype(np.float32)
    
    globe = cv2.remap(img_map, map_x, map_y, cv2.INTER_LINEAR)
    
    # Ajouter la transparence
    b, g, r = cv2.split(globe)
    alpha = (mask * 255).astype(np.uint8)
    return cv2.merge((b, g, r, alpha))

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.flip(img, 1) # Effet miroir

    # Détection des mains
    hands, img = detector.findHands(img, flipType=False)
    
    # Initialisation
    left_hand = None
    right_hand = None

    if hands:
        # Identifier les mains
        for hand in hands:
            if hand['type'] == "Left":
                left_hand = hand
            if hand['type'] == "Right":
                right_hand = hand

        # --- LOGIQUE MAIN GAUCHE (Porte la Terre) ---
        if left_hand:
            lmList = left_hand['lmList']
            # Position sur le bout de l'index (landmark 8)
            cx, cy = lmList[8][0], lmList[8][1]

        # --- LOGIQUE MAIN DROITE (Zoom + Rotation + Lien) ---
        if right_hand:
            lmListR = right_hand['lmList']
            
            # 1. Rotation 3D (basée sur horizontal de l'index)
            curr_x = lmListR[8][0]
            if last_x_right is not None:
                diff = curr_x - last_x_right
                angle += diff * 0.5 
            last_x_right = curr_x

            # 2. Zoom
            p1 = lmListR[4][0:2]
            p2 = lmListR[8][0:2]
            length, info, img = detector.findDistance(p1, p2, img)
            scale = int(np.interp(length, [20, 250], [50, 600]))
            
            # 3. Lien Google Earth (Si la main DROITE est ouverte)
            fingers = detector.fingersUp(right_hand)
            finger_count = sum(fingers)
            
            # Plus robuste : on déclenche si au moins 4 doigts sont levés
            if finger_count >= 4: 
                cv2.putText(img, "OPEN HAND!", (right_hand['bbox'][0], right_hand['bbox'][1]-40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                if not url_opened:
                    webbrowser.open('https://earth.google.com/web/')
                    url_opened = True
            else:
                url_opened = False
            
            # Afficher le type de main pour debug
            cv2.putText(img, "RIGHT", (right_hand['bbox'][0], right_hand['bbox'][1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        else:
            last_x_right = None
            url_opened = False

        if left_hand:
            cv2.putText(img, "LEFT (Globe)", (left_hand['bbox'][0], left_hand['bbox'][1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # --- AFFICHAGE DE LA TERRE 3D (Uniquement si une main gauche est présente) ---
        if left_hand:
            try:
                # Générer le globe 3D à la volée avec la bonne rotation et taille
                imgGlobe3D = get_3d_globe(imgEarthMap, angle, scale)

                # Superposition
                h, w, _ = img.shape
                x1, y1 = cx - scale // 2, cy - scale // 2
                
                if x1 < w and y1 < h and x1 + scale > 0 and y1 + scale > 0:
                    img = cvzone.overlayPNG(img, imgGlobe3D, [x1, y1])
                
            except Exception as e:
                pass

    if url_opened:
        cv2.putText(img, "Google Earth Link Active!", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Earth 3D Tracker", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
