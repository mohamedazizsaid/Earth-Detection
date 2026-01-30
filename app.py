from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import webbrowser
import numpy as np
import io
import uvicorn
import threading

app = FastAPI()

# Ajouter les middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration de la Caméra
try:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Largeur
    cap.set(4, 720)   # Hauteur
    camera_available = cap.isOpened()
except Exception as e:
    print(f"Erreur caméra: {e}")
    cap = None
    camera_available = False

# Configuration du Détecteur de main
try:
    detector = HandDetector(detectionCon=0.8, maxHands=2)
except Exception as e:
    print(f"Erreur détecteur: {e}")
    detector = None

# Variables de contrôle
angle = 0
scale = 150  # Taille de base
cx, cy = 500, 500
url_opened = False
last_x_right = None

def crop_white_borders(img):
    """Supprime les bordures blanches autour de l'image."""
    if img is None: 
        return None
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return img[y:y+h, x:x+w]
    except:
        pass
    return img

# Charger la carte plate de la terre pour la projection 3D
imgEarthMap = None
try:
    imgEarthMap = cv2.imread("earth_map.png")
    if imgEarthMap is not None:
        imgEarthMap = crop_white_borders(imgEarthMap)
    else:
        imgEarthMap = cv2.imread("earth.png")
        if imgEarthMap is not None:
            imgEarthMap = crop_white_borders(imgEarthMap)
except Exception as e:
    print(f"Erreur chargement image: {e}")

def get_3d_globe(img_map, angle_deg, size):
    """Génère une vue 3D sphérique de la Terre à partir d'une carte plate."""
    try:
        if img_map is None or img_map.size == 0:
            return None
        if size < 10: 
            size = 10
        
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
        phi = np.arcsin(np.clip(yv, -1, 1))
        theta = np.arctan2(xv, zv) + np.radians(angle_deg)
        
        # Conversion en coordonnées normalisées (0-1) pour la carte équirectangulaire
        u = (theta + np.pi) / (2 * np.pi)
        v = (phi + np.pi / 2) / np.pi
        u = np.mod(u, 1.0)  # Rotation infinie
        
        # Conversion en pixels et Remap
        map_h, map_w = img_map.shape[:2]
        map_x = (u * map_w).astype(np.float32)
        map_y = (v * map_h).astype(np.float32)
        
        globe = cv2.remap(img_map, map_x, map_y, cv2.INTER_LINEAR)
        
        # Ajouter la transparence
        b, g, r = cv2.split(globe)
        alpha = (mask * 255).astype(np.uint8)
        return cv2.merge((b, g, r, alpha))
    except Exception as e:
        print(f"Erreur get_3d_globe: {e}")
        return None

def generate_frames():
    """Générateur MJPEG pour le streaming vidéo."""
    global angle, scale, cx, cy, url_opened, last_x_right, cap, detector, imgEarthMap
    
    if not camera_available or cap is None:
        print("Caméra non disponible")
        # Créer une image de placeholder
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(img, "Camera not available", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')
    else:
        while True:
            try:
                success, img = cap.read()
                if not success:
                    break
                
                img = cv2.flip(img, 1)  # Effet miroir

                # Détection des mains
                if detector:
                    hands, img = detector.findHands(img, flipType=False)
                else:
                    hands = []
                
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
                        
                        if finger_count >= 4: 
                            cv2.putText(img, "OPEN HAND!", (right_hand['bbox'][0], right_hand['bbox'][1]-40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                            if not url_opened:
                                webbrowser.open('https://earth.google.com/web/')
                                url_opened = True
                        else:
                            url_opened = False
                        
                        cv2.putText(img, "RIGHT", (right_hand['bbox'][0], right_hand['bbox'][1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    else:
                        last_x_right = None
                        url_opened = False

                    if left_hand:
                        cv2.putText(img, "LEFT (Globe)", (left_hand['bbox'][0], left_hand['bbox'][1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                    # --- AFFICHAGE DE LA TERRE 3D ---
                    if left_hand and imgEarthMap is not None:
                        try:
                            imgGlobe3D = get_3d_globe(imgEarthMap, angle, scale)
                            if imgGlobe3D is not None:
                                h, w, _ = img.shape
                                x1, y1 = cx - scale // 2, cy - scale // 2
                                
                                if x1 < w and y1 < h and x1 + scale > 0 and y1 + scale > 0:
                                    img = cvzone.overlayPNG(img, imgGlobe3D, [x1, y1])
                        except Exception as e:
                            pass

                if url_opened:
                    cv2.putText(img, "Google Earth Link Active!", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                # Encoder l'image en JPEG
                _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Erreur dans generate_frames: {e}")
                break

@app.get("/", response_class=HTMLResponse)
def root():
    html_content = """<!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Earth 3D Tracking</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 20px;
                }
                .container {
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    padding: 30px;
                    max-width: 1400px;
                    width: 100%;
                }
                h1 {
                    color: #333;
                    font-size: 48px;
                    margin-bottom: 30px;
                    text-align: center;
                }
                .video-container {
                    position: relative;
                    width: 100%;
                    max-width: 1280px;
                    margin: 0 auto 30px;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    background: #000;
                }
                .video-container img {
                    width: 100%;
                    height: auto;
                    display: block;
                }
                .info {
                    background: #f0f0f0;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }
                .info h3 {
                    color: #667eea;
                    margin-bottom: 10px;
                }
                .info p {
                    color: #666;
                    line-height: 1.6;
                    margin-bottom: 8px;
                }
                .controls {
                    display: flex;
                    gap: 10px;
                    justify-content: center;
                    flex-wrap: wrap;
                }
                button {
                    padding: 12px 24px;
                    background: #667eea;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }
                button:hover {
                    background: #764ba2;
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
                }
                .status {
                    text-align: center;
                    padding: 15px;
                    background: #e8f5e9;
                    border-radius: 5px;
                    color: #2e7d32;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌍 Earth 3D Tracking - Live</h1>
                
                <div class="video-container">
                    <img src="/video" alt="Camera Stream" style="width: 100%; height: auto;">
                </div>
                
                <div class="info">
                    <h3>📋 Instructions:</h3>
                    <p><strong>Main Gauche:</strong> Portez la Terre 3D - déplacez votre main gauche pour contrôler la position</p>
                    <p><strong>Main Droite:</strong> Zoom et Rotation</p>
                    <p>- Écartez le pouce et l'index pour zoomer</p>
                    <p>- Déplacez l'index gauche/droite pour faire tourner la Terre</p>
                    <p>- Levez tous les doigts pour ouvrir Google Earth</p>
                </div>
                
                <div class="status">
                    ✓ Camera en direct - Main tracking actif
                </div>
            </div>
            
            <script>
                // Garder le flux vidéo à jour
                setInterval(() => {
                    const img = document.querySelector('.video-container img');
                    if (img) {
                        img.src = '/video?t=' + new Date().getTime();
                    }
                }, 100);
            </script>
        </body>
    </html>"""
    return html_content

@app.get("/video")
def video():
    """Route MJPEG pour le streaming vidéo."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
def status():
    return {
        "status": "running", 
        "app": "Earth 3D Tracker",
        "camera": camera_available
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

