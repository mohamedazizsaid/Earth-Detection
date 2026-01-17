# 🌍 Earth Hand Tracking 3D

Une application Python de vision par ordinateur utilisant **MediaPipe** et **OpenCV** pour afficher une Terre en 3D projetée sur votre main, contrôlable par gestes.

![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00BFFF?style=for-the-badge&logo=google&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## ✨ Fonctionnalités

- **🌐 Projection Sphérique 3D** : Un rendu réaliste de la Terre utilisant une projection équirectangulaire sur une sphère virtuelle (exit la rotation 2D plate !).
- **✋ Tracking de Précision** : La Terre est ancrée sur le bout de l'index de votre main gauche.
- **🔍 Gestes de Zoom** : Utilisez votre main droite (pincer/écarter le pouce et l'index) pour agrandir ou rétrécir le globe.
- **🔄 Rotation Dynamique** : Faites tourner la Terre sur son axe en bougeant votre main droite horizontalement.
- **🚀 Web Trigger** : Ouvrez automatiquement **Google Earth** en ouvrant grand votre main droite.
- **🖼️ Traitement d'Image Auto** : Suppression automatique des bordures blanches pour un rendu premium sans background.

## 🛠️ Installation

1. **Cloner le repository** :
   ```bash
   git clone https://github.com/mohamedazizsaid/Earth-Detection.git
   cd Earth-Detection
   ```

2. **Créer un environnement virtuel (recommandé)** :
   ```bash
   python -m venv env
   .\env\Scripts\activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install opencv-python cvzone mediapipe numpy
   ```

## 🎮 Commandes et Gestes

### Lancement
```bash
python main.py
```

### Gestes de contrôle
| Main | Geste | Action |
| :--- | :--- | :--- |
| **Gauche** | Lever l'index | Affiche la Terre sur le bout du doigt |
| **Droite** | Pincer/Écarter (Pouce-Index) | Zoom +/- |
| **Droite** | Mouvement horizontal | Rotation de la Terre |
| **Droite** | Ouvrir la main (4+ doigts) | Ouvre Google Earth dans le navigateur |
| **Clavier** | Appuyer sur `q` | Quitter l'application |

## 📸 Aperçu

La Terre est générée dynamiquement à partir d'une carte plate (`earth_map.png`) et projetée mathématiquement pour simuler un volume 3D parfait, offrant une expérience immersive directement dans votre webcam.

---
Développé avec ❤️ par [Mohamed Aziz Said](https://github.com/mohamedazizsaid)
