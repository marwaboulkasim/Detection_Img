# 🔥 Détection d’incendies sur images satellites avec YOLO

## 📌 Contexte du projet
Ce projet vise à entraîner et comparer plusieurs variantes de **YOLO (You Only Look Once)** pour détecter automatiquement les incendies dans des **images satellites annotées**.  

Les étapes principales :
1. Préparer et organiser les données  
2. Installer et configurer l’environnement YOLO  
3. Entraîner plusieurs modèles YOLO (nano, small, medium, large)  
4. Évaluer et comparer leurs performances  
5. Utiliser le meilleur modèle pour l’inférence  

---

## ⚙️ Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/<utilisateur>/<nom-du-projet>.git
cd <nom-du-projet>




2. Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate   

3. Installer les dépendances
pip install --upgrade pip
pip install ultralytics




📂 Arborescence du projet
.
├── datasets/
│   └── fires/
│       ├── data.yaml
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── val/
│           ├── images/
│           └── labels/
├── runs/                 # Résultats générés automatiquement par YOLO
├── train.py              # Script principal d'entraînement
└── README.md             # Documentation du projet


Exemple de data.yaml
train: datasets/fires/train/images
val: datasets/fires/val/images

nc: 1
names: ["fire"]