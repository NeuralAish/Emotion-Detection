# Emotion-Detection
Real-Time Facial Emotion Recognition using Deep Learning
It detects human emotions live and displays them on the video feed.

## 📌 Features

- 🎥 Real-time emotion detection via webcam  
- 🧠 CNN-based deep learning model  
- 😀 Supports **7 emotions**:
  - Angry  
  - Disgust  
  - Fear  
  - Happy  
  - Neutral  
  - Sad  
  - Surprise  

## 📊 Dataset Information


- **Source:** Kaggle – Facial Expression Recognition Dataset  
- **Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise  
- **Input Format:** 48×48 grayscale images  
- **Dataset is NOT included** due to size and licensing constraints.

🔗 Download dataset from Kaggle:  
👉https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset


### 📌 Usage Options

#### ✅ Option 1: Use Pre-trained Model (Recommended)
Run real-time emotion detection directly:   python realtime_emotion.py


#### 🧪 Option 2: Train the Model Yourself
1. Download the dataset from Kaggle  
2. Place it in the following structure:

images/
├── train/
└── validation/


4. Train the model:
python train_emotion_model.py

⚠️ Using the `.h5` model is **optional** and provided only for convenience.








































