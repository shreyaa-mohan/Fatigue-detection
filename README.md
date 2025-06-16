# 💤 Fatigue Detection System Using Drowsiness and Yawn Monitoring

This project is a real-time fatigue detection system using computer vision and facial landmark analysis. It monitors a user’s eye closure, blinking rate, and yawning, then triggers alerts via sound and Telegram messages if signs of drowsiness or fatigue are detected.

## 🧠 Features

- Detects prolonged eye closure and frequent blinking using EAR (Eye Aspect Ratio)
- Detects yawning using MAR (Mouth Aspect Ratio)
- Plays an alarm sound when drowsiness is detected
- Sends emergency Telegram alerts to pre-defined contacts
- Built with OpenCV, Dlib, Scipy, and Pygame

## 📁 Project Structure

fatigue-detection/
- ├── fatiguedetection.py                 # Main script
- ├── music.wav                           # Alarm sound
- ├── models/
- │   └── shape_predictor_68_face_landmarks.dat   # Pre-trained dlib model
  
## ⚙️ Installation

### 1. Clone the repository:
- git clone https://github.com/yourusername/fatigue-detection.git
- cd fatigue-detection

### 2.Install dependencies:
- pip install -r requirements.txt

### 3.🚀 How to Run
- python fatiguedetection.py
- Make sure your webcam is connected and working.
- Press q to exit the program.

### 4.📲 Telegram Alerts Setup

- Create a Telegram bot via BotFather and get your API token.
- Get the chat ID of your emergency contact:
- Start a chat with your bot
- Use tools like get_id_bot

## ✅ Output

### Displays live video feed with contours drawn on eyes and mouth

- Text annotations:
- “Drowsy!”
- “Yawning!”
- “Fatigue Detected!”
- Sends Telegram alert:
 -- 🚨 Fatigue Alert Detected! 🚨
-- The driver shows signs of fatigue. Please take immediate action!


 ## 🔮 Future Improvements

- Environment Variable Integration: Move the Telegram API token and chat ID to a .env file and use python-dotenv for secure access.
- Driver Monitoring on Mobile: Convert the solution into a mobile app using Kivy or Android’s CameraX + ML Kit for portability.
- Multi-Face Handling: Extend support to detect fatigue in multiple people simultaneously (e.g., group monitoring in transport or workplace).
- Sleep Detection Using Head Pose: Incorporate head pose estimation to detect nodding off or slumping, which are common in fatigue.

## 🙋‍♀️ Author
- Shreya Mohan:shreyamohan74@gmail.com
- Open to feedback and collaboration!

