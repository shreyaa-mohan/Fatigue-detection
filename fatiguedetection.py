import cv2
import imutils
import dlib
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer
import telebot

# Initialize Telegram bot
API_TOKEN = "7866774075:AAGkkjmwnUcC71ko6mA6rL8owjbpIoBbGxk"  # Replace with your Telegram Bot API token
chat_ids = ["5418876475"]  # Replace with the chat IDs of emergency contacts
bot = telebot.TeleBot(API_TOKEN)

# Function to send Telegram alert
def send_telegram_alert(message):
    for chat_id in chat_ids:
        bot.send_message(chat_id, message)

# Initialize mixer for playing sound
mixer.init()
mixer.music.load("music.wav")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Initialize thresholds and counters
ear_thresh = 0.28
partial_thresh = 0.22
blink_threshold = 0.15
frame_check = 20
fatigue_score_threshold = 3
frame_count = 0
blink_count = 0
fatigue_score = 0
alert_sent = False  # Prevent duplicate alerts

# Initialize dlib face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Facial landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Calculate EAR and MAR
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        # Draw contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Detect eye closure
        if ear < ear_thresh:
            frame_count += 1
            if frame_count >= frame_check:
                fatigue_score += 1
                cv2.putText(frame, "Drowsy!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not mixer.music.get_busy():  # Play only if not already playing
                    mixer.music.play(-1)  # Loop playback
        else:
            frame_count = 0
            if mixer.music.get_busy():  # Stop music when eyes open
                mixer.music.stop()

        # Detect partial eye closure
        if ear >= ear_thresh and ear < partial_thresh:
            cv2.putText(frame, "Partially Closed Eyes", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Detect frequent blinking
        if ear < blink_threshold:
            blink_count += 1
            if blink_count > 5:
                fatigue_score += 1
                cv2.putText(frame, "Frequent Blinking", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            blink_count = 0

        # Detect yawning
        if mar > 0.7:
            fatigue_score += 1
            cv2.putText(frame, "Yawning!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check fatigue score
        if fatigue_score >= fatigue_score_threshold and not alert_sent:
            cv2.putText(frame, "Fatigue Detected!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            message = (
                "🚨 Fatigue Alert Detected! 🚨\n"
                "The driver shows signs of fatigue. Please take immediate action!"
            )
            send_telegram_alert(message)
            alert_sent = True

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break loop on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
