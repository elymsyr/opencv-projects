import cv2
import mediapipe as mp
from fer import FER

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Initialize FER for emotion detection
emotion_detector = FER()

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for hand gestures
    results = hands.process(rgb_frame)
    
    # Draw hand annotations on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Detect emotion in the frame
    emotions = emotion_detector.detect_emotions(rgb_frame)
    
    # Draw emotions on the frame
    if emotions:
        for emotion in emotions:
            (x, y, w, h) = emotion['box']
            emotion_text = max(emotion['emotions'], key=emotion['emotions'].get)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Gesture and Emotion Tracker', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
