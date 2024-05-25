import cv2
import mediapipe as mp
import time


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# camera init
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open video device")
    exit()


def count_fingers(hand_landmarks):
    fingers = {"4": 0, "8": 0, "12": 0, "16": 0, "20": 0}
    
    # Thumb
    if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
        fingers["4"] = 1
        
    # Index finger
    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
        fingers["8"] = 1
        
    # Middle finger
    if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
        fingers["12"] = 1
        
    # Ring finger
    if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
        fingers["16"] = 1
        
    # Pinky finger
    if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
        fingers["20"] = 1
        
    return sum(fingers.values())

# get chord based on number of fingers
def get_chord(num_fingers):
    chords = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    if 1 <= num_fingers <= 7:
        return chords[num_fingers - 1]
    return ""

# Main loop
pTime = 0
current_chord = ""
while True:
    success, frame = cap.read()
    if not success:
        print("Could not read frame from video stream")
        break

    # frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    # Check if hands are detected
    total_fingers = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count the number of fingers held up on each hand
            total_fingers += count_fingers(hand_landmarks)

    # Get the corresponding chord
    current_chord = get_chord(total_fingers)

    # Display the chord on the frame
    if current_chord:
        cv2.putText(frame, f"Chord: {current_chord}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)


    cv2.imshow("Hand Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
