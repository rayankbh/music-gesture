import cv2
import mediapipe as mp
import pygame
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame mixer
pygame.mixer.init()

# Load chord sounds
chord_sounds = {}
chord_dir = "chords/CMaj"
for chord_file in os.listdir(chord_dir):
    if chord_file.endswith(".mp3"):
        chord_name = chord_file[:-4]
        chord_sound = pygame.mixer.Sound(os.path.join(chord_dir, chord_file))
        chord_sounds[chord_name] = chord_sound

# Camera initialization
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open video device")
    exit()

def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers_up = [0, 0, 0, 0, 0]

    # Thumb
    if hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers_up[0] = 1

    # Other fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers_up[id] = 1
    
    return sum(fingers_up)

def get_chord(num_fingers):
    chords = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    if 1 <= num_fingers <= 7:
        return chords[num_fingers - 1]
    return ""

def main():
    current_chord = ""
    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read frame from video stream")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        total_fingers = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                total_fingers += count_fingers(hand_landmarks)
        
        new_chord = get_chord(total_fingers)
        
        if new_chord != current_chord:
            current_chord = new_chord
            if current_chord in chord_sounds:
                chord_sounds[current_chord].play()
        
        if current_chord:
            cv2.putText(frame, f"Chord: {current_chord}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 3)
        
        cv2.imshow("GestureMuse", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
