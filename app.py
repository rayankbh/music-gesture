import pygame
import sys
import cv2
import mediapipe as mp
import os
import numpy as np

# Pygame Init
pygame.init()
pygame.mixer.init()

# Screen setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT), )
pygame.display.set_caption("Chordzio")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Fonts
title_font = pygame.font.Font(None, 64)
menu_font = pygame.font.Font(None, 32)

# Main Mediapipe hands set up
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# chord sounds
chord_sounds = {}
chord_dir = "music-gesture-inside/chords/CMaj"
for chord_file in os.listdir(chord_dir):
    if chord_file.endswith(".mp3"):
        chord_name = chord_file[:-4]
        chord_sound = pygame.mixer.Sound(os.path.join(chord_dir, chord_file))
        chord_sounds[chord_name] = chord_sound


'''
Counts fingers based on landmarks and handedness from MediaPipe lib: 
https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
'''
def count_fingers(hand_landmarks, handedness):
    tips_ids = [4, 8, 12, 16, 20]
    fingers_up = [0, 0, 0, 0, 0]

    is_left_hand = handedness.classification[0].label == 'Left'

    if is_left_hand:
        if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
            fingers_up[0] = 1
    else:
        if hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0] - 1].x:
            fingers_up[0] = 1

    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers_up[id] = 1
    
    return sum(fingers_up)


'''
Maps number of fingers to chord
'''
def get_chord(num_fingers):
    chords = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    if 1 <= num_fingers <= 7:
        return chords[num_fingers - 1]
    return ""


'''
Draws text on the screen for pygame 
'''
def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, 1, color)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)


'''
Runs main menu
'''
def main_menu():
    while True:
        screen.fill(BLACK)
        draw_text('Chordzio', title_font, WHITE, screen, WIDTH//2 - 100, 100)

        mx, my = pygame.mouse.get_pos()

        button_1 = pygame.Rect(WIDTH//2 - 100, 250, 200, 50)
        button_2 = pygame.Rect(WIDTH//2 - 100, 350, 200, 50)

        pygame.draw.rect(screen, WHITE, button_1)
        pygame.draw.rect(screen, WHITE, button_2)

        draw_text('Play', menu_font, BLACK, screen, WIDTH//2 - 30, 260)
        draw_text('Quit', menu_font, BLACK, screen, WIDTH//2 - 30, 360)

        click = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    click = True

        if button_1.collidepoint((mx, my)):
            if click:
                game()
        if button_2.collidepoint((mx, my)):
            if click:
                pygame.quit()
                sys.exit()

        pygame.display.update()


'''
Main game loop
'''
def game():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open video device")
        return

    # Calculate scaling factors and offsets to fit and center the camera feed within the window while maintaining aspect ratio
    camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_width = WIDTH / camera_width
    scale_height = HEIGHT / camera_height
    scale = min(scale_width, scale_height)
    new_width = int(camera_width * scale)
    new_height = int(camera_height * scale)
    x_offset = (WIDTH - new_width) // 2
    y_offset = (HEIGHT - new_height) // 2


    # TO-DO: Add resizable window  
    current_chord = ""
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        success, frame = cap.read()
        if not success:
            print("Could not read frame from video stream")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        total_fingers = 0
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                total_fingers += count_fingers(hand_landmarks, handedness)
        
        new_chord = get_chord(total_fingers)
        
        if new_chord != current_chord:
            current_chord = new_chord
            if current_chord in chord_sounds:
                chord_sounds[current_chord].play()
        
        # create image for text and then put in top right corner of video frame  
        if current_chord:

            chord_text = f"Chord: {current_chord}"
            
            text_image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            cv2.putText(text_image, chord_text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, BLUE, 3)
            
            text_image = cv2.flip(text_image, 1)
    
            frame = cv2.addWeighted(frame, 1, text_image, 1, 0)


        # Resize the frame to fit the window
        frame = cv2.resize(frame, (new_width, new_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        
        screen.fill(BLACK)
        
        screen.blit(frame, (x_offset, y_offset))
        
        pygame.display.flip()

    cap.release()

if __name__ == "__main__":
    main_menu()