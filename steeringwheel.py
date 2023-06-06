import cv2
import mediapipe as mp
import pydirectinput
import directinput
import time

import cv2
c = cv2.VideoCapture(0)

c.set(3, 1280)
c.set(4, 720)

mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def write_text(img, text, x, y):

    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (x, y)
    fontScale = 1
    fontColor = (255, 255, 255) # White.
    lineType = 2
    cv2.putText(img,
                text,
                pos,
                font,
                fontScale,
                fontColor,
                lineType)

def steering_wheel():

    prev_frame_time = 0
    new_frame_time = 0
    while c.isOpened():
        success, img = c.read()
        cv2.waitKey(1) # Continuously refreshes the webcam frame every 1ms.
        img = cv2.flip(img, 1)
        img.flags.writeable = False # Making the images not writeable for optimization.
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Processing video.
        landmarks = results.multi_hand_landmarks
        if landmarks:



            new_frame_time = time.time()
            fps = str(int(1/(new_frame_time - prev_frame_time)))
            write_text(img, fps, 150, 500)
            prev_frame_time = new_frame_time

            if(len(landmarks) == 2): # If 2 hands are in view.
                left_hand_landmarks = landmarks[1].landmark
                right_hand_landmarks = landmarks[0].landmark


                shape = img.shape
                width = shape[1]
                height = shape[0]


                left_mFingerX, left_mFingerY = (left_hand_landmarks[11].x * width), (left_hand_landmarks[11].y * height)
                right_mFingerX, right_mFingerY = (right_hand_landmarks[11].x * width), (right_hand_landmarks[11].y * height)

                slope = ((right_mFingerY - left_mFingerY)/(right_mFingerX-left_mFingerX))


                sensitivity = 0.3
                if abs(slope) > sensitivity:
                    if slope < 0:

                        print("Turn left.")
                        write_text(img, "Left.", 360, 360)
                        directinput.release_key("w")
                        directinput.release_key('a')
                        directinput.press_key('a')
                    if slope > 0:

                        print("Turn right.")
                        write_text(img, "Right.", 360, 360)
                        directinput.release_key('w')
                        directinput.release_key('a')
                        directinput.press_key('d')
                if abs(slope) < sensitivity:

                    print("Keeping straight.")
                    write_text(img, "Straight.", 360, 360)
                    directinput.release_key('a')
                    directinput.release_key('d')
                    directinput.press_key('w')


            for hand_landmarks in landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Hand Recognition", img)
    c.release()
steering_wheel()
