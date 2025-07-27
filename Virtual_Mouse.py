import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from pynput.mouse import Button, Controller
mouse = Controller()

screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,   
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

def get_angle(a,b,c):
    radian = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(np.degrees(radian))
    return angle


def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]

    return None

def is_drawing(landmark_list):
    if len(landmark_list)>=21:
        if get_angle(landmark_list[13],landmark_list[14],landmark_list[16]) < 50 and get_angle(landmark_list[9],landmark_list[10],landmark_list[12]) < 50 and get_angle(landmark_list[17],landmark_list[18],landmark_list[20]) < 50 and get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) > 50:
            return 1
        else:
            return 0
    else:
        return 2

def move_mouse(index_finder_tip):
    if index_finder_tip is not None:
        x = int(index_finder_tip.x * screen_width)
        y = int(index_finder_tip.y * screen_height)
        pyautogui.moveTo(x,y)

def main():
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils
    try:
        a=[]
        flag1 = False
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)
            frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            processed = hands.process(frameRGB)

            landmark_list = []

            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame,hand_landmarks,mpHands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x,lm.y))
            
            index_finger_tip = find_finger_tip(processed)
            move_mouse(index_finger_tip)
            drawing = is_drawing(landmark_list)

            if drawing==1:
                flag1 = True
                print("drawing")
                a.append(index_finger_tip)
            elif drawing ==2:  #if speed of and increase the landmark get null
                flag1 = False
            elif flag1:
                break

            cv2.imshow('Frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print(a)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


