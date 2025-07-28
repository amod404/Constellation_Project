import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from scipy.interpolate import interp1d
from pynput.mouse import Button, Controller
mouse = Controller()

screen_width, screen_height = pyautogui.size()

ANSWER = None
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,   
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

def resample_points(points, num_points=50):
    # Compute cumulative distance between points
    diffs = np.diff(points, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative_dist = np.concatenate([[0], np.cumsum(dists)])

    # Create interpolation function
    f_interp = interp1d(cumulative_dist, points, axis=0, kind='linear')

    # New evenly spaced distances
    new_distances = np.linspace(0, cumulative_dist[-1], num_points)

    # Get resampled points
    new_points = f_interp(new_distances)
    return new_points

def get_angle(a,b,c):
    radian = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(np.degrees(radian))
    return angle


def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        tip.y = tip.y/0.7
        return tip
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
        a = []
        flag1 = False
        no_draw_frames = 0  # NEW

        question_img = cv2.imread("image.png")
        cv2.namedWindow("FullScreen", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("FullScreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("FullScreen", question_img)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            processed = hands.process(frameRGB)

            landmark_list = []

            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            index_finger_tip = find_finger_tip(processed)
            if index_finger_tip is not None:
                move_mouse(index_finger_tip)

            drawing = is_drawing(landmark_list)

            if drawing == 1:
                flag1 = True
                no_draw_frames = 0
                print("drawing")
                a.append([index_finger_tip.x, index_finger_tip.y])
            elif drawing == 2:
                # landmark missing, maybe move too fast
                no_draw_frames += 1
            elif flag1 and drawing == 0:
                no_draw_frames += 1

            # Grace period of 20 non-drawing frames before exit
            if flag1 and no_draw_frames > 20:
                print("Stopping due to no drawing for 20 frames")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if a:
            # np.save('ursa_major.npy', np.array(a))
            target_points = np.load('ursa_major.npy',allow_pickle=True)
            mtx1, mtx2, disparity = procrustes(resample_points(target_points,50), resample_points(np.array(a),50))
            
            
            x_coords = [p[0] for p in a]
            y_coords = [1 - p[1] for p in a]

            # Lower disparity = more similar
            print(f"Shape similarity score: {1 - disparity:.2f}")

            if 1 - disparity > 0.80:  # threshold (tune this)
                print("🎉 You Win! Pattern matched.")
            else:
                print("❌ Try Again. Pattern mismatch.")
            plt.figure(figsize=(6, 6))
            plt.plot(x_coords, y_coords, marker='o', linewidth=2, color='blue')
            plt.title("Index Finger Drawing Path")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
            plt.axis("equal")
            plt.show()

print(ANSWER)

if __name__ == '__main__':
    main()

