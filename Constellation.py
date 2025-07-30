import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pyautogui
from scipy.spatial import procrustes
from scipy.interpolate import interp1d
from pynput.mouse import Button, Controller
mouse = Controller()
screen_width, screen_height = pyautogui.size()

question_list = ["question/question1.png", "question/question2.png","question/question3.png","question/question4.png","question/question5.png","question/question6.png"]
answer_list = ["answer/answer1.png", "answer/answer2.png", "answer/answer3.png", "answer/answer4.png", "answer/answer5.png", "answer/answer6.png"]
target_list = ["target/target1.npy", "target/target2.npy", "target/target3.npy", "target/target4.npy", "target/target5.npy", "target/target6.npy"]
min_accuracy_list = [0.8,0.8,0.8,0.8,0.8,0.8]

map = 6  # this is used to select the constellation
mode = 0 # if mode = 1 means saving the data in target, otherwise using the target for accuracy calculation

QUESTION = question_list[map-1]
ANSWER = answer_list[map-1]
TARGET = target_list[map-1]
WRONG = "wrong.png"
MIN_ACCURACY = min_accuracy_list[map-1]

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,   
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    max_num_hands=1
)


def resample_points(points, num_points=50):
    if len(points) < 2:
        raise ValueError("Need at least 2 points to resample")

    diffs = np.diff(points, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))

    if np.any(dists == 0):
        raise ValueError("Duplicate points in drawing detected")

    cumulative_dist = np.concatenate([[0], np.cumsum(dists)])

    try:
        f_interp = interp1d(cumulative_dist, points, axis=0, kind='linear')
        new_distances = np.linspace(0, cumulative_dist[-1], num_points)
        new_points = f_interp(new_distances)
    except Exception as e:
        raise ValueError(f"Interpolation failed: {e}")

    if np.any(np.isnan(new_points)) or np.any(np.isinf(new_points)):
        raise ValueError("Resampled points contain NaN or inf")

    return new_points

def get_angle(a=(0.5,1),b=(0.5,0.5),c=(0.5,0.5)):
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
    if len(landmark_list)>=14:
        if get_angle(landmark_list[0],landmark_list[10],landmark_list[12]) < 90 and get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) > 90:
        # if get_angle(landmark_list[13],landmark_list[14],landmark_list[16]) < 90 and get_angle(landmark_list[9],landmark_list[10],landmark_list[12]) < 90 and get_angle(landmark_list[17],landmark_list[18],landmark_list[20]) < 90 and get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) > 90:
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
        no_draw_frames = 0  
        question_img = cv2.imread(QUESTION)
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
                a.append([index_finger_tip.x, index_finger_tip.y])
            elif drawing == 2:
                no_draw_frames += 1
            elif flag1 and drawing == 0:
                no_draw_frames += 1

            if flag1 and no_draw_frames > 20:
                print("Stopping due to no drawing for 20 frames")
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if a:
            if mode == 1:
                np.save(TARGET, np.array(a))
            else:
                target_points = np.load(TARGET,allow_pickle=True)
                mtx1, mtx2, disparity = procrustes(resample_points(target_points,50), resample_points(np.array(a),50))
                accuracy = 1 - disparity

                print(f"Shape similarity score: {accuracy:.2f}")
                if accuracy > MIN_ACCURACY:  # threshold (tune this)
                    print("🎉 You Win! Pattern matched.")
                    answer = cv2.imread(ANSWER)
                    cv2.waitKey(1)
                else:
                    answer = cv2.imread(WRONG)
                cv2.imshow("FullScreen", answer)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

