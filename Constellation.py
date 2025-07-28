import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from scipy.spatial import procrustes
from scipy.interpolate import interp1d

screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
draw = mp.solutions.drawing_utils

question_list = ["question1.png", "question2.png"]
answer_list = ["answer1.png", "answer2.png"]
target_list = ["target1.npy", "target2.npy"]

def resample_points(points, num_points=50):
    diffs = np.diff(points, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative_dist = np.concatenate([[0], np.cumsum(dists)])
    f_interp = interp1d(cumulative_dist, points, axis=0, kind='linear')
    new_distances = np.linspace(0, cumulative_dist[-1], num_points)
    return f_interp(new_distances)

def get_angle(a, b, c):
    radian = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    return np.abs(np.degrees(radian))

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        tip = processed.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        tip.y = tip.y / 0.7
        return tip
    return None

def is_drawing(landmark_list):
    if len(landmark_list) >= 21:
        angles = [
            get_angle(landmark_list[13], landmark_list[14], landmark_list[16]),
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]),
            get_angle(landmark_list[17], landmark_list[18], landmark_list[20]),
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8])
        ]
        if angles[0] < 50 and angles[1] < 50 and angles[2] < 50 and angles[3] > 50:
            return 1
        return 0
    return 2

def move_mouse(tip):
    if tip:
        x = int(tip.x * screen_width)
        y = int(tip.y * screen_height)
        pyautogui.moveTo(x, y)

def run_session(mode, question_img_path, answer_img_path, target_npy_path):
    cap = cv2.VideoCapture(0)
    a = []
    flag1 = False
    no_draw_frames = 0

    question_img = cv2.imread(question_img_path)
    question_img = cv2.resize(question_img, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(frameRGB)

        landmark_list = []
        if processed.multi_hand_landmarks:
            hand = processed.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)
            landmark_list = [(lm.x, lm.y) for lm in hand.landmark]

        tip = find_finger_tip(processed)
        if tip:
            move_mouse(tip)

        drawing = is_drawing(landmark_list)
        if drawing == 1:
            flag1 = True
            no_draw_frames = 0
            a.append([tip.x, tip.y])
        elif drawing == 2 or (flag1 and drawing == 0):
            no_draw_frames += 1

        # Combine webcam feed and question image
        frame_resized = cv2.resize(frame, (640, 480))
        combined = cv2.hconcat([question_img, frame_resized])
        cv2.imshow("Question + Webcam", combined)

        if no_draw_frames > 20:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if a:
        if mode == 1:
            np.save(target_npy_path, np.array(a))
        else:
            user_drawn = np.array(a)
            target = np.load(target_npy_path, allow_pickle=True)
            _, _, disparity = procrustes(resample_points(target, 50), resample_points(user_drawn, 50))

            if 1 - disparity > 0.7:
                result_img = cv2.imread(answer_img_path)
                print("🎉 You Win! Pattern matched.")
            else:
                result_img = cv2.imread("wrong.png")
                print("❌ Try Again. Pattern mismatch.")

            cv2.namedWindow("Result", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Result", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    mode = int(input("Enter mode (1 = Save Data, 2 = Match Pattern): "))
    while True:
        print("\nUrsa Major -> 1\nCassiopeia -> 2")
        map_choice = int(input("Choose constellation: "))
        if map_choice not in [1, 2]:
            print("Invalid input. Try again.")
            continue

        QUESTION = question_list[map_choice - 1]
        ANSWER = answer_list[map_choice - 1]
        TARGET = target_list[map_choice - 1]

        # Run one session (webcam + drawing + checking)
        run_session(mode, QUESTION, ANSWER, TARGET)

        print("Press any key to retry the same question...")
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
