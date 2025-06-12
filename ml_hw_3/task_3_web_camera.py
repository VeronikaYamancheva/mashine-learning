import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

MY_FIRST_NAME = 'Veronika'
MY_LAST_NAME = 'Yamancheva'

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def load_owner_embedding(image_path="my_photo.jpg"):
    image = cv2.imread(image_path)
    return DeepFace.represent(image, model_name="Facenet", enforce_detection=False)[0]["embedding"]


def detect_faces(image_rgb, face_detector):
    return face_detector.process(image_rgb).detections


def detect_hands(image_rgb, hands_detector):
    return hands_detector.process(image_rgb).multi_hand_landmarks


def count_fingers(hand_landmarks, handedness_label):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    if handedness_label == "Right":
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    for i in range(1, 5):
        if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)


def is_owner(face_crop, owner_embedding):
    try:
        face_embedding = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        distance = np.linalg.norm(np.array(owner_embedding) - np.array(face_embedding))
        #print(f"Distance: {distance}")
        return distance < 10
    except:
        return False


def get_label_by_fingers(fingers_count, face_crop):
    if fingers_count == 1:
        return "Veronika"
    elif fingers_count == 2:
        return "Yamancheva"
    elif fingers_count == 3:
        try:
            emotion = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)[0]['dominant_emotion']
            return f"Emotion: {emotion}"
        except:
            return "Emotion: ?"
    return "Owner"


def draw_results(frame, x, y, w, h, label, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def main():
    face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    hands_detector = mp_hands.Hands(min_detection_confidence=0.7)
    owner_embedding = load_owner_embedding()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Кадр не получен")
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_detections = detect_faces(image_rgb, face_detector)
        results = hands_detector.process(image_rgb)
        hand_landmarks_list = results.multi_hand_landmarks
        handedness_list = results.multi_handedness
        fingers_count = 0
        if hand_landmarks_list:
            for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
                label = handedness.classification[0].label  # "Left" или "Right"
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers_count = count_fingers(hand_landmarks, label)
                cv2.putText(frame, f'Fingers: {fingers_count}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                break
        if face_detections:
            for detection in face_detections:
                ih, iw, _ = frame.shape
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                x, y = max(0, x), max(0, y)
                face_crop = frame[y:y + h, x:x + w]

                if is_owner(face_crop, owner_embedding):
                    label = get_label_by_fingers(fingers_count, face_crop)
                    color = (0, 255, 0)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                draw_results(frame, x, y, w, h, label, color)

        cv2.imshow("Recognition", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
