import cv2
import mediapipe as mp

# Initialize MediaPipe Pose for body detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load clothing image (Make sure it has a transparent background)
clothing = cv2.imread("clothing.png", cv2.IMREAD_UNCHANGED)

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using Pose model
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        # Extract key points for alignment (e.g., shoulders)
        h, w, _ = frame.shape
        landmarks = result.pose_landmarks.landmark
        left_shoulder = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)
        right_shoulder = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w)

        # Resize clothing to fit between shoulders
        clothing_resized = cv2.resize(clothing, (right_shoulder - left_shoulder, clothing.shape[0]))

        # Overlay clothing image on frame (need alpha blending for realistic effect)
        frame[left_shoulder:left_shoulder + clothing_resized.shape[0], :] = clothing_resized

    cv2.imshow("Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
