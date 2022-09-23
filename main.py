import math

import cv2
import mediapipe as mp


def mosaic(src, ratio=0.05):
    small = cv2.resize(src, None, fx=ratio, fy=ratio,
                       interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)


def mosaicArea(src, x, y, width, height, ratio=0.05):
    h, w, _ = src.shape
    if x < 0:
        x = 0
    if x > w - width - 1:
        x = w - width - 1
    if y < 0:
        y = 0
    if y > h - height - 1:
        y = h - height - 1
    dst = src.copy()
    dst[y:y + height, x:x +
        width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst


def getDegree(x1, y1, x2, y2) -> float:
    radian = math.atan2(y2-y1, x2-x1)
    return radian * 180 / math.pi


if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Pose", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Pose", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    past_landmarks_num = 60  # past frame num
    past_landmarks = [] * past_landmarks_num  # 0:latest

    with mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            frame.flags.writeable = False
            results = pose.process(frame)
            if not results.pose_landmarks:
                continue
            landmarks = results.pose_landmarks.landmark
            past_landmarks.insert(0, landmarks)
            if len(past_landmarks) >= past_landmarks_num:
                del past_landmarks[-1]

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            for i, landmark in enumerate(landmarks):
                if landmark.visibility < 0.5:
                    continue

                if i == 0:  # nose
                    pass

                elif i == 11:  # left shoulder
                    pass

                elif i == 12:  # right shoulder
                    pass

                elif i == 13:  # left elbow
                    pass

                elif i == 14:  # right elbow
                    pass

                elif i == 15:  # left wrist
                    pass

                elif i == 16:  # right wrist
                    pass

            # mosaic
            frame = mosaicArea(
                frame, int(landmarks[0].x * w-128), int(landmarks[0].y * h - 128), 256, 256)

            # nose
            cv2.putText(
                frame,
                f"Nose pos x: {round(landmarks[0].x, 2)} y: {round(landmarks[0].y, 2)} z: {round(landmarks[0].z, 2)}",
                (12, 20), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)

            # body degree
            ls = landmarks[11]  # left shoulder
            rs = landmarks[12]  # right shoulder

            deg = getDegree(rs.x, rs.y, ls.x, ls.y)
            cv2.putText(
                frame,
                f"Body deg {round(deg, 1)} deg",
                (12, 48), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)

            if abs(deg) <= 20:
                cv2.putText(
                    frame,
                    "Center",
                    (int(w/2) - 64, int(h/2) + 36), cv2.FONT_ITALIC, 1, (0, 255, 0), 4)
            elif deg < -20:
                cv2.putText(
                    frame,
                    "Left",
                    (12, int(h/2) + 36), cv2.FONT_ITALIC, 1, (0, 255, 0), 4)
            elif deg > 20:
                cv2.putText(
                    frame,
                    "Right",
                    (int(w) - 128, int(h/2) + 36), cv2.FONT_ITALIC, 1, (0, 255, 0), 4)

            # arms
            l0 = landmarks[13]
            lw = landmarks[15]
            r0 = landmarks[14]
            rw = landmarks[16]

            r_cos = ((rw.x - r0.x) * (rs.x - r0.x) +
                     (rw.y - r0.y) * (rs.y - r0.y)) / (math.sqrt((rw.x - r0.x) ** 2 + (rw.y - r0.y) ** 2) * math.sqrt((rs.x - r0.x) ** 2 + (rs.y - r0.y) ** 2))
            l_cos = ((lw.x - l0.x) * (ls.x - l0.x) +
                     (lw.y - l0.y) * (ls.y - l0.y)) / (math.sqrt((lw.x - l0.x) ** 2 + (lw.y - l0.y) ** 2) * math.sqrt((ls.x - l0.x) ** 2 + (ls.y - l0.y) ** 2))

            cv2.putText(
                frame,
                f"Right Arm Cos {round(r_cos, 2)}",
                (12, 72), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)

            cv2.putText(
                frame,
                f"Left Arm Cos  {round(l_cos, 2)}",
                (12, 96), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)

            if -0.85 < l_cos < 0.85:
                cv2.putText(
                    frame,
                    "Right",
                    (int(w) - 96, int(h/2)), cv2.FONT_ITALIC, 1, (0, 0, 255), 4)
                cv2.putText(
                    frame,
                    "Straight",
                    (int(w) - 96, int(h/2) + 24), cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2)

            if -0.85 < r_cos < 0.85:
                cv2.putText(
                    frame,
                    "Left",
                    (12, int(h/2)), cv2.FONT_ITALIC, 1, (0, 0, 255), 4)
                cv2.putText(
                    frame,
                    "Straight",
                    (12, int(h/2) + 24), cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2)

            landmarks
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    thickness=3, circle_radius=20))

            cv2.imshow("Pose", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
