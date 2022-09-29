import math
import os
import random
import subprocess

import cv2
import mediapipe as mp


def playSound(path):
    if os.name != 'nt':
        subprocess.Popen(["aplay", "--quiet", path])
    else:
        subprocess.Popen(
            ["powershell", "-c", f"(New-Object Media.SoundPlayer '{path}').PlaySync();"])


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

    blDebug = False
    past_landmarks_num = 10  # past frame num
    past_landmarks = [] * past_landmarks_num  # 0:latest
    past_poses = [] * past_landmarks_num  # 0:latest
    pose_list = ["bl", "br", "ls", "rs"]
    pose_sheet = random.choices(pose_list, k=20)
    pose_sheet_index = 0
    pose_phase = False
    frame_cnt = 0
    score = 0

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
            frame_cnt += 1

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            poseTmp = None

            # nose
            nose = landmarks[0]

            if nose.visibility > 0.5:
                # mosaic
                frame = mosaicArea(
                    frame, int(nose.x * w-128), int(nose.y * h - 128), 256, 256)

                # nose
                cv2.putText(
                    frame,
                    f"Nose pos x: {round(nose.x, 2)} y: {round(nose.y, 2)} z: {round(nose.z, 2)}",
                    (12, 20), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)

            # body degree
            ls = landmarks[11]  # left shoulder
            rs = landmarks[12]  # right shoulder

            if ls.visibility > 0.5 and rs.visibility > 0.5:
                deg = getDegree(rs.x, rs.y, ls.x, ls.y)
                cv2.putText(
                    frame,
                    f"Body deg {round(deg, 1)} deg",
                    (12, 48), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)

                if abs(deg) <= 20:
                    if blDebug:
                        cv2.putText(
                            frame,
                            "Center",
                            (int(w/2) - 64, int(h/2) + 36), cv2.FONT_ITALIC, 1, (0, 255, 0), 4)
                    poseTmp = "bc"
                elif deg < -20:
                    if blDebug:
                        cv2.putText(
                            frame,
                            "Left",
                            (12, int(h/2) + 36), cv2.FONT_ITALIC, 1, (0, 255, 0), 4)
                    poseTmp = "bl"
                elif deg > 20:
                    if blDebug:
                        cv2.putText(
                            frame,
                            "Right",
                            (int(w) - 128, int(h/2) + 36), cv2.FONT_ITALIC, 1, (0, 255, 0), 4)
                    poseTmp = "br"

            # arms
            l0 = landmarks[13]
            lw = landmarks[15]
            r0 = landmarks[14]
            rw = landmarks[16]

            r_cos = ((rw.x - r0.x) * (rs.x - r0.x) +
                     (rw.y - r0.y) * (rs.y - r0.y)) / (math.sqrt((rw.x - r0.x) ** 2 + (rw.y - r0.y) ** 2) * math.sqrt((rs.x - r0.x) ** 2 + (rs.y - r0.y) ** 2))
            l_cos = ((lw.x - l0.x) * (ls.x - l0.x) +
                     (lw.y - l0.y) * (ls.y - l0.y)) / (math.sqrt((lw.x - l0.x) ** 2 + (lw.y - l0.y) ** 2) * math.sqrt((ls.x - l0.x) ** 2 + (ls.y - l0.y) ** 2))

            if r0.visibility > 0.5 and rw.visibility > 0.5:
                cv2.putText(
                    frame,
                    f"Right Arm Cos {round(r_cos, 2)}",
                    (12, 72), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)

                if -0.85 < l_cos < 0.85:
                    if blDebug:
                        cv2.putText(
                            frame,
                            "Right",
                            (int(w) - 96, int(h/2)), cv2.FONT_ITALIC, 1, (0, 0, 255), 4)
                        cv2.putText(
                            frame,
                            "Straight",
                            (int(w) - 96, int(h/2) + 24), cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2)
                    poseTmp = "rs"

            if l0.visibility > 0.5 and lw.visibility > 0.5:
                cv2.putText(
                    frame,
                    f"Left Arm Cos  {round(l_cos, 2)}",
                    (12, 96), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)

                if -0.85 < r_cos < 0.85:
                    if blDebug:
                        cv2.putText(
                            frame,
                            "Left",
                            (12, int(h/2)), cv2.FONT_ITALIC, 1, (0, 0, 255), 4)
                        cv2.putText(
                            frame,
                            "Straight",
                            (12, int(h/2) + 24), cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2)
                    poseTmp = "ls"

            past_poses.insert(0, poseTmp)
            if len(past_poses) >= past_landmarks_num:
                cntDict = {}
                for ps in past_poses:
                    if ps in cntDict:
                        cntDict[ps] += 1
                    else:
                        cntDict[ps] = 1

                cntMax = 0
                poseMax = ""
                for ps, cnt in cntDict.items():
                    if cntMax < cnt:
                        cntMax = cnt
                        poseMax = ps

                cv2.putText(
                    frame,
                    f"[{poseMax}] ({cntMax}/{past_landmarks_num} frames)",
                    (int(w/2) - 128, int(h) - 36), cv2.FONT_ITALIC, 0.6, (255, 0, 0), 2)

                if frame_cnt % 50 == 0 and pose_sheet_index < len(pose_sheet):
                    if pose_phase is False:
                        ps = pose_sheet[pose_sheet_index]
                        pose_phase = True
                        playSound(f"sound/{ps}.wav")
                    else:
                        ps = pose_sheet[pose_sheet_index]
                        pose_sheet_index += 1
                        pose_phase = False
                        if ps == poseMax:
                            playSound("sound/correct.wav")
                            score += 100
                        else:
                            playSound("sound/wrong.wav")

            cv2.putText(
                frame,
                f"Score: {score}",
                (int(w/2) - 48, int(h) - 72), cv2.FONT_ITALIC, 0.8, (0, 0, 255), 2)

            if len(past_landmarks) >= past_landmarks_num:
                del past_landmarks[-1]
            if len(past_poses) >= past_landmarks_num:
                del past_poses[-1]

            if blDebug:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        thickness=3, circle_radius=20))

            cv2.imshow("Pose", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
