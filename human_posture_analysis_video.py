import cv2
import time
import math as m
import mediapipe as mp
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi * theta)
    return degree

# Function to send alert.
def sendWarning():
    pygame.mixer.music.load('mixkit-classic-short-alarm-993.wav')
    pygame.mixer.music.play()

# Constants and Initializations
good_frames = 0
bad_frames = 0
font = cv2.FONT_HERSHEY_SIMPLEX
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def process_frame(image, fps):
    global good_frames, bad_frames
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark

    if lm is not None and lm.landmark is not None:
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
    else:
        print("No landmarks detected in the frame.")
        return image

    if offset < 100:
        cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
    else:
        cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

    neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
    cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
    cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
    cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
    cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
    cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

    angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

    if neck_inclination < 40 and torso_inclination < 10:
        bad_frames = 0
        good_frames += 1
        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)
    else:
        good_frames = 0
        bad_frames += 1
        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

    good_time = (1 / fps) * good_frames
    bad_time = (1 / fps) * bad_frames

    if good_time > 0:
        time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
        cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
    else:
        time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
        cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

    if bad_time > 3:
        sendWarning()

    return image
