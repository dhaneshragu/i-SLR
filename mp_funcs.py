import cv2
import numpy as np
import os
import json
import pandas as pd
import mediapipe as mp

mp_holistic = mp.solutions.holistic # holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color conversion
    image.flags.writeable = False # img no longer writeable
    pred = model.process(image) # make landmark prediction
    image.flags.writeable = True  # img now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # color reconversion
    return image, pred

def draw(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=0))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,150,0), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(200,56,12), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(250,56,12), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
    

def extract_coordinates(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.empty((468, 3))*np.nan
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.empty((33, 3))*np.nan
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.empty((21, 3))*np.nan
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.empty((21, 3))*np.nan
    return np.concatenate([face, lh, pose, rh])


# cap = cv2.VideoCapture(0)
# final_landmarks=[]
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         image, results = mediapipe_detection(frame, holistic)
#         #draw(image, results)
#         landmarks = extract_coordinates(results)
#         #print(len(landmarks))
#         final_landmarks.extend(landmarks)
#         #cv2.imshow('Hello',image)
#         cv2.imshow('Webcam Feed',image)
#         if cv2.waitKey(10) & 0xFF == ord("q"):
#             break

# df1 = pd.DataFrame(final_landmarks,columns=['x','y','z'])
# df1.to_csv("colours.csv",index=False)
