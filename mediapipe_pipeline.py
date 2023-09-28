from report_generation import PipelineOutputFileStructure
import cv2
import os
import mediapipe as mp
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python import vision
from pprint import pprint
import numpy as np
from typing import List
from tqdm import tqdm
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(SCRIPT_DIR, "models")
SAMPLE_IMAGES_ROOT = os.path.join(SCRIPT_DIR, "demo_images")
HAND_MODEL_PATH = os.path.join(MODELS_ROOT, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(MODELS_ROOT, "face_landmarker.task")
# HAND_IMG_SAMPLE = os.path.join(SAMPLE_IMAGES_ROOT, "hand_image.jpg")
HAND_IMG_SAMPLE = os.path.join(SAMPLE_IMAGES_ROOT, "partial_hand.jpg")
FACE_SAMPLE = os.path.join(SAMPLE_IMAGES_ROOT, "face_image.jpg")
VIDEO_PATH = os.path.join(SAMPLE_IMAGES_ROOT, "test_video.mp4")


class ModelsFactory:
    @staticmethod
    def init_face_model() -> vision.FaceLandmarker:
        base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        return detector

    @staticmethod
    def init_hand_model() -> vision.HandLandmarker:
        base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        detector = vision.HandLandmarker.create_from_options(options)

        return detector


class VizUtils:
    @staticmethod
    def draw_landmark(image: np.ndarray, landmark: NormalizedLandmark) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        norm_x = landmark.x
        norm_y = landmark.y
        img_x = image_width * norm_x
        img_y = image_height * norm_y
        center_point = (int(img_x), int(img_y))
        image = cv2.circle(image, center_point, 2, (0, 255, 0), 2)
        return image

    @staticmethod
    def draw_landmarks(
        image: np.ndarray, landmarks: List[NormalizedLandmark]
    ) -> np.ndarray:
        for landmark in landmarks:
            image = VizUtils.draw_landmark(image, landmark)
        return image

    @staticmethod
    def draw_hand_detection_result(
        image: np.ndarray, result: HandLandmarkerResult
    ) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        hands_landmarks: List[List[NormalizedLandmark]] = result.hand_landmarks
        for hand_points in hands_landmarks:
            for hand_point in hand_points:
                norm_x = hand_point.x
                norm_y = hand_point.y
                img_x = image_width * norm_x
                img_y = image_height * norm_y
                center_point = (int(img_x), int(img_y))
                image = cv2.circle(image, center_point, 2, (0, 255, 0), 2)
        return image


def save_face_output(
    video_path: str, pipeline_structure: PipelineOutputFileStructure
) -> None:
    face_detector = ModelsFactory.init_face_model()
    video = cv2.VideoCapture(video_path)
    perframe_face_detections = []
    face_output = {"video_height": 0, "video_width": 0}
    index = 0
    while True:
        ret, bgr_frame = video.read()
        if not ret:
            break
        if index % 100 == 0:
            print(index)
        bgr_frame = bgr_frame[:, ::-1, :]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        face_detection_result: FaceLandmarkerResult = face_detector.detect(mp_image)
        face_landmarks: List[
            List[NormalizedLandmark]
        ] = face_detection_result.face_landmarks
        perframe_face_detections.append(face_landmarks)
        index += 1
    face_output["perframe_faces"] = perframe_face_detections
    print(len(perframe_face_detections))
    with open(pipeline_structure.perframe_face_landmaks, "wb") as f:
        pickle.dump(face_output, f)


def save_hands_output(
    video_path: str, pipeline_structure: PipelineOutputFileStructure
) -> None:
    hand_detector = ModelsFactory.init_hand_model()
    video = cv2.VideoCapture(video_path)
    perframe_hands_detections = []
    hands_output = {"video_height": 0, "video_width": 0}
    index = 0
    while True:
        ret, bgr_frame = video.read()
        if not ret:
            break
        if index % 100 == 0:
            print(index)
        bgr_frame = bgr_frame[:, ::-1, :]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        hand_detection_result: HandLandmarkerResult = hand_detector.detect(mp_image)
        hands_landmarks: List[
            List[NormalizedLandmark]
        ] = hand_detection_result.hand_landmarks
        perframe_hands_detections.append(hands_landmarks)
        index += 1
    hands_output["perframe_hands"] = perframe_hands_detections
    print(len(perframe_hands_detections))
    with open(pipeline_structure.perframe_hands_landmarks, "wb") as f:
        pickle.dump(hands_output, f)


if __name__ == "__main__":
    pipeline_output = PipelineOutputFileStructure(
        "/home/andrey/AS/dev/BootCampHak/test_pipelines/pipeline1"
    )
    # video_path = "/home/andrey/AS/dev/BootCampHak/sample_video.webm"
    # save_hands_output(video_path, pipeline_output)
    # save_face_output(video_path, pipeline_output)
    with open(pipeline_output.perframe_hands_landmarks, "rb") as f:
        data = pickle.load(f)
    print(data.keys())
    print(len(data["perframe_hands"][0][0]))
