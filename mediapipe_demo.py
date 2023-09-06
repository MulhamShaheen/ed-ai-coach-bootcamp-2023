import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python import vision
from pprint import pprint
import numpy as np
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(SCRIPT_DIR, "models")
SAMPLE_IMAGES_ROOT = os.path.join(SCRIPT_DIR, "demo_images")
HAND_MODEL_PATH = os.path.join(MODELS_ROOT, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(MODELS_ROOT, "face_landmarker.task")
HAND_IMG_SAMPLE = os.path.join(SAMPLE_IMAGES_ROOT, "hand_image.jpg")
FACE_SAMPLE = os.path.join(SAMPLE_IMAGES_ROOT, "face_image.jpg")


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


def mediapipe_hands_inference_demo() -> None:
    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(HAND_IMG_SAMPLE)
    # STEP 4: Detect hand landmarks from the input image.
    detection_result: HandLandmarkerResult = detector.detect(image)
    pprint(detection_result)
    viz_image = cv2.imread(HAND_IMG_SAMPLE)
    viz_image = VizUtils.draw_hand_detection_result(viz_image, result=detection_result)
    cv2.imshow("demo", viz_image)
    cv2.waitKey(0)


def mediapipe_face_inference_demo() -> None:
    base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(FACE_SAMPLE)
    viz_image = cv2.imread(FACE_SAMPLE)
    # STEP 4: Detect face landmarks from the input image.
    face_detection_result: FaceLandmarkerResult = detector.detect(image)
    face = face_detection_result.face_landmarks[0]
    for point in face:
        viz_image = VizUtils.draw_landmark(viz_image, point)
    cv2.imshow("Face Detection", viz_image)
    cv2.waitKey(0)


def articulation_detection() -> bool:
    pass


def webcam_demo() -> None:
    vid = cv2.VideoCapture(0)
    hand_detector = ModelsFactory.init_hand_model()
    face_detector = ModelsFactory.init_face_model()

    while True:
        ret, bgr_frame = vid.read()
        bgr_frame = bgr_frame[:, ::-1, :]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        hand_detection_result: HandLandmarkerResult = hand_detector.detect(mp_image)
        face_detection_result: FaceLandmarkerResult = face_detector.detect(mp_image)
        rgb_frame = VizUtils.draw_hand_detection_result(
            rgb_frame, result=hand_detection_result
        )
        faces = face_detection_result.face_landmarks
        if faces:
            rgb_frame = VizUtils.draw_landmarks(rgb_frame, faces[0])
        bgr_viz = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", bgr_viz)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    vid.release()


if __name__ == "__main__":
    webcam_demo()
