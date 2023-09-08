import cv2
import os
import mediapipe as mp
import torch
import torch.nn.functional as F
from deep_head_pose_lite import stable_hopenetlite
from torchvision import transforms
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
from moviepy.editor import AudioFileClip, VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(SCRIPT_DIR, "models")
SAMPLE_IMAGES_ROOT = os.path.join(SCRIPT_DIR, "demo_images")
HAND_MODEL_PATH = os.path.join(MODELS_ROOT, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(MODELS_ROOT, "face_landmarker.task")
SEGMENT_MODEL_PATH = os.path.join(MODELS_ROOT, "selfie_segmenter.tflite")
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
    
    @staticmethod
    def init_direction_model():
        pos_net = stable_hopenetlite.shufflenet_v2_x1_0()  
        saved_state_dict = torch.load('deep_head_pose_lite/model/shuff_epoch_120.pkl',
                                      map_location="cpu")   
        pos_net.load_state_dict(saved_state_dict, strict=False)  
        pos_net.eval()

        return pos_net
    
    @staticmethod
    def init_segmentation_model()-> vision.ImageSegmenter:
        base_options = python.BaseOptions(model_asset_path=SEGMENT_MODEL_PATH)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )
        segmenter = vision.ImageSegmenter.create_from_options(options)
        return segmenter


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
    
    @staticmethod
    def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
        y1 = size * (np.cos(pitch) * np.sin(roll) + np.sin(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
        y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (np.sin(yaw)) + tdx
        y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(255,0,0),3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(0,0,255),2)

        return img
    
class Utils:
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    @staticmethod
    def preprocess_direction_input(image, bbox_data) -> torch.Tensor:
        x_left, y_top = bbox_data['x_left'], bbox_data['y_top']
        x_right, y_bottom = bbox_data['x_right'], bbox_data['y_bottom']
    
        image = image[y_top:y_bottom, x_left:x_right,:]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = Utils.transformations(img)
        img = torch.unsqueeze(img, 0)

        return img
    
    @staticmethod
    def get_bbox_coords(orig_img, detect_results):
        kpoints = detect_results.face_landmarks[0]
        x_s = [dot.x for dot in kpoints]
        y_s = [dot.y for dot in kpoints]

        x_s.sort()
        y_s.sort()

        height, width = orig_img.shape[:2]
        x_left, x_right = int(x_s[0] * width), int(x_s[-1] * width)
        y_top, y_bottom = int(y_s[0] * height), int(y_s[-1] * height)

        bbox_data = {
            'x_left': x_left,
            'y_top': y_top,
            'x_right': x_right,
            'y_bottom': y_bottom,
            'tdx': x_left + (x_right - x_left) / 2,
            'tdy':  y_top + (y_bottom - y_top) / 2
        }
        return bbox_data
    
    @staticmethod
    def postprocess_direction_output(yaw, pitch, roll):
        idx_tensor = torch.FloatTensor(list(range(66)))

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)

        yaw_predicted = torch.sum(yaw_predicted.data * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data * idx_tensor) * 3 - 99

        return yaw_predicted, pitch_predicted, roll_predicted
    
    @staticmethod
    def count_background_std(category_mask, orig_img_npy):
        category_mask = category_mask.numpy_view()
        category_mask = category_mask[:, :, np.newaxis].copy() 
        category_mask = np.repeat(category_mask, 3, axis=2)

        bool_mask = np.array(category_mask, dtype=bool)
        bool_mask_flatten = bool_mask[:,:,0].flatten()

        result = bool_mask * orig_img_npy

        res = []
        for chanel in range(3):
            flat_chan = result[:,:,chanel].flatten()
            flat_chan = flat_chan * bool_mask_flatten
            res.append(np.std(flat_chan))
        return np.ndarray(res)


# def mediapipe_hands_inference_demo() -> None:
#     base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
#     options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
#     detector = vision.HandLandmarker.create_from_options(options)

#     # STEP 3: Load the input image.
#     image = mp.Image.create_from_file(HAND_IMG_SAMPLE)
#     # STEP 4: Detect hand landmarks from the input image.
#     detection_result: HandLandmarkerResult = detector.detect(image)
#     pprint(detection_result)
#     viz_image = cv2.imread(HAND_IMG_SAMPLE)
#     viz_image = VizUtils.draw_hand_detection_result(viz_image, result=detection_result)
#     cv2.imshow("demo", viz_image)
#     cv2.waitKey(0)


# def mediapipe_face_inference_demo() -> None:
#     base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
#     options = vision.FaceLandmarkerOptions(
#         base_options=base_options,
#         output_face_blendshapes=True,
#         output_facial_transformation_matrixes=True,
#         num_faces=1,
#     )
#     detector = vision.FaceLandmarker.create_from_options(options)

#     # STEP 3: Load the input image.
#     image = mp.Image.create_from_file(FACE_SAMPLE)
#     viz_image = cv2.imread(FACE_SAMPLE)
#     # STEP 4: Detect face landmarks from the input image.
#     face_detection_result: FaceLandmarkerResult = detector.detect(image)

#     with open('face_detection_result.pkl', 'wb') as f:
#         pickle.dump(face_detection_result, f)

#     face = face_detection_result.face_landmarks[0]
#     for point in face:
#         viz_image = VizUtils.draw_landmark(viz_image, point)
#     cv2.imwrite('result.jpg', viz_image)
#     # cv2.imshow("Face Detection", viz_image)
#     # cv2.waitKey(0)



def mediapipe_save_video_demo(NAME) -> None:
    VIDEO_PATH = os.path.join(SAMPLE_IMAGES_ROOT, f"{NAME}.mp4")

    std_list = []
    direction_list = []


    hand_detector = ModelsFactory.init_hand_model()
    face_detector = ModelsFactory.init_face_model()
    direction_detector = ModelsFactory.init_direction_model()
    segmenter = ModelsFactory.init_segmentation_model()

    video = cv2.VideoCapture(VIDEO_PATH)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    for frame_idx in tqdm(range(frame_count)):
        _, frame_cv2 = video.read()

        rgb_frame = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        hand_detection_result: HandLandmarkerResult = hand_detector.detect(mp_image)
        face_detection_result: FaceLandmarkerResult = face_detector.detect(mp_image)
        segmentation_result = segmenter.segment(mp_image)

        if len(face_detection_result.face_landmarks) != 0:
            bbox_data = Utils.get_bbox_coords(frame_cv2, face_detection_result)
            img_tensor = Utils.preprocess_direction_input(frame_cv2, bbox_data)
            yaw, pitch, roll = direction_detector(img_tensor)
            yaw, pitch, roll = Utils.postprocess_direction_output(yaw, pitch, roll)
            direction_list.append(np.ndarray((yaw, pitch, roll)))
            
            rgb_std = Utils.count_background_std(segmentation_result.category_mask, rgb_frame)
            std_list.append(rgb_std)


            rgb_frame = VizUtils.draw_axis(rgb_frame, yaw, pitch, roll,
                                           bbox_data['tdx'], bbox_data['tdy'], size=200)
            rgb_frame = VizUtils.draw_hand_detection_result(
                rgb_frame, result=hand_detection_result
            )
            frame_cv2 = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            frame_cv2 = cv2.rectangle(frame_cv2, (bbox_data['x_left'], bbox_data['y_top']),
                                    (bbox_data['x_right'], bbox_data['y_bottom']),
                                    (0,255,0), 2)
        
        cv2.imwrite(f'./RESULTS/{frame_idx}.jpg', frame_cv2)
    video.release()

    frames_list = [f'./RESULTS/{frame_idx}.jpg'
                   for frame_idx
                   in range(frame_count)]

    swapped_video = ImageSequenceClip(frames_list, fps=fps)
    swapped_video.write_videofile('OUTPUT.mp4',
                                  audio_codec='aac',
                                  verbose=False,
                                  logger=None)
    
    direction_np = np.stack(direction_list, axis=0)
    std_np = np.stack(std_list, axis=0)
    np.savez(f'{NAME}.npz', direction_np=direction_np, std_np=std_np)



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
    name = 'Макаров'
    mediapipe_save_video_demo(name)

    # mediapipe_mask_inference_demo()
    
