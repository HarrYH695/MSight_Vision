from numpy import ndarray
import numpy as np
from msight_vision.base import DetectionResult2D, DetectedObject2D
from .base import ImageDetector2DBase
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List
import cv2

class YoloDetector(ImageDetector2DBase):
    """YOLOv5 detector for 2D images."""

    def __init__(self, model_path: Path, device: str = "cpu", confthre: float = 0.25, nmsthre: float = 0.45, fp16: bool = False, class_agnostic_nms: bool = False, mask_path: Dict[str, Path] = None):
        """
        Initialize the YOLO detector.
        :param model_path: path to the YOLO model
        :param device: device to run the model on (e.g., 'cpu', 'cuda')
        """
        super().__init__()
        self.model = YOLO(str(model_path))
        self.device = device
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.fp16 = fp16
        self.class_agnostic_nms = class_agnostic_nms
        self.mask = {
            key: np.repeat((np.load(item).astype(bool).astype(np.uint8) * 255)[:, :, np.newaxis], 3, axis=2) for key, item in mask_path.items()
        } if mask_path is not None else None

    def convert_yolo_result_to_detection_result(self, yolo_output_results, timestamp, sensor_type):
        """
        Convert YOLO output results to DetectionResult2D.
        :param yolo_output_results: YOLO output results
        :param timestamp: timestamp of the image
        :param sensor_type: type of the sensor
        :return: DetectionResult2D instance
        """
        # Convert YOLO output to DetectionResult2D
        bboxes = yolo_output_results[0].boxes.xyxy.cpu().numpy()
        confs = yolo_output_results[0].boxes.conf.cpu().numpy()
        class_ids = yolo_output_results[0].boxes.cls.cpu().numpy()
        
        detected_objects = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            class_id = int(class_ids[i])
            score = float(confs[i])
            # calculate the center coordinates of the bounding box
            center_x = float((box[0] + box[2]) / 2)
            center_y = float((box[1] + box[3]) / 2)
            # print(class_id)
            detected_object = DetectedObject2D(
                box=[float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                class_id=class_id,
                score=score,
                pixel_bottom_center=[center_x, center_y],
            )
            detected_objects.append(detected_object)
        
        return detected_objects
    
    def detect(self, image: ndarray, timestamp, sensor_type, sensor_name) -> DetectionResult2D:
        if self.mask is not None and sensor_name in self.mask:
            if self.mask[sensor_name].shape[:2] != image.shape[:2]:
                raise ValueError(
                    f"Mask dimensions {self.mask[sensor_name].shape[:2]} do not match image dimensions {image.shape[:2]}"
                )
            else:
                image = cv2.bitwise_and(image, self.mask[sensor_name])
        yolo_output_results = self.model(image, device=self.device, conf=self.confthre, iou=self.nmsthre, half=self.fp16, verbose=False, agnostic_nms=self.class_agnostic_nms)
        ## Convert results to DetectionResult2D
        detection_result = self.convert_yolo_result_to_detection_result(
            yolo_output_results,
            timestamp,
            sensor_type,
        )
        return detection_result
    
class Yolo26Detector(YoloDetector):
    """YOLOv2.6 detector for 2D images."""
    def __init__(self, model_path: Path, device: str = "cpu", confthre: float = 0.25, nmsthre: float = 0.45, fp16: bool = False, class_agnostic_nms: bool = False, mask_path: Dict[str, Path] = None, end2end: bool = False):
        super().__init__(model_path, device, confthre, nmsthre, fp16, class_agnostic_nms, mask_path)

        self.end2end = end2end
    def detect(self, image: ndarray, timestamp, sensor_type, sensor_name) -> DetectionResult2D:
        if self.mask is not None and sensor_name in self.mask:
            if self.mask[sensor_name].shape[:2] != image.shape[:2]:
                raise ValueError(
                    f"Mask dimensions {self.mask[sensor_name].shape[:2]} do not match image dimensions {image.shape[:2]}"
                )
            else:
                image = cv2.bitwise_and(image, self.mask[sensor_name])
        yolo_output_results = self.model(image, device=self.device, conf=self.confthre, iou=self.nmsthre, half=self.fp16, verbose=False, agnostic_nms=self.class_agnostic_nms, end2end=self.end2end)
        ## Convert results to DetectionResult2D
        detection_result = self.convert_yolo_result_to_detection_result(
            yolo_output_results,
            timestamp,
            sensor_type,
        )
        return detection_result

class Yolo26OBBDetector(Yolo26Detector):
    """YOLOv2.6 OBB detector for 2D images."""
    def __init__(self, model_path: Path, device: str = "cpu", confthre: float = 0.25, nmsthre: float = 0.45, fp16: bool = False, class_agnostic_nms: bool = False, mask_path: Dict[str, Path] = None, end2end: bool = False):
        super().__init__(model_path, device, confthre, nmsthre, fp16, class_agnostic_nms, mask_path, end2end)

    def convert_yolo_result_to_detection_result(self, yolo_output_results, timestamp, sensor_type):
        """
        Convert YOLO output results to DetectionResult2D.
        :param yolo_output_results: YOLO output results
        :param timestamp: timestamp of the image
        :param sensor_type: type of the sensor
        :return: DetectionResult2D instance
        """
        # Convert YOLO output to DetectionResult2D
        bboxes = yolo_output_results[0].obb.xyxyxyxy.cpu().numpy()
        confs = yolo_output_results[0].obb.conf.cpu().numpy()
        class_ids = yolo_output_results[0].obb.cls.cpu().numpy()
        
        detected_objects = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            class_id = int(class_ids[i])
            score = float(confs[i])
            # calculate the center coordinates of the bounding box
            center = box.mean(axis=0)
            center_x = float(center[0])
            center_y = float(center[1])

            detected_object = DetectedObject2D(
                box=[float(box[0][0]), float(box[0][1]), float(box[1][0]), float(box[1][1]), float(box[2][0]), float(box[2][1]), float(box[3][0]), float(box[3][1])],
                class_id=class_id,
                score=score,
                pixel_bottom_center=[center_x, center_y],
            )
            detected_objects.append(detected_object)
        
        return detected_objects

class Yolo26OBBPedestrianDetector(Yolo26OBBDetector):
    """YOLOv2.6 OBB pedestrian detector for 2D images."""
    def __init__(self, model_path: Path, camera_center: Dict[str, List[int]], device: str = "cpu", confthre: float = 0.25, nmsthre: float = 0.45, fp16: bool = False, class_agnostic_nms: bool = False, mask_path: Dict[str, Path] = None, end2end: bool = False):
        super().__init__(model_path, device, confthre, nmsthre, fp16, class_agnostic_nms, mask_path, end2end)

        self.camera_center = camera_center
    def convert_yolo_result_to_detection_result(self, yolo_output_results, timestamp, sensor_type, sensor_name):
        """
        Convert YOLO output results to DetectionResult2D.
        :param yolo_output_results: YOLO output results
        :param timestamp: timestamp of the image
        :param sensor_type: type of the sensor
        :param sensor_name: name of the sensor
        :return: DetectionResult2D instance
        """
        # Convert YOLO output to DetectionResult2D
        bboxes = yolo_output_results[0].obb.xyxyxyxy.cpu().numpy()
        confs = yolo_output_results[0].obb.conf.cpu().numpy()
        
        detected_objects = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            score = float(confs[i])
            # calculate the bottom point of the pedestrian
            center = self.predict_bottom_from_obb_box(box, tuple(self.camera_center[sensor_name]))
            center_x = float(center[0])
            center_y = float(center[1])

            detected_object = DetectedObject2D(
                box=[float(box[0][0]), float(box[0][1]), float(box[1][0]), float(box[1][1]), float(box[2][0]), float(box[2][1]), float(box[3][0]), float(box[3][1])],
                class_id=4,
                score=score,
                pixel_bottom_center=[center_x, center_y],
            )
            detected_objects.append(detected_object)
        
        return detected_objects
    
    def detect(self, image: ndarray, timestamp, sensor_type, sensor_name) -> DetectionResult2D:
        if self.mask is not None and sensor_name in self.mask:
            if self.mask[sensor_name].shape[:2] != image.shape[:2]:
                raise ValueError(
                    f"Mask dimensions {self.mask[sensor_name].shape[:2]} do not match image dimensions {image.shape[:2]}"
                )
            else:
                image = cv2.bitwise_and(image, self.mask[sensor_name])
        yolo_output_results = self.model(image, device=self.device, conf=self.confthre, iou=self.nmsthre, half=self.fp16, verbose=False, agnostic_nms=self.class_agnostic_nms, end2end=self.end2end)
        ## Convert results to DetectionResult2D
        detection_result = self.convert_yolo_result_to_detection_result(
            yolo_output_results,
            timestamp,
            sensor_type,
            sensor_name
        )
        return detection_result
    
    def predict_bottom_from_obb_box(self, corners: ndarray, image_center: tuple[int, int]) -> List[float]:
        """Return the pedestrian bottom-center (x, y) given a pedestrian OBB.
        :param corners: The four OBB vertices, ordered around the rectangle.
        :param image_center: (cx, cy) of the source image.

        :return: The bottom-center (x, y) in image coordinates.
        """
        center = corners.mean(axis=0)
        direction = np.array([image_center[0] - center[0],
                              image_center[1] - center[1]], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return [float(center[0]), float(center[1])]
        direction /= norm

        s_best = np.inf
        for i in range(4):
            a = corners[i].astype(np.float64)
            b = corners[(i + 1) % 4].astype(np.float64)
            s = self._ray_segment_intersect(center, direction, a, b)
            if s < s_best:
                s_best = s

        if not np.isfinite(s_best):
            return [float(center[0]), float(center[1])]

        hit = center + s_best * direction
        return [float(hit[0]), float(hit[1])]
    
    def _ray_segment_intersect(self, origin: np.ndarray, direction: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        e = b - a
        denom = direction[0] * e[1] - direction[1] * e[0]   
        if abs(denom) < 1e-12:
            return np.inf                                   
        diff = a - origin
        s = (diff[0] * e[1] - diff[1] * e[0]) / denom      
        t = (diff[0] * direction[1] - diff[1] * direction[0]) / denom
        if s < -1e-9 or t < -1e-9 or t > 1 + 1e-9:
            return np.inf
        return max(s, 0.0)