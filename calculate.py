import cv2
import numpy as np
from collections import defaultdict
from numpy import ndarray
from typing import Tuple


class ProcessCalculate:
    def __init__(self):
        """
        Calculations required for processes.
        """
        self.frame_id = None
        self.detections = None
        self.in_detect = defaultdict(lambda: defaultdict(set))
        self.num_detect = dict()
        self.area_counts = dict()
        self.removed_detect = list()
        self.removed_num = list()
        self._mask_info = {"1": ["in", "out"], "2": ["out", "in"]}

    @staticmethod
    def line_shift(areas: dict, imgsz: tuple) -> dict:
        """
        Areas are created from the specified line.

        Args:
            areas (dict): Coordinates of the areas.
            imgsz (tuple): The size of the video frame.
        Returns:
            dict: Coordinates of the processed areas.
        """
        processed_areas = defaultdict(dict)
        aspect_ratio = imgsz[0] / imgsz[1]
        dist_factor = aspect_ratio * 100
        ang_factor = aspect_ratio * 50

        for name, coord in areas.items():
            x1, y1, x2, y2 = coord[0][0], coord[0][1], coord[1][0], coord[1][1]

            direction_vector = np.array([x2 - x1, y2 - y1])
            unit_vector = direction_vector / (np.linalg.norm(direction_vector) + 1e-9)
            perpendicular_vector = np.array([-unit_vector[1], unit_vector[0]])

            dist_vector = dist_factor * perpendicular_vector
            ang_vector = ang_factor * perpendicular_vector

            a = np.array([x1, y1]) + dist_vector + np.array([-ang_vector[1], ang_vector[0]])
            b = np.array([x2, y2]) + dist_vector + np.array([ang_vector[1], -ang_vector[0]])
            c = np.array([x1, y1]) - dist_vector + np.array([-ang_vector[1], ang_vector[0]])
            d = np.array([x2, y2]) - dist_vector + np.array([ang_vector[1], -ang_vector[0]])

            processed_areas[name]["1"] = np.array([[x1, y1], a, b, [x2, y2]], dtype=np.int32)
            processed_areas[name]["2"] = np.array([[x1, y1], c, d, [x2, y2]], dtype=np.int32)

        return dict(processed_areas)
    
    @staticmethod
    def area_coord_sorted(points: ndarray) -> ndarray:
        """
        Area coordinates sorted.

        Args:
            points (ndarray): Coordinates of the area.
        Returns:
            ndarray: Returns the sorted coordinates.
        """
        ox, oy = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - oy, points[:, 0] - ox)
        sorted_indices = np.argsort(angles)
        return points[sorted_indices]
    
    def get_area_mask(self, areas: dict, imgsz: tuple, task: str) -> dict:
        """
        Image masks of the areas are created.

        Args:
            areas (dict): Coordinates of the areas.
            imgsz (tuple): The size of the video frame.
            task (str): Computer vision task type.
        Returns:
            dict: Returns a masked image of the area to be detected.
        """
        if task == "Count":
            masks = dict()
            for i, points in enumerate(areas.values(), 1):
                frame = np.zeros(imgsz)
                points = self.area_coord_sorted(np.array(points))
                masks["area" + str(i)] = cv2.fillPoly(frame, [points], color=[1])
            return masks

        elif task == "Track":
            masks = dict()
            for i, points in enumerate(areas.values(), 1):
                frame = np.zeros(imgsz)
                for j, pts in enumerate(points.values()):
                    pts = self.area_coord_sorted(pts)
                    frame = cv2.fillPoly(frame, [pts], color=[j + 1])
                masks["area" + str(i)] = frame

            return masks

    def create_area_counts(self, masks: dict) -> None:
        """
        Keeping information about the area to be processed.

        Args:
            masks (dict): The areas to be processed.
        """
        for key in masks.keys(): 
            self.area_counts[key] = 0
        self.area_counts = dict(self.area_counts)

    def create_num_detect(self, masks: dict) -> None:
        """
        Keeping information about the area to be processed.
        
        Args:
            masks (dict): The areas to be processed.
        """
        for key in masks.keys():
            self.num_detect[key] = {}
            for name in ["in", "out"]:
                self.num_detect[key][name] = 0
        self.num_detect = dict(self.num_detect)
        
    @staticmethod
    def get_detections(results, task):
        """
        Creates a dictionary containing the detections.

        Args:
            results (-): Results of the model.
            task (str): Computer vision task type.
        Returns:
            dict: Returns the results to be used in dictionary form.
        """
        detections = dict()
        detections["xyxy"] = results.boxes.xyxy.cpu().numpy()
        detections["xywh"] = results.boxes.xywh.cpu().numpy()
        detections["confidence"] = results.boxes.conf.cpu().numpy()
        detections["class_id"] = results.boxes.cls.cpu().numpy().astype(int)

        if task == "Track":
            detections["track_id"] = results.boxes.id.cpu().numpy().astype(int)

        return detections

    def count_area(self, masks: dict) -> Tuple[dict, dict]:
        """
        The number of detections within the area is calculated.

        Args:
            masks (dict): Mask image of the area to be detected.
        Returns:
            Tuple[dict, dict]: Detection results.
        """
        self.create_area_counts(masks)
        in_detect = defaultdict(lambda: defaultdict(list))

        for key, mask in masks.items():
            for xyxy, xywh in zip(self.detections["xyxy"], self.detections["xywh"]):
                x, y, _, _ = xyxy
                if mask[y][x] == 1:
                    self.area_counts[key] += 1
                    in_detect[key]["xyxy"].append(xyxy)
                    in_detect[key]["xywh"].append(xywh)

        return self.area_counts, dict(in_detect)

    def track_area(self, masks: dict) -> tuple[dict, dict, dict]:
        """
        The number of detections passing through the border area is calculated.

        Args:
            masks (dict): The area to be processed.
        Returns:
            Tuple[dict, dict, dict]: Detection results.
        """
        detect_xyxy = defaultdict(dict)
        
        for name, mask in masks.items():
            detect_xyxy[name]["xyxy"] = self.detections["xyxy"]
            detect_xyxy[name]["xywh"] = self.detections["xywh"]
            for xywh, tid in zip(self.detections["xywh"], self.detections["track_id"]):
                x, y, _, _ = xywh.astype(np.int32)
                if mask[y][x]:
                    f_name, s_name = self._mask_info[str(int(mask[y][x]))]
                    self.in_detect[name][f_name].add(tid)
                    if tid in self.in_detect[name]["passed"] and tid in self.in_detect[name][s_name]:
                        self.in_detect[name]["passed"].remove(tid)
                        self.in_detect[name][s_name].remove(tid)
                    elif tid in self.in_detect[name][s_name]:
                        self.in_detect[name]["passed"].add(tid)
                        self.in_detect[name][s_name].remove(tid)

                else:
                    if tid in self.in_detect[name]["in"] and tid in self.in_detect[name]["passed"]:
                        self.num_detect[name]["out"] += 1
                        self.in_detect[name]["in"].remove(tid)
                        self.in_detect[name]["passed"].remove(tid)
                    elif tid in self.in_detect[name]["out"] and tid in self.in_detect[name]["passed"]:
                        self.num_detect[name]["in"] += 1
                        self.in_detect[name]["out"].remove(tid)
                        self.in_detect[name]["passed"].remove(tid)
                    else:
                        if tid in self.in_detect[name]["in"]:
                            self.in_detect[name]["in"].remove(tid)
                        elif tid in self.in_detect[name]["out"]:
                            self.in_detect[name]["out"].remove(tid)

        return self.num_detect, self.in_detect, dict(detect_xyxy)

    @staticmethod
    def to_center_base(detections: dict) -> None:
        """
        The detection center is set to the base.

        Args:
            detections (dict): Dictionary of detection results to be used.
        """
        arr = np.array(detections["xyxy"])
        x = (arr[:, 0] + arr[:, 2]) // 2
        y = (arr[:, 1] + arr[:, 3]) // 2
        c = (arr[:, 3] - arr[:, 1]) // 2

        res = np.column_stack((x, y + c, x, y + c))
        detections["xyxy"] = res.astype(int)
    