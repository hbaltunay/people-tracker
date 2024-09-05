import cv2
from input import SetInput
from utils import CFGRead, JSONFile
from model import YOLOModel
from draw import ProcessDraw
from video import ProcessVideo
from calculate import ProcessCalculate
from numpy import ndarray


class CVTask:
    def __init__(self, cfg: JSONFile, model_path: str, save: bool) -> None:
        """
        Computer Vision Task.

        Args:
            cfg (JSONFile): cfg.
            model_path (str): Path to the model file.
            save (bool): Recording the processed video.
        """
        self.cfg = cfg
        self.model_path = model_path
        self.save = save
        self.calc = ProcessCalculate()
        self.draw = ProcessDraw()
        self.areas = None

    def frame_show_save(self, frame, process):
        cv2.imshow("", frame)
        if self.save:
            process.out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return True

    def process_frame(self, frame, zones, results, frame_id):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError


class PeopleCounter(CVTask):
    def __init__(self, cfg: JSONFile, model_path: str, save: bool) -> None:
        """
        People Counter.

        Args:
            cfg (JSONFile): cfg.
            model_path (str): Path to the model file.
            save (bool): Recording the processed video.
        """
        super().__init__(cfg, model_path, save)

    def process_frame(self, frame: ndarray, masks: dict, results, frame_id: int) -> ndarray:
        """
        Frame processed.

        Args:
            frame (ndarray): Video frame to be processed.
            masks (dict): Areas to be estimated in video frames.
            results (-): Results of the model.
            frame_id (int): Frame number of the video.
        Returns:
            ndarray : Processed video frame.
        """
        self.calc.frame_id = frame_id
        self.calc.detections = self.calc.get_detections(results, self.cfg.task)
        self.calc.to_center_base(self.calc.detections)
        area_counts, in_detect = self.calc.count_area(masks)

        frame = self.draw.draw_elips(frame, in_detect)
        frame = self.draw.draw_area(frame, self.cfg.areas, self.cfg.task)
        frame = self.draw.draw_info(frame, area_counts)

        return frame

    def process(self):
        """Process"""
        with ProcessVideo(source_path=self.cfg.video_path, save=self.save) as process:
            model = YOLOModel(self.model_path, self.cfg.task)
            masks = self.calc.get_area_mask(self.cfg.areas, process.imgsz, self.cfg.task)
            self.calc.create_area_counts(masks)

            for frame_id, frame in process.video_frames:
                results = model(frame)
                frame = self.process_frame(frame, masks, results[0], frame_id)
                if self.frame_show_save(frame, process):
                    break
                    
                    
class PeopleTracker(CVTask):
    def __init__(self, cfg, model_path, save):
        """
        People Tracker.

        Args:
            cfg (JSONFile): cfg.
            model_path (str): Path to the model file.
            save (bool): Recording the processed video.
        """
        super().__init__(cfg, model_path, save)

    def process_frame(self, frame: ndarray, masks: dict, results, frame_id: int) -> ndarray:
        """
        Frame processed.

        Args:
            frame (ndarray): Video frame to be processed.
            masks (dict): Areas to be estimated in video frames.
            results (-): Results of the model.
            frame_id (int): Frame number of the video.
        Returns:
            ndarray : Processed video frame.
        """
        self.calc.frame_id = frame_id
        self.calc.detections = self.calc.get_detections(results, self.cfg.task)
        num_detect, in_detect, detect_xyxy = self.calc.track_area(masks)
        frame = self.draw.draw_boxes(frame, detect_xyxy)
        frame = self.draw.draw_area(frame, self.areas, self.cfg.task)
        frame = self.draw.draw_track_info(frame, num_detect)

        return frame
    
    def process(self):
        """Process"""
        with ProcessVideo(source_path=self.cfg.video_path, save=self.save) as process:
            model = YOLOModel(self.model_path, self.cfg.task)
            self.areas = self.calc.line_shift(self.cfg.areas, process.imgsz)
            masks = self.calc.get_area_mask(self.areas, process.imgsz, self.cfg.task)
            self.calc.create_num_detect(masks)

            for frame_id, frame in process.video_frames:
                results = model(frame)
                frame = self.process_frame(frame, masks, results[0], frame_id)
                if self.frame_show_save(frame, process):
                    break


class PeopleTrackingMonitor:
    def __init__(self, cfg_path: str, model_path: str, save: bool = False) -> None:
        """
        People Tracking Monitor.

        Args:
            cfg_path (str): Path to cfg file.
            model_path (str): Path to the model file.
            save (bool): Recording the processed video.
        """
        self.cfg_path = cfg_path
        self.model_path = model_path
        self.save = save
        self.cfg = None
        self._tasks = {
            "Count": PeopleCounter,
            "Track": PeopleTracker
        }
        self.input = SetInput()

    @staticmethod
    def cfg_read(file_path: str) -> JSONFile:
        """
        Cfg file read.

        Args:
            file_path (str): Path to cfg file.
        Returns:
            JSONFile: cfg.
        """
        cfg_read = CFGRead(file_path=file_path)
        return cfg_read.read_file()

    def cfg_write(self, file_path: str, areas: list) -> None:
        """
        Cfg file write.

        Args:
            file_path (str): Path to cfg file.
            areas (list): Areas.
        """
        self.cfg.get_cfg_dict["areas"] = {}
        for i, area in enumerate(areas, 1):
            self.cfg.get_cfg_dict["areas"][f"area{i}"] = area

        cfg_read = CFGRead(file_path=file_path)
        cfg_read.write_file(self.cfg.get_cfg_dict)

    def process(self):
        """Process"""
        self.cfg = self.cfg_read(self.cfg_path)
        areas = self.input.process(self.cfg.video_path, self.cfg.task)
        if areas:
            self.cfg_write(self.cfg_path, areas)

        people_process = self._tasks[self.cfg.task](cfg=self.cfg, model_path=self.model_path, save=self.save)
        people_process.process()
