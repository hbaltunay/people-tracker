import cv2
from dataclasses import dataclass
from typing import Optional


@dataclass
class InfoVideo:
    width: int
    height: int
    fps: int
    total_frames: Optional[int] = None

    @classmethod
    def get_video_writer(cls, source_path: str):
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video at {source_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return InfoVideo(width, height, fps, total_frames)


class GetVideo:

    def __init__(self, source_path: str):
        self.frame_id = 0
        self.source_path = source_path
        self.cap = cv2.VideoCapture(self.source_path)

    def __iter__(self):
        return self

    def __next__(self):
        ret, video_frame = self.cap.read()
        if ret:
            self.frame_id += 1
            return self.frame_id, video_frame
        else:
            self.cap.release()
            raise StopIteration


class ProcessVideo:
    def __init__(self, source_path: str, target_path: str = "./result.mp4", save: bool = False):
        self.source_path = source_path
        self.target_path = target_path
        self.save = save
        self.video_frames = None
        self.out = None
        self.imgsz = None

    def __enter__(self):
        video_info = InfoVideo.get_video_writer(source_path=self.source_path)
        self.video_frames = GetVideo(source_path=self.source_path)
        self.imgsz = (video_info.height, video_info.width)
        if self.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(self.target_path, fourcc, video_info.fps, (video_info.width, video_info.height))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
