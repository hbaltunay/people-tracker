from abc import ABC, abstractmethod
from ultralytics import YOLO


class Model(ABC):
    def __init__(self, model_path: str, task: str):
        self.model = None
        self.model_path = model_path
        self.task = task

    @abstractmethod
    def get_model(self):
        pass


class YOLOModel(Model):
    def __init__(self, model_path, task):
        super().__init__(model_path, task)
        self.get_model()

    def get_model(self):
        self.model = YOLO(self.model_path)

    def __call__(self, frame):
        if self.task == "Count":
            return self.model.predict(frame, classes=[0])
        elif self.task == "Track":
            return self.model.track(frame, persist=True, classes=[0])
