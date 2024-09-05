import cv2
import numpy as np


class SetInput:
    def __init__(self):
        self.area = list()
        self.total_area = list()
        self.task = None
        self.frame = None
        self.processed_frame = None
        self.draw = True

    def get_first_frame(self, source_path):
        """
        :param: Video file path.
        :return: First video frame.
        """
        cap = cv2.VideoCapture(source_path)
        _, self.frame = cap.read()
        cap.release()
        return self.frame

    def draw_areas(self, sing_areas):

        areas = self.total_area + sing_areas

        for area in areas:
            for x, y in area:
                self.processed_frame = cv2.circle(img=self.processed_frame,
                                                  center=(x, y),
                                                  radius=10,
                                                  color=(0, 255, 0),
                                                  thickness=-1)

            cv2.polylines(img=self.processed_frame,
                          pts=[np.array([area])],
                          isClosed=True,
                          color=(255, 0, 0),
                          thickness=3)

        cv2.putText(img=self.processed_frame,
                    text="Press Q to save and exit.",
                    org=(25, self.frame.shape[0] - 25),
                    fontFace=1,
                    fontScale=3,
                    color=(0, 0, 255),
                    thickness=3)

    def click_event(self, event, x, y, flags, params):

        if event == cv2.EVENT_LBUTTONUP and flags == cv2.EVENT_FLAG_CTRLKEY:
            if self.area:
                self.total_area.append(self.area)
                self.draw_areas(self.total_area)
                self.area = list()

        elif event == cv2.EVENT_RBUTTONUP and flags == cv2.EVENT_FLAG_CTRLKEY:
            if self.total_area:
                self.total_area.pop()
                self.processed_frame = self.frame.copy()
                self.draw_areas(self.total_area)

        elif event == cv2.EVENT_LBUTTONUP:
            self.area.append((x, y))
            self.processed_frame = self.frame.copy()
            self.draw = True

        elif event == cv2.EVENT_RBUTTONUP:
            if self.area:
                self.area.pop()
            self.processed_frame = self.frame.copy()
            self.draw = True

        if self.task == "Track":
            self.area = self.area[-2:]

        if self.draw:
            self.draw_areas([self.area])
            self.draw = False

        cv2.imshow("image", self.processed_frame)

    def process(self, source_path, task):
        self.task = task
        self.frame = self.get_first_frame(source_path)
        self.processed_frame = self.frame.copy()
        cv2.imshow('image', self.frame)
        cv2.setMouseCallback("image", self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.total_area
