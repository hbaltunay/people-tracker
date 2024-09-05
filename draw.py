import cv2
import numpy as np


class ProcessDraw:
    def __init__(self):
        pass

    @staticmethod
    def draw_elips(frame, in_detect):
        """
        :param frame: Video frame to be processed.
        :param in_detect: Detections in the area.
        :return: Returns the processed frame.
        """
        for key in in_detect.keys():
            for i in range(len(in_detect[key]["xyxy"])):
                x1, y1, _, _ = in_detect[key]["xyxy"][i].astype(int)
                color = (0, 0, 255)

                frame = cv2.ellipse(
                    img=frame,
                    center=(x1, y1 - 10),
                    axes=((in_detect[key]["xywh"][i][-2] / 2).astype(int),
                          (in_detect[key]["xywh"][i][-1] / 8).astype(int)),
                    angle=10,
                    startAngle=0,
                    endAngle=360,
                    color=color,
                    thickness=2,
                )

        return frame

    @staticmethod
    def draw_boxes(frame, in_detect):
        """
        :param frame: Video frame to be processed.
        :param in_detect: Detections in the area.
        :return: Returns the processed frame.
        """
        for key in in_detect.keys():
            for i in range(len(in_detect[key]["xyxy"])):
                x1, y1, x2, y2 = in_detect[key]["xyxy"][i].astype(int)
                x, y, _, _ = in_detect[key]["xywh"][i].astype(int)
                color = (0, 0, 255)

                cv2.rectangle(img=frame,
                              pt1=(x1, y1),
                              pt2=(x2, y2),
                              color=color,
                              thickness=2)

                cv2.circle(frame, (x, y), 4, color, -1)

        return frame

    @staticmethod
    def draw_info(frame, area_counts):
        """
        :param frame: Video frame to be processed.
        :param area_counts: Number of people in the area.
        :return: Returns the processed frame.
        """
        n = len(area_counts)
        cv2.rectangle(frame, (0, 0), (300, 60 * (n + 1)), color=(255, 0, 0), thickness=-1)
        cv2.putText(frame, "People Tracking", (25, 50), 2, 1, (255, 255, 255), 2)

        for i, (name, count) in enumerate(area_counts.items(), 1):
            text = name.upper() + ":   " + str(count)
            cv2.putText(frame, text, (50, 50 * (i + 1)),
                        2, 1, (255, 255, 255), 2, bottomLeftOrigin=False)
        return frame

    @staticmethod
    def draw_track_info(frame, num_detect):
        """
        :param frame: Video frame to be processed.
        :param num_detect: Number of people passing through entrance and exit.
        :return: Returns the processed frame.
        """
        n = len(num_detect) + 1
        cv2.rectangle(frame, (0, 0), (300, 70 + (n * 70)), color=(255, 0, 0), thickness=-1)
        cv2.putText(frame, "People Tracking", (25, 50), 2, 1, (255, 255, 255), 2)
        cv2.putText(frame, "IN  OUT", (155, 100), 2, 1, (255, 255, 255), 1)
        for i, (key, value) in enumerate(num_detect.items(), 1):
            text = key.upper()
            cv2.putText(frame, text, (10, 70 * (i + 1)),
                        2, 1, (255, 255, 255), 1, bottomLeftOrigin=False)
            for j, count in enumerate(value.values(), 1):
                cv2.putText(frame, str(count), (80 + 80 * j, 70 * (i + 1)),
                            2, 1, (255, 255, 255), 2, bottomLeftOrigin=False)

        return frame

    @staticmethod
    def draw_area(frame, points, task):
        """
        :param frame: Video frame to be processed.
        :param points: Coordinates of the polygon.
        :param task: Computer vision task type.
        :return: Returns the processed frame.
        """
        for point in points.values():
            if task == "Count":
                cv2.polylines(
                    frame, [np.array(point)], isClosed=True, color=(0, 255, 0), thickness=2
                )
            else:
                for pt in point.values():
                    cv2.polylines(
                        frame, [np.array(pt)], isClosed=True, color=(0, 255, 0), thickness=2
                    )
        return frame
