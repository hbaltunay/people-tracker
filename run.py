from core import PeopleTrackingMonitor

if __name__ == "__main__":
    cfg_path = "cfg/track_area.json"
    model_path = "models/yolov8s.pt"

    pcounter = PeopleTrackingMonitor(cfg_path=cfg_path, model_path=model_path, save=False)
    pcounter.process()
