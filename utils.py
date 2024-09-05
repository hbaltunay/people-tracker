import json


class JSONFile:
    def __init__(self):
        self.cfg_dict = None

    def read(self, file_path):
        with open(file_path, "r") as reader:
            self.cfg_dict = json.load(reader)
            reader.close()

    @staticmethod
    def write(file_path, cfg_dict):
        with open(file_path, "w") as writer:
            json.dump(cfg_dict, writer, ensure_ascii=False, indent=2)

    @property
    def video_path(self):
        return self.cfg_dict["video_name"]

    @property
    def areas(self):
        return self.cfg_dict["areas"]

    @property
    def task(self):
        return self.cfg_dict["task"]

    @property
    def get_cfg_dict(self):
        return self.cfg_dict


class CFGRead:
    def __init__(self, file_path):
        self.file_path = file_path
        self._files = {
            "json": JSONFile()
        }

    @staticmethod
    def get_type(file_path):
        return file_path.split(".")[-1]

    def read_file(self):
        file_type = self.get_type(self.file_path)
        reader = self._files.get(file_type)
        if reader:
            reader.read(self.file_path)
            return reader
        else:
            raise "Unsupported file type."

    def write_file(self, cfg):
        file_type = self.get_type(self.file_path)
        writer = self._files.get(file_type)
        if writer:
            writer.write(self.file_path, cfg)
            return writer
        else:
            raise "Unsupported file type."
