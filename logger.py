import os

class Logger():
    def __init__(self):
        self.log_file = None
        self.log_dir = None

    def log(self, str):
        if self.log_file is not None:
            self.log_file.write(str)

    def set_logdir(self, log_dir):
        self.log_dir = log_dir

    def set_filename(self, filename):
        self.log_file = open(os.path.join('./', filename), 'a')

    def close(self):
        if self.log_file is not None:
            self.log_file.close()
