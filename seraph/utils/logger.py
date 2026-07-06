import os
import sys
import time
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs", runner_name = "runner", mode="discord"):
        self.runner_name = runner_name

        self.mode = mode
        if self.mode == "discord":
            import requests
            self.requests = requests

        self.terminal = sys.stdout
        self.write(f"====== {self.runner_name} started ======\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self.start_time = time.time()

    def __del__(self):
        elapsed_time = time.time() - self.start_time
        self.write(f"\n====== {self.runner_name} finished ======\nElapsed time: {elapsed_time:.2f} sec\n")

    def flush(self):
        self.terminal.flush()


    #==== Write functions ====

    # logger.write("message")
    def write(self, message):
        self.terminal.write(message)
        self.flush()


    def warn(self, message):
        message = f"[WARNING] {message}\n"
        self.write(message)

    def timer_start(self):
        self.timer_start_time = time.time()
    def timer_end(self, message=""):
        elapsed_time = time.time() - self.timer_start_time
        message = f"{message}  Elapsed time: {elapsed_time:.2f} sec\n"
        self.write(message)


    def timestamp(self, message=""):
        message = f"{message}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        self.write(message)