import os
import sys
import time
from datetime import datetime

class Logger:
    # logger = Logger("", __name__)
    def __init__(self, log_dir, runner_name):
        self.runner_name = runner_name

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.runner_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        self.terminal = sys.stdout
        self.log = open(log_path, "a")
        
        self.write(f"====== {self.runner_name} started ======\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self.start_time = time.time()

    def __del__(self):
        elapsed_time = time.time() - self.start_time
        self.write(f"\n====== {self.runner_name} finished ======\nElapsed time: {elapsed_time:.2f} sec\n")
        self.close()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    #==== Write functions ====

    # logger.write("message")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()


    def timer_start(self):
        self.timer_start_time = time.time()
    def timer_end(self, message=""):
        elapsed_time = time.time() - self.timer_start_time
        message = f"{message}  Elapsed time: {elapsed_time:.2f} sec\n"
        self.write(message)


    def timestamp(self, message=""):
        message = f"{message}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        self.write(message)