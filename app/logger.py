import os
import logging


class Logger:
    def __init__(self, log_name):
        self.log_name = log_name
        self.log_dir = "logs"
        self.file_name = f"{log_name}.log"
        self.ensure_log_dir_exists()

    def ensure_log_dir_exists(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def get_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            log_path = os.path.join(self.log_dir, self.file_name)
            file_handler = logging.FileHandler(log_path)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger
