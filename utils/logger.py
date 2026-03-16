import os
import datetime


class Logger:
    """
    Simple experiment logger.
    """

    def __init__(self, log_dir="logs", experiment_name="experiment"):
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(
            log_dir,
            f"{experiment_name}_{timestamp}.log"
        )

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {msg}"

        print(formatted)

        with open(self.filepath, "a") as f:
            f.write(formatted + "\n")


# Backward-compatible function
_default_logger = Logger()


def log(msg):
    _default_logger.log(msg)