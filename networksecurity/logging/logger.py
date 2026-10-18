import logging, os
from datetime import datetime

# LOGGING SETUP -----------------------------------------------------

def setup_logging(log_dir="logs"):
    """
    Configure the root logger to write to a per-run timestamped file.
    Returns the full path of the log file that was created.
    """
    # Unique filename per run. %Y-%m-%d sorts chronologically.
    log_file = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"

    # The directory that holds the logs -- NOT the file itself.
    logs_path = os.path.join(os.getcwd(), log_dir)
    os.makedirs(logs_path, exist_ok=True)

    # The actual log file lives directly inside logs_path.
    log_file_path = os.path.join(logs_path, log_file)

    # basicConfig is a no-op if the root logger already has handlers,
    # so only call it when nothing has configured logging yet.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            filename=log_file_path,
            format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    return log_file_path


# Run the configuration when this module is imported, and expose a logger
setup_logging()
logger = logging.getLogger(__name__)