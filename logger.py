import logging
import datetime
import os
from logging.handlers import TimedRotatingFileHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a formatter
format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create a folder called logs if not already created
folder_name = "logs"
# current_directory = os.getcwd()
current_directory = os.path.dirname(os.path.abspath(__file__))
logs_folder_path = os.path.join(current_directory, folder_name)
if not os.path.exists(logs_folder_path):
    os.mkdir(logs_folder_path)

# Create file handlers
current_day = datetime.datetime.now().strftime('%Y-%m-%d')
log_filename = f"{current_day}.log"
log_filename = os.path.join(logs_folder_path, log_filename)
f_handler = TimedRotatingFileHandler(log_filename, when='D', interval=1)
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(format)

# Create console handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_handler.setFormatter(format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)