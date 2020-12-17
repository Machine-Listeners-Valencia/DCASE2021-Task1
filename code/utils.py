import os
from datetime import datetime
import config

__authors__ = "Javier Naranjo, Sergi Perez and Irene Martín"
__copyright__ = "Machine Listeners Valencia"
__credits__ = ["Machine Listeners Valencia"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Javier Naranjo"
__email__ = "janal2@alumni.uv.es"
__status__ = "Dev"
__date__ = "2020"


def create_folder_time():
    home_path = os.getenv('HOME')
    folder_path = home_path + config.outputs_path
    dt = datetime.today().strftime('%Y-%m-%d-%H:%M')
    os.mkdir(folder_path + dt)

    return folder_path + dt + '/'


def moving_config_file_to_folder(folder_path):
    home_path = os.getenv('HOME')

    with open(home_path + config.code_path + 'config.py', 'r') as f:
        with open(folder_path + 'config.txt', 'w') as f1:
            for line in f:
                f1.write(line)
