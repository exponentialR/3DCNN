import os
from pathlib import Path
from trainer_factory import TrainerFactory

from utils import get_logger
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import configparser
import argparse
import time
import sys


def train():
    logger = get_logger(__name__)
    parent_dir = os.path.dirname(Path(os.getcwd()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', choices=['train', 'test', 'tune', 'resume'], default='train')
    args = parser.parse_args()

    seed_everything(42, workers=True)
    tb_logger = TensorBoardLogger('training-logs', name='Experimental3DCNN')
    configfile = configparser.ConfigParser()
    configfile.read('config.ini')

    configfile_head = configfile['hyperparameters']
    data_BS = int(configfile_head['batch_size'])
    learning_rate = float(configfile_head['lr'])
    epochs = int(configfile_head['epoch'])

    model_folder = os.path.join(parent_dir, 'OUTPUT')
    os.makedirs(model_folder, exist_ok=True) if not os.path.exists(model_folder) else None
    model_name = f'EXPERIMENTAL3DCNN-{epochs}-{learning_rate}-{data_BS}'
    trainer_factory = TrainerFactory(args, configfile, configfile_head, tb_logger, logger, model_name, model_folder, parent_dir)
    start_time = time.time()

