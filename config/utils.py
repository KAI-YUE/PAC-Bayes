import os
import logging
import numpy as np
import pickle
import datetime

# PyTorch libraries
import torch
import torch.nn as nn

import copy
from collections import OrderedDict

def init_logger(config, output_dir, seed=0, attach=True):
    """Initialize a logger object. 
    """
    log_level = "INFO"
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    fh = logging.FileHandler(os.path.join(output_dir, "main.log"))
    fh.setLevel(log_level)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)

    logger.addHandler(fh)
    if attach:
        logger.addHandler(sh)
    logger.info("-"*80)
    logger.info("Run with seed {:d}.".format(seed))

    attributes = filter(lambda a: not a.startswith('__'), dir(config))
    for attr in attributes:
        logger.info("{:<20}: {}".format(attr, getattr(config, attr)))

    return logger


def init_outputfolder(config):
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)

    current_time = datetime.datetime.now()
    current_time_str = datetime.datetime.strftime(current_time, '%m%d_%H%M')

    output_dir = os.path.join(config.output_folder, current_time_str)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config.output_dir = output_dir

    return output_dir


def save_record(record, output_dir):
    # current_path = os.path.dirname(__file__)
    # current_time = datetime.datetime.now()
    # current_time_str = datetime.datetime.strftime(current_time ,'%H_%M_%S')
    # file_name = config.record_dir.format(current_time_str)
    with open(os.path.join(output_dir, "record.dat"), "wb") as fp:
        pickle.dump(record, fp)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_record():
    record = {}
    
    record["train_loss"] = []
    record["train_acc"] = []
    record["test_loss"] = []
    record["test_acc"] = []

    return record

def load_idx(config):
    if os.path.exists(config.idx_path):

        with open(config.idx_path, "rb") as fp:
            idx = pickle.load(fp)
    else:
        idx = {"lowscore_sensitive_idx": np.arange(50000),
               "highscore_sensitive_idx": np.arange(50000),
               "insensitive_idx": np.arange(50000), 
               }

    return idx


