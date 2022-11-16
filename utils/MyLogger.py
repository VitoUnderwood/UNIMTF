# -*- coding: utf-8 -*-

import logging


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def int_arr_to_str(arr: list):
    arr = [str(i) for i in arr]
    return ' '.join(arr)


def label_to_idx(label_arr: list):
    # 词袋形 label arr，转成 索引位置：[1,0,1,1,0]>>>>>[0,2,3]
    return [i for i, li in enumerate(label_arr) if li == 1]


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params
