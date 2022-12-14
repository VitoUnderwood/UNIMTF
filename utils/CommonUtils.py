# -*- coding:utf-8 -*-
import datetime

import torch
import random


def setSeed(seed):
    # 结果可复现
    torch.manual_seed(seed)
    random.seed(seed)
    # gpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def getTimestamp():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestamp
