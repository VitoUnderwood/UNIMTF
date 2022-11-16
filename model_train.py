# -*- coding: utf-8 -*-
import os

import torch
from trainer import Trainer

from configs.PrefixNlgConfig import PrefixNlgConfig
import utils.MyLogger as MyLogger
from utils.common_utils import setSeed

def main():
    # 获取模型参数
    config = PrefixNlgConfig.getParser()
    # 自定义logger
    logger = MyLogger.init_logger(os.path.join(config.log_path, "train.log"))
    # 固定随机种子
    setSeed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Project Run On Device %s' % device)

    trainer = Trainer(config, logger)
    if config.run_mode == 'train':
        trainer.train()
    elif config.run_mode == 'predict':
        model_name = "checkpoints/step2000_latest.pt"
        trainer.infer(model_name)


if __name__ == '__main__':
    main()
