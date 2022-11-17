# -*- coding:utf-8 -*-

import os

import torch
from torch.utils.data import DataLoader

from configs.DataEnhaceConfig import DataEnhanceConfig
from datasets.DataEnhanceDataset import DataEnhanceDataset, collate_fn
from trainers.DataEnhanceTrainer import Trainer
from utils.CommonUtils import getTimestamp, setSeed
from utils.MyLogger import init_logger


def main():
    # 获取模型参数
    config = DataEnhanceConfig.getParser()
    # 自定义logger
    logger = init_logger(os.path.join(config.log_path, f"{getTimestamp()}.log"))
    # 固定随机种子
    setSeed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Project Run On Device %s' % device)

    trainer = Trainer(config, logger)

    if config.runMode == 'train':
        trainDataloader = DataLoader(dataset=DataEnhanceDataset(config.trainFile), batch_size=config.batch_size,
                                     shuffle=True, drop_last=True, collate_fn=collate_fn)
        devDataloader = DataLoader(dataset=DataEnhanceDataset(config.devFile), batch_size=config.batch_size,
                                   shuffle=True, drop_last=True, collate_fn=collate_fn)
        trainer.train(trainDataloader, devDataloader)
    elif config.runMode == 'predict':
        testDataLoader = DataLoader(dataset=DataEnhanceDataset(config, "test"), batch_size=config.batch_size,
                                    shuffle=False, drop_last=False, collate_fn=collate_fn)
        trainer.predict(config, testDataLoader)


if __name__ == '__main__':
    main()
