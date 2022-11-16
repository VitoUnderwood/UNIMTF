# -*- coding: utf-8 -*-
import datetime
import os

import torch
from Trainer import Trainer

import utils.MyLogger as MyLogger
from configs.PrefixNlgConfig import PrefixNlgConfig
from utils.common_utils import setSeed, getTimestamp


def main():
    # 获取模型参数
    config = PrefixNlgConfig.getParser()
    # 自定义logger
    logger = MyLogger.init_logger(os.path.join(config.log_path, f"{getTimestamp()}.log"))
    # 固定随机种子
    setSeed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Project Run On Device %s' % device)

    trainer = Trainer(config, logger)
    if config.run_mode == 'train':
        train_dataloader = DataLoader(dataset=MyDataset(self.config.train_file), batch_size=self.config.batch_size,
                                      shuffle=True, drop_last=True, collate_fn=collate_fn)
        dev_dataloader = DataLoader(dataset=MyDataset(self.config.dev_file), batch_size=self.config.batch_size,
                                    shuffle=True, drop_last=True, collate_fn=collate_fn)
        trainer.train()
    elif config.run_mode == 'predict':
        test_dataloader = DataLoader(dataset=MyDataset(self.config.test_file), batch_size=self.config.batch_size,
                                     shuffle=False, drop_last=False, collate_fn=collate_fn)
        model_name = "checkpoints/step2000_latest.pt"
        trainer.infer(model_name)


if __name__ == '__main__':
    main()
