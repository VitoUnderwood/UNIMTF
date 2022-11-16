# -*- coding: utf-8 -*-
import os
import torch
from configs.prefix_nlg_config import args
from trainer import Trainer
from utils import init_logger

logger = init_logger(os.path.join(args.log_path, "train.log"))

#
# def set_seed(args):
#     # 结果可复现
#     torch.manual_seed(args.seed)
#     random.seed(args.seed)
#
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(args.seed)


if __name__ == '__main__':
    # set_seed(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device %s' % device)
    trainer = Trainer(args, logger)
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'infer':
        model_name = "checkpoints/step2000_latest.pt"
        # model_name = "checkpoints/step748_latest.pt"
        trainer.infer(model_name)
