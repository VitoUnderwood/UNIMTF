# -*- coding:utf-8 -*-
import argparse
import os
from configs import Config


class PrefixNlgConfig(Config):
    @staticmethod
    def getParser():
        parser = argparse.ArgumentParser()
        # 模型相关路径配置
        prefix_path = os.getcwd()
        parser.add_argument("-prefix_path", default=prefix_path, type=str)
        parser.add_argument("-train_file", default='./data/test2.json', type=str)
        parser.add_argument("-dev_file", default='./data/test2.json', type=str)
        parser.add_argument("-test_file", default='./data/test2.json', type=str)
        parser.add_argument("-bert_path", default='./bert-base-chinese', type=str)
        parser.add_argument("-model_save_path", default='checkpoints')
        parser.add_argument("-result_save_path", default='result.tsv')
        parser.add_argument("-model_load_path", default='checkpoints/checkpoints')
        parser.add_argument('-log_path', default='logs/')
        # 模型训练参数配置
        parser.add_argument('-seed', default=20220908, type=int)
        parser.add_argument("-mode", default='infer', type=str, choices=['train', 'infer', 'predict'])
        parser.add_argument("-batch_size", default=4, type=int)
        parser.add_argument("-hidden_size", default=768, type=int)
        parser.add_argument("-encoder", default='classifier', type=str, choices=['classifier'])

        parser.add_argument('-max_epoch', default=100, type=int)
        parser.add_argument('-patience', default=100, type=int)
        parser.add_argument('-max_pred_sents', default=2, type=int)
        parser.add_argument("-save_checkpoint_steps", default=1000, type=int)
        # 最多训练次数
        parser.add_argument("-train_steps", default=40000, type=int)
        parser.add_argument("-train_log_steps", default=10, type=int)

        # 模型超参数配置
        # processor
        # 最小句子量，文章不能低于3句话
        parser.add_argument('-min_sents', default=3, type=int)
        # 最大句子量，文章超过100句话
        parser.add_argument('-max_sents', default=100, type=int)
        # 句子最短长度
        parser.add_argument('-min_tokens', default=3, type=int)
        # 句子最大长度
        parser.add_argument('-max_tokens', default=150, type=int)
        parser.add_argument('-max_position_embeddings', default=512, type=int)
        # odps 参数
        parser.add_argument('--tables', default="", type=str, help='ODPS input table names')
        parser.add_argument('--outputs', default="", type=str, help='ODPS output table names')
        args = parser.parse_args()
        return args
