# -*- coding:utf-8 -*-
import argparse
import os


# from configs import Config


class PrefixNlgConfig:
    seed = 110

    def __init__(self):
        args = self.getParser()
        for key in args.__dict__.keys():
            if key in self.__dict__.keys():
                self.__dict__[key] = args.__dict__[key]

    @staticmethod
    def getParser():
        parser = argparse.ArgumentParser()
        # 模型相关路径配置
        prefix_path = os.getcwd()
        parser.add_argument("-prefix_path", type=str, default=prefix_path)
        parser.add_argument("-log_path", type=str, default=os.path.join(prefix_path, "checkpoints/prefix_nlg/log"))
        parser.add_argument("-model_path", type=str, default=os.path.join(prefix_path, "checkpoints/prefix_nlg/checkpoint"))
        parser.add_argument("-train_file", type=str, default=os.path.join(prefix_path, "data/train.json"))
        parser.add_argument("-dev_file", type=str, default=os.path.join(prefix_path, "data/dev.json"))
        parser.add_argument("-test_file", type=str, default=os.path.join(prefix_path, "data/test.json"))
        parser.add_argument("-pretrained_model_path", type=str, default=os.path.join(prefix_path, "pretrained_model/bert-base-chinese"))
        parser.add_argument("-predict_save_path", type=str, default=os.path.join(prefix_path, "checkpoints/prefix_nlg/predict"))
        parser.add_argument("-summary_path", type=str, default=os.path.join(prefix_path, "checkpoints/prefix_nlg/summary"))

        # 模型训练参数配置
        parser.add_argument("-seed", type=int, default=20220908)
        parser.add_argument("-run_mode", type=str, default="train", choices=["train", "predict"])
        parser.add_argument("-learning_rate", type=float, default="1e-5")
        parser.add_argument("-batch_size", type=int, default=4)
        parser.add_argument("-encoder", type=str, default="classifier", choices=["classifier"])
        parser.add_argument("-max_epoch", default=100, type=int)
        parser.add_argument("-patience", default=100, type=int)
        parser.add_argument("-max_pred_sents", default=2, type=int)
        parser.add_argument("-save_checkpoint_steps", default=1000, type=int)
        parser.add_argument("-train_steps", default=40000, type=int)
        parser.add_argument("-train_log_steps", default=10, type=int)

        # 模型结构
        parser.add_argument("-hidden_size", type=int, default=768)

        # 模型超参数配置
        parser.add_argument("-min_sents", default=3, type=int)
        parser.add_argument("-max_sents", default=100, type=int)
        parser.add_argument("-min_tokens", default=3, type=int)
        parser.add_argument("-max_tokens", default=150, type=int)
        parser.add_argument("-max_position_embeddings", default=512, type=int)
        # odps 参数
        parser.add_argument("--tables", type=str, default="odps://kbalgo_dev/tables/jx_text_sum_predict", help="ODPS input table names")
        parser.add_argument("--outputs", type=str, default="odps://kbalgo_dev/tables/jx_text_sum_predict_outputs", help="ODPS output table names")
        args = parser.parse_args()
        return args
