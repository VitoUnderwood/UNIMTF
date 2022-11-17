# -*- coding: utf-8 -*-
import datetime
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import utils
from models.prefix_nlg.PrefixNlgModel import PrefixNlgModel
# from my_dataset import MyDataset, OdpsDataset, collate_fn

from utils.EarlyStop import EarlyStopping
from utils.CommonUtils import getTimestamp


class Trainer(object):
    def __init__(self, config, logger):
        self.config = config
        self.model = PrefixNlgModel(config).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = torch.nn.BCELoss()
        self.writer = SummaryWriter(config.summary_path)
        self.logger = logger

    def train(self, train_dataloader, dev_dataloader):
        self.logger.info('Start training...')

        early_stopping = EarlyStopping(config=self.config, patience=self.config.patience, verbose=True)

        for i in range(self.config.max_epoch):
            self.logger.info("-------epoch  {} -------".format(i + 1))
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                src = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                segs = batch[2].to(self.device)
                clss = batch[3].to(self.device)
                mask = batch[4].to(self.device)
                mask_cls = batch[5].to(self.device)

                sent_scores = self.model(src, segs, clss, mask, mask_cls)

                clip_sent_scores = torch.masked_select(sent_scores, mask_cls)
                clip_labels = torch.masked_select(labels, mask_cls)
                loss = self.loss_fn(clip_sent_scores, clip_labels.float())
                # print(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_step = len(train_dataloader) * i + step + 1
                # self.writer.add_scalar("train_loss", loss.item(),train_step)
                # if train_step % self.config.save_checkpoint_steps == 0:
                # self.save(train_step)
                if train_step % self.config.train_log_steps == 0:
                    self.logger.info("train time：{}, Loss: {}".format(train_step, loss.item()))
                    self.writer.add_scalar("train_loss", loss.item(), train_step)

                if train_step % 200 == 0:
                    self.save(train_step)

            self.model.eval()
            dev_loss_list = []
            with torch.no_grad():
                for batch in dev_dataloader:
                    src = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    segs = batch[2].to(self.device)
                    clss = batch[3].to(self.device)
                    mask = batch[4].to(self.device)
                    mask_cls = batch[5].to(self.device)

                    sent_scores = self.model(src, segs, clss, mask, mask_cls)

                    clip_sent_scores = torch.masked_select(sent_scores, mask_cls)
                    clip_labels = torch.masked_select(labels, mask_cls)
                    loss = self.loss_fn(clip_sent_scores, clip_labels.float())

                    dev_loss_list.append(loss.item())

            total_dev_loss = np.average(dev_loss_list)
            self.logger.info("dev dataset：{}, Loss: {}".format(train_step, total_dev_loss))
            early_stopping(total_dev_loss, self.model, train_step)

            if early_stopping.early_stop:
                print("Early stopping")
                self.save(train_step)
                break
                # print(f"dev loss {total_dev_loss}")
                # accuracy = (outputs.argmax(1) == targets).sum()
                # total_accuracy = total_accuracy + accuracy

            # self.logger.info("dev set Loss: {}".format(total_dev_loss))
            # print("dev set accuracy: {}".format(total_accuracy / len(test_data)))
            self.writer.add_scalar("dev_loss", total_dev_loss, i)
            # self.writer.add_scalar("test_accuracy", total_accuracy / len(test_data), i)
            #
            # torch.save(module, "{}/module_{}.pth".format(work_dir, i + 1))
            # print("saved epoch {}".format(i + 1))
        self.writer.close()

    def infer(self, model_name, test_dataloader):
        self.model.load_state_dict(torch.load(model_name))
        print(f"model {model_name} load finish...")
        self.model.eval()
        result_s = {'real_idx': [], 'predict_idx': [], 'src_text': [], 'summarized_text': []}
        save_path = 'result.tsv'
        with torch.no_grad():
            for batch in test_dataloader:
                src = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                segs = batch[2].to(self.device)
                clss = batch[3].to(self.device)
                mask = batch[4].to(self.device)
                mask_cls = batch[5].to(self.device)
                src_strs = batch[6]

                sent_scores = self.model(src, segs, clss, mask, mask_cls)
                # 为了排除pad mask的影响，是的即便是小概率0也能比mask位置的概率大
                sent_scores = sent_scores + mask_cls.float()
                sent_scores = sent_scores.cpu().data.numpy()
                # 从大到小概率对应的位置id,指的是sent socore的位置
                selected_ids = np.argsort(-sent_scores, 1)

                for i, idx in enumerate(selected_ids):
                    pred_idx = []
                    if len(src_strs[i]) == 0:
                        continue
                    for j in selected_ids[i][:len(src_strs[i])]:
                        if j >= len(src_strs[i]):
                            continue
                        # candidate = batch.src_str[i][j].strip()
                        if sent_scores[i][j] >= 1.98:
                            pred_idx.append(j)
                        if len(pred_idx) == self.config.max_pred_sents:
                            break
                    if len(pred_idx) == 0:
                        pred_idx.append(selected_ids[i][0])

                    # pred_idx.sort()

                    result_s['src_text'].append('[SEP]'.join([str(w) for w in src_strs[i]]))
                    result_s['summarized_text'].append('[SEP]'.join([src_strs[i][p] for p in pred_idx]))
                    result_s['predict_idx'].append(utils.int_arr_to_str(pred_idx))
                    label_idx = utils.label_to_idx(labels[i].tolist())
                    result_s['real_idx'].append(utils.int_arr_to_str(label_idx))

        save_df = pd.DataFrame()
        save_df['real_idx'] = result_s['real_idx']
        save_df['predict_idx'] = result_s['predict_idx']
        save_df['src_text'] = result_s['src_text']
        save_df['summarized_text'] = result_s['summarized_text']
        save_df.to_csv(self.config.result_save_path, sep='\t', index=False)

    def predict_online(self, model_name):
        import common_io
        self.config.tables = "odps://kbalgo_dev/tables/jx_text_sum_predict"
        self.config.outputs = "odps://kbalgo_dev/tables/jx_text_sum_predict_outputs"
        reader = common_io.table.TableReader(self.config.tables, selected_cols="text")  # 列名必须是小写
        total_records_num = reader.get_row_count()  # Get total records number, 获得表的总行数
        # ● Read读取操作返回一个python数组，数组中每个元素为表的一行数据组成的一个tuple。
        records = reader.read(total_records_num)
        reader.close()

        processor = Processor(config)

        datasets = []

        for r in records:
            src = r[0]
            ids = []
            # print(src, ids)
            data = processor.preprocess(src, ids)
            if data is None:
                continue
            token_ids, labels, segments_ids, cls_ids, src_txt = data
            data_dict = {"src": token_ids, "labels": labels, "segs": segments_ids, 'clss': cls_ids, 'src_txt': src_txt}
            datasets.append(data_dict)

        test_dataloader = DataLoader(dataset=OdpsDataset(datasets), batch_size=32,
                                     shuffle=False, drop_last=False, collate_fn=collate_fn)

        self.model.load_state_dict(torch.load(model_name))
        print(f"model {model_name} load finish...")
        self.model.eval()

        results = []
        with torch.no_grad():
            for batch in test_dataloader:
                src = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                segs = batch[2].to(self.device)
                clss = batch[3].to(self.device)
                mask = batch[4].to(self.device)
                mask_cls = batch[5].to(self.device)
                src_strs = batch[6]

                sent_scores = self.model(src, segs, clss, mask, mask_cls)
                # 为了排除pad mask的影响，是的即便是小概率0也能比mask位置的概率大
                sent_scores = sent_scores + mask_cls.float()
                sent_scores = sent_scores.cpu().data.numpy()
                # 从大到小概率对应的位置id,指的是sent socore的位置
                selected_ids = np.argsort(-sent_scores, 1)

                for i, idx in enumerate(selected_ids):
                    pred_idx = []
                    if len(src_strs[i]) == 0:
                        continue
                    for j in selected_ids[i][:len(src_strs[i])]:
                        if j >= len(src_strs[i]):
                            continue
                        # candidate = batch.src_str[i][j].strip()
                        if sent_scores[i][j] >= 0.95:
                            pred_idx.append(j)
                        if len(pred_idx) == self.config.max_pred_sents:
                            break
                    if len(pred_idx) == 0:
                        pred_idx.append(selected_ids[i][0])

                    # pred_idx.sort()
                    results.append(
                        ('[SEP]'.join([str(w) for w in src_strs[i]]), '[SEP]'.join([src_strs[i][p] for p in pred_idx])))

        writer = common_io.table.TableWriter(self.config.outputs)
        writer.write(results, (0, 1))
        writer.close()

    def save(self, step):
        checkpoint_path = os.path.join(self.config.model_save_path, f'step{step}_{getTimestamp()}.pt')
        torch.save(self.model.state_dict(), checkpoint_path)