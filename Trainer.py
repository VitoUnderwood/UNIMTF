# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import utils
from model.summarizer import Summarizer
from my_dataset import MyDataset, OdpsDataset, collate_fn
from utils import EarlyStopping


class Trainer(object):
    def __init__(self, args, logger):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Summarizer(self.args).to(self.device)
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.loss_fn = torch.nn.BCELoss()
        self.writer = SummaryWriter("./logs")
        self.logger = logger
        self.model.train()

    def train(self):
        early_stopping = EarlyStopping(args=self.args, patience=self.args.patience, verbose=True)

        self.logger.info('Start training...')
        train_dataloader = DataLoader(dataset=MyDataset(self.args.train_file), batch_size=self.args.batch_size,
                                      shuffle=True, drop_last=True, collate_fn=collate_fn)
        dev_dataloader = DataLoader(dataset=MyDataset(self.args.dev_file), batch_size=self.args.batch_size,
                                    shuffle=True, drop_last=True, collate_fn=collate_fn)

        for i in range(self.args.max_epoch):
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
                # if train_step % self.args.save_checkpoint_steps == 0:
                # self.save(train_step)
                if train_step % self.args.train_log_steps == 0:
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

    def infer(self, model_name):
        self.model.load_state_dict(torch.load(model_name))
        print(f"model {model_name} load finish...")
        self.model.eval()
        test_dataloader = DataLoader(dataset=MyDataset(self.args.test_file), batch_size=self.args.batch_size,
                                     shuffle=False, drop_last=False, collate_fn=collate_fn)

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
                        if len(pred_idx) == self.args.max_pred_sents:
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
        save_df.to_csv(self.args.result_save_path, sep='\t', index=False)

    def predict(self, model_name):
        import common_io
        self.args.tables = "odps://kbalgo_dev/tables/jx_text_sum_predict"
        self.args.outputs = "odps://kbalgo_dev/tables/jx_text_sum_predict_outputs"
        reader = common_io.table.TableReader(self.args.tables, selected_cols="text")  # 列名必须是小写
        total_records_num = reader.get_row_count()  # Get total records number, 获得表的总行数
        # ● Read读取操作返回一个python数组，数组中每个元素为表的一行数据组成的一个tuple。
        records = reader.read(total_records_num)
        reader.close()

        processor = Processor(args)

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
                        if len(pred_idx) == self.args.max_pred_sents:
                            break
                    if len(pred_idx) == 0:
                        pred_idx.append(selected_ids[i][0])

                    # pred_idx.sort()
                    results.append(
                        ('[SEP]'.join([str(w) for w in src_strs[i]]), '[SEP]'.join([src_strs[i][p] for p in pred_idx])))

        writer = common_io.table.TableWriter(self.args.outputs)
        writer.write(results, (0, 1))
        writer.close()

    def save(self, step):
        # checkpoint_path = os.path.join(self.args.model_save_path, f'step{step}.pt')
        # self.logger.info("Saving checkpoint %s" % checkpoint_path)
        # if not os.path.exists(checkpoint_path):
        # torch.save(self.model.state_dict(), checkpoint_path)

        checkpoint_path = os.path.join(self.args.model_save_path, f'step{step}_latest.pt')
        torch.save(self.model.state_dict(), checkpoint_path