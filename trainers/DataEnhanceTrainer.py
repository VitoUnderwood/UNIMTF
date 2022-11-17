# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from models.DataEnhance.DataEnhace import DataEnhance
from utils.CommonUtils import getTimestamp
from utils.EarlyStop import EarlyStopping


# from my_dataset import MyDataset, OdpsDataset, collate_fn


class Trainer(object):
    def __init__(self, config, logger):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DataEnhance(config).to(self.device)
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

    def predict(self, config, testDataLoader):
        for step, batch in enumerate(testDataLoader):
            srcTokens = batch[0]
            inputIds = batch[1].to(self.device)
            attentionMask = batch[2].to(self.device)
            tokenTypeIds = batch[3].to(self.device)

            with torch.no_grad():
                predictResult = self.model(inputIds, attentionMask, tokenTypeIds).logits.argmax(-1).tolist()

            tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_path)
            # maskToken = (inputIds == tokenizer.mask_token_id)
            # predictedToken = torch.masked_select(logits.argmax(axis=-1), maskToken)
            # print(tokenizer.decode(predictedToken))
            for src, pre in zip(srcTokens, predictResult):
                srcPaddingText = "".join(src)
                trueLength = 0
                srcText = ""
                for w in tokenizer.tokenize(srcPaddingText):
                    if w != "[PAD]":
                        trueLength = trueLength + 1
                        srcText = srcText + w
                print(srcText)
                sent = "".join(tokenizer.decode(pre, clean_up_tokenization_spaces=False).split()[1:trueLength+1])
                print(sent)

    def save(self, step):
        checkpoint_path = os.path.join(self.config.model_save_path, f'step{step}_{getTimestamp()}.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
