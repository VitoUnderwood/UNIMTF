# -*- coding:utf-8 -*-

import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


def collate_fn(batch):
    # [batchSize, object]
    # self.srcTokens[index], self.inputIds[index], self.attentionMask[index], self.tokenTypeIds[index]
    def pad(unPadding, element):
        width = max(len(d) for d in unPadding)
        rtn = [d + [element] * (width - len(d)) for d in unPadding]
        return rtn

    srcTokens = [x[0] for x in batch]
    inputIds = [x[1] for x in batch]
    attentionMask = [x[2] for x in batch]
    tokenTypeIds = [x[3] for x in batch]

    srcTokens = pad(srcTokens, "[PAD]")
    inputIds = torch.tensor(pad(inputIds, 0))
    attentionMask = torch.tensor(pad(attentionMask, 0))
    tokenTypeIds = torch.tensor(pad(tokenTypeIds, 0))

    return srcTokens, inputIds, attentionMask, tokenTypeIds


class DataEnhanceDataset(Dataset):
    def __init__(self, config, dataType):
        super(DataEnhanceDataset, self).__init__()

        if dataType == "train":
            fileName = config.trainFile
        elif dataType == "dev":
            fileName = config.devFile
        elif dataType == "test":
            fileName = config.testFile
        else:
            raise ValueError(f"数据集类型选择错误，{dataType}")

        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_path)
        self.srcTokens = []
        self.inputIds = []
        self.attentionMask = []
        self.tokenTypeIds = []
        with open(fileName, 'r') as fr:
            jsonList = json.load(fr)
            for jsonData in jsonList:
                srcText = jsonData["srcText"]
                tokens = tokenizer.tokenize(srcText)
                # tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(srcText))
                inp = tokenizer(srcText, return_tensors="pt")
                self.srcTokens.append(tokens)
                self.inputIds.append(inp.data["input_ids"].tolist()[0])
                self.attentionMask.append(inp.data["attention_mask"].tolist()[0])
                self.tokenTypeIds.append(inp.data["token_type_ids"].tolist()[0])

    def __getitem__(self, index):
        return self.srcTokens[index], self.inputIds[index], self.attentionMask[index], self.tokenTypeIds[index]

    def __len__(self):
        return len(self.srcTokens)
