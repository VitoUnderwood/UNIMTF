# -*- coding:utf-8 -*-

import json

import torch
from torch.utils.data import Dataset, DataLoader


def collate_fn(batch_data):
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # 接受dataset getitem 返回的数据，注意顺序是一致的
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 返回的顺序可以进行变化，决定了dataloader返回值
    def pad(unpadded_data, pad_id):
        width = max(len(d) for d in unpadded_data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in unpadded_data]
        return rtn_data

    pre_src = [x[0] for x in batch_data]
    pre_labels = [x[1] for x in batch_data]
    pre_segs = [x[2] for x in batch_data]
    pre_clss = [x[3] for x in batch_data]

    src = torch.tensor(pad(pre_src, 0))
    labels = torch.tensor(pad(pre_labels, 0))
    segs = torch.tensor(pad(pre_segs, 0))
    mask = ~(src == 0)

    clss = torch.tensor(pad(pre_clss, -1))
    mask_cls = ~ (clss == -1)
    clss[clss == -1] = 0
    src_str = [x[-1] for x in batch_data]
    return src, labels, segs, clss, mask, mask_cls, src_str


class PrefixNlgDataset(Dataset):
    def __init__(self, file_name):
        super(PrefixNlgDataset, self).__init__()
        self.mask_clss = []
        self.masks = []
        self.clss = []
        self.segs = []
        self.labels = []
        self.srcs = []
        self.src_str = []
        json_list = json.load(open(file_name, 'r'))
        for i in range(len(json_list)):
            self.srcs.append(json_list[i]['src'])
            self.labels.append(json_list[i]['labels'])
            self.segs.append(json_list[i]['segs'])
            self.clss.append(json_list[i]['clss'])

            self.masks.append([0])
            self.mask_clss.append([0])
            self.src_str.append(json_list[i]['src_txt'])

    def __getitem__(self, index):
        return self.srcs[index], self.labels[index], self.segs[index], self.clss[index], self.masks[index], \
               self.mask_clss[index], self.src_str[index]

    def __len__(self):
        return len(self.srcs)


