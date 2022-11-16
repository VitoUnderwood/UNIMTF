# -*- coding:utf-8 -*_
from torch import nn


class PrefixNlgModel(nn.Module):
    def __init__(self, config):
        super(PrefixNlgModel, self).__init__()
        self.config = config

    def forward(self):
        pass
