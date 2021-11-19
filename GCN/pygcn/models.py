import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pygcn.layers import GraphConvolution
from pygcn.utils import accuracy


MAX_NB_EPOCHS = 1000
NB_CONSECUTIVE_EPOCHS = 10


class GCN(nn.Module):
    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        nb_additional_h: int = 0
    ):

        super(GCN, self).__init__()

        # params
        self.dropout = dropout
        self.nb_additional_h = nb_additional_h

        # first layers
        self.gc1 = GraphConvolution(nfeat, nhid)

        # additional hidden layers
        self.additional_gcs = []
        for i in range(nb_additional_h):
            self.additional_gcs.append(
                GraphConvolution(nhid, nhid)
            )

        # last layer
        self.gc2 = GraphConvolution(nhid, nclass)

        # init optimizer
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=0.01,
        )
        return

    def forward(self, x, adj):
        # first layer
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)

        # additional layers
        for i in range(self.nb_additional_h):
            current_layer = self.additional_gcs[i]

            x = F.relu(current_layer(x, adj))
            # x = F.dropout(x, self.dropout, training=self.training)

        # last layer
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def fit(
        self,
        features,
        adj,
        labels,
        idx_train,
        idx_val
    ):

        # parmas
        tracker = {}
        arr_10_vals = []

        # train loop
        for epoch in range(MAX_NB_EPOCHS):
            # set train phase
            self.train()
            self.optimizer.zero_grad()

            # forward
            output = self(features, adj)
            # train
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            # validation
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

            # condition the training
            arr_10_vals.append(loss_val.item())

            if len(arr_10_vals) == NB_CONSECUTIVE_EPOCHS:
                # stop training
                if arr_10_vals[0] < np.array(arr_10_vals).mean():
                    break

                # remove first item
                arr_10_vals.pop(0)

            # backward
            loss_train.backward()
            self.optimizer.step()

            # save meta data
            tracker[epoch] = {
                'loss_train': loss_train.item(),
                'acc_train': acc_train.item(),
                'loss_val': loss_val.item(),
                'acc_val': acc_val.item()
            }

        return tracker

    def test_model(
        self,
        features,
        adj,
        labels,
        idx_test
    ):
        # set test  phase
        self.eval()

        # forward
        output = self(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        # print meta data
        print(
            "Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item())
        )

        return loss_test.item(), acc_test.item()
