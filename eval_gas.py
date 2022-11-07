import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import sys, os

sys.path.append("./gas_model")

from gas_model import mlp, Model_for_shap
import argparse
import time
# sys.path.append("../DeepCTR-Torch/deepctr_torch/models")


def validate(model, val_loader, criterion, epoch, device=torch.device("cpu")):

    correct_counting = 0
    counting = 0
    loss_value = 0
    model.eval()
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device)
            y = y.to(device)
            y_ = model(x)
            y_hat = y_.argmax(axis=1)
            # loss_value = criterion(y_, y)
            correct_counting += (y_hat == y).sum() # accuracy_score(y_hat, y, normalize=False)
            counting += y.shape[0]

    acc_val = correct_counting*1./counting
    print("Epoch {}, ACC {}".format(epoch, acc_val))
    return acc_val, loss_value

if __name__ == "__main__":

    # checkpoint = torch.load("./gas_model/model_gas_division.pth.tar", map_location=torch.device('cpu'))
    checkpoint = torch.load("./gas_model/model_gas_vanilla.pth.tar", map_location=torch.device('cpu'))

    dense_feat_index  = checkpoint["dense_feat_index"]
    sparse_feat_index = checkpoint["sparse_feat_index"]
    cate_attrib_book  = checkpoint["cate_attrib_book"]

    model = mlp(input_dim=checkpoint["input_dim"],
                output_dim=checkpoint["output_dim"],
                layer_num=checkpoint["layer_num"],
                hidden_dim=checkpoint["hidden_dim"],
                activation=checkpoint["activation"])

    model.load_state_dict(checkpoint["state_dict"])
    # model = model.to(args.device)
    # model_for_shap = Model_for_shap(model, dense_feat_index, sparse_feat_index, cate_attrib_book)

    x_test = checkpoint["test_data_x"]
    y_test = checkpoint["test_data_y"]
    z_test = checkpoint["test_data_z"]

    test_loader = DataLoader(TensorDataset(x_test, y_test, z_test), batch_size=256, shuffle=False, drop_last=False, pin_memory=True)
    test_acc, _ = validate(model, test_loader, None, 0)
