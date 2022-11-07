import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import copy

import sys, os
sys.path.append("./gas_model")

from gas_model import mlp
import sys
sys.path.append("./fdmp")
from conf import config, config_init

import argparse
from module import FDMP_Module

parser = argparse.ArgumentParser(description="DIVISION MLP", add_help=True)
parser.add_argument("--vanilla", action="store_true")
parser.add_argument("--lfc_block", type=int, default=64, help="")
parser.add_argument("--hfc_bit_num", type=int, default=2, help="")
parser.add_argument('--simulate', action='store_true', help='Simulate the quantization.')
parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
parser.add_argument("--gpu", type=int, help="gpu id", default=None) # for good shapley value ranking
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
parser.add_argument("--save", action="store_true")

config_global = parser.parse_args()

config_init(config_global)


def load_raw_data(path):

    input_data = pd.read_csv(path, na_values="?") # , names=column_names)
    # input_data = (pd.read_csv(path, names=column_names,
    #                           na_values="?", sep=r'\s*,\s*', engine='python'
    #                           ).loc[lambda df: df['race'].isin(['White', 'Black'])])

    # print(input_data)
    # sensitive_attribs = "sex"
    # sensitive_value = input_data.loc[:, sensitive_attribs].values
    # Z = pd.DataFrame((sensitive_value == "Male").astype(int), columns=["sex"])
    y = input_data["target"].values.astype(np.long)
    # categorical_attribs = []
    # categorical_value = input_data.loc[:, categorical_attribs].pipe(pd.get_dummies, drop_first=False)
    real_attribs = [str(x) for x in range(128)]
    real_value = StandardScaler().fit_transform(input_data.loc[:, real_attribs].values)
    real_attrib_index_buf = np.arange(real_value.shape[-1])

    # cate_attrib_encoded_buf = []
    # cate_attrib_raw_buf = []
    # cate_attrib_index_buf = []
    # cate_attrib_book_buf = []
    #
    # for cate_attrib_index, cate_attrib in enumerate(categorical_attribs):
    #
    #     enc = OneHotEncoder() # handle_unknown='ignore')
    #     enc.fit(input_data.loc[:, cate_attrib].values.reshape(-1,1))
    #     cate_attrib_encoded = enc.transform(input_data.loc[:, cate_attrib].values.reshape(-1,1)).toarray()
    #     cate_attrib_book = np.eye(cate_attrib_encoded.shape[-1]) # np.unique(cate_attrib_encoded, axis=0)
    #
    #     scaler = StandardScaler()
    #     scaler.fit(cate_attrib_encoded)
    #     cate_attrib_scaled = scaler.transform(cate_attrib_encoded)
    #     cate_attrib_book_scaled = scaler.transform(cate_attrib_book)
    #     reference = cate_attrib_encoded.mean(axis=0).reshape(1,-1)
    #     cate_attrib_book_scaled = np.concatenate((cate_attrib_book_scaled, reference), axis=0) # -1 for background noise
    #     # cate_attrib_book_scaled = np.concatenate((reference, cate_attrib_book_scaled), axis=0) # 0 for background noise
    #
    #     cate_attrib_raw = cate_attrib_encoded.argmax(axis=1).reshape(-1,1) # -1 for background noise
    #     # cate_attrib_raw = cate_attrib_encoded.argmax(axis=1).reshape(-1,1) + 1 # 0 for background noise
    #     cate_attrib_encoded_buf.append(cate_attrib_scaled)
    #     cate_attrib_raw_buf.append(cate_attrib_raw)
    #     cate_attrib_index_buf.append(real_value.shape[-1] + cate_attrib_index)
    #     cate_attrib_book_buf.append(cate_attrib_book_scaled)
    #
    #     # print(input_data.loc[:, cate_attrib].values.reshape(-1, 1))
    #     # print(cate_attrib_encoded)
    #     # print(cate_attrib_raw)
    #     # print(cate_attrib_book)
    #     # print(cate_attrib_book.shape)
    #     # print(reference)
    #     # print(cate_attrib_book_scaled)
    #
    # cate_attrib_index_buf = np.array(cate_attrib_index_buf, dtype=np.long)
    #
    # # print(real_value.values)
    # # print(cate_attrib_encoded_buf)
    #
    # X_raw = np.concatenate([real_value] + cate_attrib_raw_buf, axis=1)
    # # X = pd.concat([real_value, categorical_value], axis=1)
    # X = np.concatenate([real_value] + cate_attrib_encoded_buf, axis=1)

    # print(X_raw[:, cate_attrib_index_buf].min())
    # print(X[:, 0:4])
    # print(X_raw[:, 0:4])
    # stop

    X = real_value.copy()
    X_raw = real_value.copy()

    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape} samples")
    # print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")
    return X, y, None, X_raw, None, real_attrib_index_buf, None


def load_data(path, val_size, test_size, run_num=0):

    X, Y, Z, X_raw, cate_attrib_book, dense_feat_index, sparse_feat_index = load_raw_data(path)

    print(X.shape)
    print(Y.shape)

    x_train_all, x_test, y_train_all, y_test, z_train_all, z_test = train_test_split(X, Y, X_raw, test_size=test_size, random_state=run_num)
    x_train, x_val, y_train, y_val, z_train, z_val = train_test_split(x_train_all, y_train_all, z_train_all, test_size=val_size, random_state=0) # , random_state=fold)

    # print(x_train[:, 0:5])
    # print(z_train[:, 0:5])

    datasets_np = [(x_train, y_train, z_train), (x_val, y_val, z_val), (x_test, y_test, z_test)]
    datasets_torch = ()
    for dataset in datasets_np:
        x, y, z = dataset
        x = torch.from_numpy(x).type(torch.float)
        y = torch.from_numpy(y).type(torch.long) # .squeeze(dim=1)
        z = torch.from_numpy(z).type(torch.float)
        datasets_torch += (x, y, z)

    # print(datasets_torch)
    # print(datasets_torch[0][:, 0:5])
    # print(datasets_torch[2][:, 0:5])

    # cate_attrib_book = [torch.from_numpy(x).type(torch.float) for x in cate_attrib_book]
    # dense_feat_index = torch.from_numpy(dense_feat_index).type(torch.long)
    # sparse_feat_index = torch.from_numpy(sparse_feat_index).type(torch.long)

    return datasets_torch, None, dense_feat_index, None


def train_epoch(model, train_loader, criterion, optimizer, device=torch.device("cpu")):

    model.train()
    for x, y, _ in train_loader:
        x = x.to(device)
        y = y.to(device)
        y_ = model(x)
        # print(y_.shape, y.shape)
        loss_value = criterion(y_, y)
        optimizer.zero_grad()
        loss_value.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=2)  # necessary!!!
        optimizer.step()

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


def save_checkpoint(fname, **kwargs):

    checkpoint = {}

    for key, value in kwargs.items():
        checkpoint[key] = value
        # setattr(self, key, value)

    torch.save(checkpoint, fname)


def load_checkpoint(fname):

    return torch.load(fname)


def train(model, train_loader, val_loader, criterion, optimizer, max_epoch=50, device=torch.device("cpu")): # , round_num=0, checkpoint=None, save_path="./"):

    best_acc = 0
    best_state_dict = None
    for epoch in range(max_epoch):

        train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _ = validate(model, val_loader, criterion, epoch, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())

    return best_state_dict









if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    args.device = torch.device("cuda:" + str(args.gpu)) if args.gpu is not None else torch.device("cpu")

    round_num = 0
    datasets_torch, cate_attrib_book, dense_feat_index, sparse_feat_index = load_data('./gas_dataset/gas.csv', val_size=0.2, test_size=0.2, run_num=round_num) #
    x_train, y_train, z_train, x_val, y_val, z_val, x_test, y_test, z_test = datasets_torch

    print("Train:")
    print(x_train.shape)
    print("Val:")
    print(x_val.shape)
    print("Test:")
    print(x_test.shape)

    original_model = mlp(input_dim=x_train.shape[1], output_dim=6,
                   layer_num=3, hidden_dim=64,
                   activation="torch.relu")

    original_model = original_model.to(args.device)
    ######## Convert model to Division ########
    if args.vanilla:
        model = original_model
        print(original_model)
    else:
        model = FDMP_Module(original_model, args.device)
    ###########################################


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(x_train, y_train, z_train), batch_size=256, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val, z_val), batch_size=256, shuffle=False, drop_last=False, pin_memory=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test, z_test), batch_size=256, shuffle=False, drop_last=False, pin_memory=True)

    # checkpoint = {}

    best_state_dict = train(model, train_loader, val_loader, criterion, optimizer, max_epoch=20, device=args.device) # , round_num=0, save_path="./adult-dataset/")

    model.load_state_dict(best_state_dict)
    test_acc, _ = validate(model, test_loader, None, 0, device=args.device)
    # print(test_acc)

    if args.save:

        # original_model.load_state_dict(best_state_dict)
        # original_model = original_model.to(torch.device("cpu"))

        checkpoint_name = "./gas_model/model_gas_division.pth.tar" if not args.vanilla else "./gas_model/model_gas_vanilla.pth.tar"
        save_checkpoint(checkpoint_name,
                        round_index=round_num,
                        state_dict=best_state_dict, # original_model.state_dict(),
                        layer_num=original_model.layer_num,
                        input_dim=original_model.input_dim,
                        hidden_dim=original_model.hidden_dim,
                        output_dim=original_model.output_dim,
                        activation=original_model.activation_str,
                        test_data_x = x_test,
                        test_data_y = y_test,
                        test_data_z = z_test,
                        test_acc = test_acc,
                        cate_attrib_book=cate_attrib_book,
                        dense_feat_index=dense_feat_index,
                        sparse_feat_index=sparse_feat_index,
                        )