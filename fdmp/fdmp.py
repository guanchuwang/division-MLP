import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.cuda.amp import autocast
import cpp_extension.quantization as ext_quantization
# import exact.cpp_extension.quantization as ext_quantization
import cpp_extension.minimax as ext_minimax
# import cpp_extension.backward_func as ext_backward_func
# from torch.cuda.amp import autocast as autocast
import math

# import numpy as np
from conf import config
import time

from utils import *

total_act_mem = 0
total_act_mem_lfc = 0 # torch.tensor(0).type(torch.long)
total_act_mem_hfc = 0 # torch.tensor(0).type(torch.long)

# @torch.no_grad()
# def abs_window_size(N, window_size):
#     if config.round_window:
#         return round(window_size*N + 0.5)
#     else:
#         return round(window_size*N)

class FDMP_(Function):

    @staticmethod
    @torch.no_grad()
    def quantize_and_pack1d(data, bits, mn, mx, N):

        mn_ = mn.view(N, 1, 1).repeat(1, data.shape[1], 1)
        mx_ = mx.view(N, 1, 1).repeat(1, data.shape[1], 1)

        # print(data.shape)
        # print(mn_.shape)
        # print(mx_.shape)
        # # print(scale.shape)
        # print(bits, type(bits))

        # output = pack_func(data, mn, mx, scale.to(data.dtype), bits, True)
        output, scale = ext_quantization.pack_single_precision(data, mn_, mx_, bits, True)
        scale = scale[:, 0, 0].clone()
        # import pdb
        # pdb.set_trace()

        return output, scale

    @staticmethod
    @torch.no_grad()
    def dequantize_and_unpack1d(data, shape, bits, scale, mn):

        # Pad to group_size
        Batch, feature_dim = shape

        # print(data.shape)
        # print(shape)
        # print(scale.shape)
        # print(mn.shape)

        if feature_dim > config.max_thread:
            thread_loop = math.ceil(feature_dim / config.max_thread)
            thread = feature_dim // thread_loop
            mn_ = mn.view(Batch, 1, 1).repeat(1, thread_loop, 1)
            scale_ = scale.view(Batch, 1, 1).repeat(1, thread_loop, 1)
            data = ext_quantization.unpack_single_precision(data, bits, scale_, mn_, Batch, thread_loop, thread)

        else:
            mn_ = mn.view(Batch, 1, 1)
            scale_ = scale.view(Batch, 1, 1)
            data = ext_quantization.unpack_single_precision(data, bits, scale_, mn_, Batch, 1, feature_dim)

        return data

    @staticmethod
    @torch.no_grad()
    def fdmp1d(x):

        Batch, feature_dim = x.shape

        # print(Batch, feature_dim)

        if feature_dim == 1:
            return x, None, None, None

        x_reshape = x.reshape(Batch, 1, feature_dim)
        pool_kernel_size = config.lfc_block if feature_dim >= config.lfc_block else feature_dim
        x_lfc = F.avg_pool1d(x_reshape, pool_kernel_size, stride=pool_kernel_size, padding=0)
        x_lfc_float16 = x_lfc.to(torch.bfloat16)
        x_lfc_large = F.interpolate(x_lfc_float16.to(x_lfc.dtype), size=[feature_dim],
                                    scale_factor=None)  # Batch, feature_dim, Channel/(block^2)

        x_hfc = x_reshape - x_lfc_large  # feature_dim must <= 1024
        if feature_dim > config.max_thread:
            thread_loop = math.ceil(feature_dim / config.max_thread)  # .type(torch.int)
            thread = feature_dim // thread_loop
            x_hfc_groups = x_hfc.reshape(Batch, thread_loop, thread)
        else:
            x_hfc_groups = x_hfc.reshape(Batch, 1, feature_dim)

        q_min = x_hfc_groups.min(dim=-1).values.min(dim=-1).values
        mx = x_hfc_groups.max(dim=-1).values.max(dim=-1).values
        q_bits = config.hfc_bit_num
        q_input, q_scale = FDMP.quantize_and_pack1d(x_hfc_groups, q_bits, q_min, mx, Batch)

        return x_lfc_float16, q_input, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16)

    @staticmethod
    @torch.no_grad()
    def de_fdmp1d(feature_pack, q_input_shape):

        Batch, feature_dim = q_input_shape

        # print(Batch, feature_dim)

        if feature_dim == 1:
            x, _, _, _ = feature_pack
            return x

        x_lfc_float16, q_input, q_scale, q_min = feature_pack

        # Estimate valid group size
        if not config.half_precision:
            x_lfc = x_lfc_float16.to(torch.float32)  # Remove it if accuracy drops
            q_scale = q_scale.to(torch.float32)
            q_min = q_min.to(torch.float32)
        else:
            x_lfc = x_lfc_float16.to(torch.float16)  # Remove it if accuracy drops
            q_scale = q_scale.to(torch.float16)
            q_min = q_min.to(torch.float16)

        q_bits = config.hfc_bit_num

        x_hfc_dequant = FDMP.dequantize_and_unpack1d(q_input, q_input_shape, q_bits, q_scale, q_min)
        x_hfc_dequant = x_hfc_dequant.view(*q_input_shape).contiguous()
        x_lfc_large = F.interpolate(x_lfc, size=[feature_dim], scale_factor=None).reshape(*q_input_shape)
        # print(x_lfc_large.shape)
        # print(x_hfc_dequant.shape)

        return x_lfc_large + x_hfc_dequant



class WDCT(Function):

    @staticmethod
    @torch.no_grad()
    def generate_dct_matrix(N, n, device):

        i_vector = torch.arange(n, device=device)
        j_vector = torch.arange(N, device=device)

        i_matrix, j_matrix = torch.meshgrid(i_vector, j_vector)

        dct_matrix = torch.sqrt((1 + (i_matrix != 0) * 1) / N) * \
                     torch.cos((2 * j_matrix + 1) * 3.141592653589793 / (2 * N) * i_matrix)

        return dct_matrix
        # return torch.nn.Parameter(dct_matrix, requires_grad=False)

    @staticmethod
    @torch.no_grad()
    def dct_1d_(dct_matrix, x):  # needs to debug

        n2, n1 = dct_matrix.shape

        x_shape = x.shape
        x = x.contiguous().view(-1, n1)
        # x = layer(x)
        x = x.mm(dct_matrix.T)

        # print(x.shape)
        x = x.view((n2, x_shape[0])).permute(1, 0)

        return x

    @staticmethod
    @torch.no_grad()
    def idct_1d_(dct_matrix, x_dct):
        return WDCT.dct_1d_(dct_matrix.T, x_dct)

    @staticmethod
    @torch.no_grad()
    def dct_1d(dct_matrix, x, inverse=False):

        if not inverse:
            return WDCT.dct_1d_(dct_matrix, x)

        else:
            return WDCT.idct_1d_(dct_matrix, x)

    @staticmethod
    @torch.no_grad()
    def dct_2d_(dct_matrix, x):
        # layer = self.idct_layer if inverse else self.dct_layer
        # n1, n2 = (self.n, self.N) if inverse else (self.N, self.n)
        # print("====================")
        # print(x.device, dct_matrix.device)

        n2, n1 = dct_matrix.shape

        x_shape = x.shape
        x = x.contiguous().view(-1, n1)
        # x = layer(x)
        x = x.mm(dct_matrix.T)
        x = x.T.contiguous().view(-1, n1)
        # x = layer(x).T
        x = x.mm(dct_matrix.T).T

        x = x.view((n2, n2, x_shape[0], x_shape[1])).permute(2, 3, 0, 1)

        return x

    @staticmethod
    @torch.no_grad()
    def idct_2d_(dct_matrix, x_dct):
        return WDCT.dct_2d_(dct_matrix.T, x_dct)

    @staticmethod
    @torch.no_grad()
    def dct_2d(dct_matrix, x, inverse=False):

        if not inverse:
            return WDCT.dct_2d_(dct_matrix, x)

        else:
            return WDCT.idct_2d_(dct_matrix, x)

    @staticmethod
    @torch.no_grad()
    def dct_3d_(dct_matrix, x):

        n2, n1 = dct_matrix.shape

        x_shape = x.shape
        x = x.contiguous().view(-1, n1)
        x = x.mm(dct_matrix.T)

        x = x.T.contiguous().view(-1, n1)
        x = x.mm(dct_matrix.T).T

        x = x.T.contiguous().view(-1, n1)
        x = x.mm(dct_matrix.T).T

        x = x.view((n2, n2, n2, x_shape[0], x_shape[1])).permute(2, 3, 4, 0, 1)

        return x

    @staticmethod
    @torch.no_grad()
    def idct_3d_(dct_matrix, x_dct):
        return WDCT.dct_3d_(dct_matrix.T, x_dct)

    @staticmethod
    @torch.no_grad()
    def dct_3d(dct_matrix, x, inverse=False):

        if not inverse:
            return WDCT.dct_3d_(dct_matrix, x)

        else:
            return WDCT.idct_3d_(dct_matrix, x)


# class WDCT1d(WDCT):
#
#     @staticmethod
#     @torch.no_grad()
#     def dct(dct_matrix, x):  # needs to debug
#
#         n2, n1 = dct_matrix.shape
#
#         x_shape = x.shape
#         x = x.contiguous().view(-1, n1)
#         # x = layer(x)
#         x = x.mm(dct_matrix.T)
#         x = x.view((n2, x_shape[0], x_shape[1])).permute(2, 0, 1)
#
#         return x
#
#     @staticmethod
#     @torch.no_grad()
#     def idct(dct_matrix, x_dct):
#         return WDCT1d.dct(dct_matrix.T, x_dct)
#
#
# class WDCT2d(WDCT):
#
#     @staticmethod
#     @torch.no_grad()
#     def dct(dct_matrix, x):
#         # layer = self.idct_layer if inverse else self.dct_layer
#         # n1, n2 = (self.n, self.N) if inverse else (self.N, self.n)
#         # print("====================")
#         # print(x.device, dct_matrix.device)
#
#         n2, n1 = dct_matrix.shape
#
#         x_shape = x.shape
#         x = x.contiguous().view(-1, n1)
#         # x = layer(x)
#         x = x.mm(dct_matrix.T)
#         x = x.T.contiguous().view(-1, n1)
#         # x = layer(x).T
#         x = x.mm(dct_matrix.T).T
#
#         x = x.view((n2, n2, x_shape[0], x_shape[1])).permute(2, 3, 0, 1)
#
#         return x
#
#     @staticmethod
#     @torch.no_grad()
#     def idct(dct_matrix, x_dct):
#         return WDCT2d.dct(dct_matrix.T, x_dct)


class FDMP(Function):

    @staticmethod
    @torch.no_grad()
    def quantize_and_pack1d(data, bits, mn, mx, N):
        if not config.half_precision:
            return FDMP_.quantize_and_pack1d(data, bits, mn, mx, N)
        else:
            with autocast():
                return FDMP_.quantize_and_pack1d(data, bits, mn, mx, N)

    @staticmethod
    @torch.no_grad()
    def dequantize_and_unpack1d(data, shape, bits, scale, mn):
        if not config.half_precision:
            return FDMP_.dequantize_and_unpack1d(data, shape, bits, scale, mn)
        else:
            with autocast():
                return FDMP_.dequantize_and_unpack1d(data, shape, bits, scale, mn)

    @staticmethod
    @torch.no_grad()
    def fdmp1d(x):
        if not config.half_precision:
            return FDMP_.fdmp1d(x)
        else:
            with autocast():
                return FDMP_.fdmp1d(x)

    @staticmethod
    @torch.no_grad()
    def de_fdmp1d(feature_pack, q_input_shape):
        if not config.half_precision:
            return FDMP_.de_fdmp1d(feature_pack, q_input_shape)
        else:
            with autocast():
                return FDMP_.de_fdmp1d(feature_pack, q_input_shape)







# class DCT_matrix:
#
#     def __init__(self):
#
#         self.dct_matrix_dict = nn.ParameterDict()
#
#     @torch.no_grad()
#     def new_dct_matrix(self, N, n):
#         print(len(self.dct_matrix_dict))
#
#         if N in self.dct_matrix_dict.keys():
#
#             return
#
#         self.dct_matrix_dict[str(N)] = WDCT.generate_dct_matrix(N, n)
#
#         return
#
#     def __getitem__(self, key):
#
#         return self.dct_matrix_dict[str(key)]
#
#
# dct_matrix_buf = DCT_matrix()



    # def dct(self, x):
    #
    #     # dct_matrix = self.generate_dct_matrix()
    #     # print("====================")
    #     # print(x.device, self.dct_layer.weight.device)
    #
    #     x_shape = x.shape
    #     x = x.view(-1, self.N)
    #     x = self.dct_layer(x)
    #     x = x.T.contiguous().view(-1, self.N)
    #     x = self.dct_layer(x).T
    #     x = x.view((self.n, self.n, x_shape[0], x_shape[1])).permute(2, 3, 0, 1)
    #
    #     return self.padder(x)
    #
    # def idct(self, x_dct):
    #
    #     return x_dct



# def generate_dct_matrix(N, n):
#
#     i_vector = torch.arange(n).cuda()
#     j_vector = torch.arange(N).cuda()
#
#     i_matrix, j_matrix = torch.meshgrid(i_vector, j_vector)
#
#     dct_matrix = torch.sqrt((1 + (i_matrix != 0) * 1) / N) \
#                  * torch.cos((2 * j_matrix + 1) * 3.14159265 / (2 * N) * i_matrix)
#
#     return dct_matrix
#
# def dct_2d(x, dct_matrix):
#
#     return torch.matmul(torch.matmul(dct_matrix, x), dct_matrix.T)
#
#
# def zero_padding(x, left, right, up, down):
#
#     padder = nn.ZeroPad2d((left, right, up, down))
#     return padder(x)

    # return F.pad(x, (left, right, up, down), "constant", 0)

