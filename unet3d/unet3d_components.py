# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 21/08/2019 15:52
import torch
from torch.nn import init
import torch.nn.functional as F


def weight_init(in_channels, out_channels, k_size):
    weight = torch.empty(size=(out_channels, in_channels, k_size, k_size, k_size))
    weight = init.kaiming_normal_(tensor=weight, mode="fan_in", nonlinearity="leaky_relu")
    return weight


def bias_init(out_channels, val):
    b = torch.empty(out_channels)
    return init.constant_(tensor=b, val=val)


def conv3d_block(x, w, b, stride=1, padding=1, dropout_rate=0.3, training=True):
    x = F.conv3d(input=x, weight=w, bias=b, stride=stride, padding=padding)
    x = torch.nn.BatchNorm3d(x.shape[1])(x)
    x = F.leaky_relu(x)
    return F.dropout3d(input=x, p=dropout_rate, training=training)


def max_pool3d(x, k_size=2, stride=2, padding=0):
    return torch.nn.MaxPool3d(kernel_size=k_size, stride=stride, padding=padding)(x)


def conv3d_transpose(x, w, b, stride=2, padding=0, output_padding=1):
    return F.conv_transpose3d(input=x, weight=w, bias=b, stride=stride,
                              padding=padding, output_padding=output_padding)


def analysis_path(x, model_depth=4, root_feature_map_channels=16, w_k_size=3, pool_k_size=2, conv_stride=1,
                  pool_stride=2, conv_padding=1, pool_padding=0, dropout_rate=0.3, training=True):
    # each analysis layer contains two convolution blocks, this should not be changed
    num_conv_blocks = 2
    # create dicts to store weight, bias and conv ops (include pooling op) for each conv block
    weights = dict()
    bias = dict()
    conv_ops = dict()

    for depth in range(model_depth):
        conv = conv_ops["max_pool_{}".format(depth - 1)] if conv_ops else x
        for i in range(num_conv_blocks):
            feature_map_channels = 2 ** (depth + i) * root_feature_map_channels

            if depth == 0:
                in_channels = x.shape[1]
            else:
                in_channels = conv_ops["conv_{}_{}".format(depth - 1, 1)].size()[1]
            weights["weight_{}_{}".format(depth, i)] = weight_init(in_channels=in_channels,
                                                                   out_channels=feature_map_channels,
                                                                   k_size=w_k_size)
            bias["bias_{}_{}".format(depth, i)] = bias_init(out_channels=feature_map_channels, val=0.5)
            conv_ops["conv_{}_{}".format(depth, i)] = conv3d_block(x=conv,
                                                                   w=weights["weight_{}_{}".format(depth, i)],
                                                                   b=bias["bias_{}_{}".format(depth, i)],
                                                                   stride=conv_stride, padding=conv_padding,
                                                                   dropout_rate=dropout_rate, training=training)

            print(conv_ops["conv_{}_{}".format(depth, i)].shape)
        if depth < model_depth - 1:
            conv_ops["max_pool_{}".format(depth)] = max_pool3d(conv_ops["conv_{}_{}".format(depth, 1)],
                                                               k_size=pool_k_size, stride=pool_stride,
                                                               padding=pool_padding)
            print("The shape of x after max pooling: ", conv_ops["max_pool_{}".format(depth)].shape)
        else:
            break

    return conv_ops, weights, bias


def synthesis_path(conv_ops, model_depth, root_feature_map_channels=16):
    # create dicts to store weight, bias and conv ops (include pooling op) for each conv block
    weights = dict()
    bias = dict()
    conv_ops_up = dict()
    num_conv_blocks = 2

    for depth in range(model_depth - 2, -1, -1):
        print(depth)
        feature_map_channels = 2 ** (depth + 1) * root_feature_map_channels
        # convolution transpose step
        weights["deconv_weight_{}".format(depth)] = weight_init(in_channels=conv_ops["conv_{}_{}".format(depth + 1, 1)].shape[1],
                                                                out_channels=conv_ops["conv_{}_{}".format(depth + 1, 1)].shape[1],
                                                                k_size=3)
        bias["deconv_bias_{}".format(depth)] = bias_init(out_channels=conv_ops["conv_{}_{}".format(depth + 1, 1)].shape[1],
                                                         val=0.5)
        conv_ops_up["conv_up_{}".format(depth)] = conv3d_transpose(x=conv_ops["conv_{}_{}".format(depth + 1, 1)],
                                                                   w=weights["deconv_weight_{}".format(depth)],
                                                                   b=bias["deconv_bias_{}".format(depth)],
                                                                   stride=2, padding=1)
        print("The shape after conv transpose: ", conv_ops_up["conv_up_{}".format(depth)].shape)

        conv_ops_up["concat_{}".format(depth)] = torch.cat((conv_ops_up["conv_up_{}".format(depth)],
                                                            conv_ops["conv_{}_{}".format(depth, 1)]), dim=1)
        print("After concatenation: ", conv_ops_up["concat_{}".format(depth)].shape)
        # up convolution followed by two 3 x 3 x 3 convolution operations
        for i in range(num_conv_blocks):
            weights["weight_{}_{}".format(depth, i)] = weight_init(in_channels=conv_ops_up["concat_{}".format(depth)].shape[1],
                                                                   out_channels=feature_map_channels,
                                                                   k_size=3)
            bias["bias_{}_{}".format(depth, i)] = bias_init(out_channels=feature_map_channels, val=0.5)
            conv_ops_up["conv_{}_{}".format(depth, i)] = conv3d_block(x=conv_ops_up["concat_{}".format(depth)],
                                                                      w=weights["weight_{}_{}".format(depth, i)],
                                                                      b=bias["bias_{}_{}".format(depth, i)])
            print("hello: ", conv_ops_up["conv_{}_{}".format(depth, i)].shape)
    return conv_ops_up, weights, bias


def final_output(conv_ops_up):
    weights, bias = dict(), dict()
    weights["final"] = weight_init(out_channels=1, in_channels=32, k_size=3)
    bias["final"] = bias_init(1, val=0.1)
    output = conv3d_block(x=conv_ops_up["conv_{}_{}".format(0, 1)],
                          w=weights["final"],
                          b=bias["final"],
                          stride=1,
                          padding=1,
                          dropout_rate=0.5,
                          training=True)
    print("The shape of final output: ", output.shape)
    return output


def crop_feature_maps():
    pass


if __name__ == "__main__":
    w1 = weight_init(in_channels=1, out_channels=16, k_size=3)
    b1 = bias_init(out_channels=16, val=0.5)
    inputs = torch.randn(1, 1, 96, 96, 96)

    conv_results, analysis_weight_dict, analysis_bias_dict = analysis_path(inputs, model_depth=4)
    print(conv_results.keys())
    print("\n")
    up_conv_results, w_dict, b_dict = synthesis_path(conv_results, model_depth=4)
    print(up_conv_results.keys())
    final_output(up_conv_results)
