#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

caffe_root='../'
import sys
sys.path.insert(0,caffe_root+'python')

import caffe
from caffe.proto import caffe_pb2

def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_net_proto_file',
                        help='Input network prototxt file')

    args = parser.parse_args()
    return args

def Calculate_rf(param, layernum):
    rf = 1
    for layer in reversed(range(layernum)):
        kernel, stride, pad = param[layer]
        rf = ((rf -1)* stride) + kernel
    return rf


def main():
    args = parse_args()
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(args.input_net_proto_file).read(), net)
    layers=[];
    param=[];
    for layer in net.layer:
        if layer.type == 'Convolution':
            kernel = layer.convolution_param.kernel_size
            stride = layer.convolution_param.stride
            pad = layer.convolution_param.pad
            layers.append(layer.name)
            param.append([kernel,stride,pad])
        if layer.type == 'Pooling':
            kernel = layer.pooling_param.kernel_size
            stride = layer.pooling_param.stride
            pad = layer.pooling_param.pad
            layers.append(layer.name)
            param.append([kernel,stride,pad])
    for i in range(len(layers)):
            print layers[i]+": receptive field size is "+str(Calculate_rf(param,i+1));



if __name__ == '__main__':
    main()
