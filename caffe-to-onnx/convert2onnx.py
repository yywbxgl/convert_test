#!/usr/bin/python3

import argparse
from src.caffe2onnx import *
import onnx
import onnxruntime.backend as backend
import numpy as np
import sys
#import cv2
from PIL import Image
from importlib import reload

# 保存onnx model
def saveonnxmodel(onnxmodel,onnx_save_path):
    try:
        onnx.checker.check_model(onnxmodel)
        onnx.save_model(onnxmodel, onnx_save_path+".onnx")
        print("模型保存成功,已保存至"+onnx_save_path+".onnx")
    except Exception as e:
        print("模型存在问题,未保存成功:\n",e)


def convert(NetPath, ModelPath, OnnxName, OnnxSavePath):
    # 创建转换器，load caffe model
    c2o = Caffe2Onnx(NetPath,ModelPath)
    # 生成onnx model
    onnxmodel = c2o.createOnnxModel()
    # 保存onnx model
    saveonnxmodel(onnxmodel,OnnxSavePath+OnnxName)


if __name__ == '__main__':

    # 参数解析  -h 查看帮助
    parser = argparse.ArgumentParser()
    parser.add_argument('CNP',help="caffe's caffemodel file path",nargs='?',default="./test/conv_test/conv.prototxt")
    parser.add_argument('CMP',help="caffe's prototxt file path",nargs='?',default="./test/conv_test/conv.caffemodel")
    parser.add_argument('ON',help="onnx model name",nargs='?',default="conv")
    parser.add_argument('OSP',help="onnx model file saved path",nargs='?',default="./")
    args = parser.parse_args()
    NetPath = args.CNP
    ModelPath = args.CMP
    OnnxName = args.ON
    OnnxSavePath = args.OSP

    convert(NetPath, ModelPath, OnnxName, OnnxSavePath)



