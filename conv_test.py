#!/usr/bin/python3

import numpy as np
# from PIL import Image
import sys, os
# import cv2
import caffeModel.conv
import onnx
import onnxruntime
import onnxruntime.backend as backend
from onnx import version_converter, helper
from onnx import TensorProto
import random_convolution

# 编译器路径，用于onnx编译成loadable文件
COMPILER = 'complier/onnc.nv_full'
# caffe 转onnx 工具路径， 工具下载 https://github.com/yywbxgl/caffe-to-onnx.git
CONVERTER = 'caffe-to-onnx/convert2onnx.py'


# # 图片数据读取为numpy
# def get_numpy_from_img(file):

#     img = Image.open("cat.jpg")
#     # x = np.array(img, dtype='float32')
#     # x = x.reshape(net.blobs['data'].data.shape)

#     # img = cv2.imread(file)
#     # cv2默认为 BGR顺序，而其他软件一般使用RGB，所以需要转换
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2默认为bgr顺序
#     x = np.array(img, dtype=np.float32)
#     # print(x)
#     x = np.reshape(x, (1,5,5,3))
#     # 矩阵转置换，img读取后的格式为W*H*C 转为model输入格式 C*W*H
#     x = np.transpose(x,(0,3,1,2))
#     return x

def onnx_run(OnnxName, x):
    
    model_name = OnnxName
    model = onnx.load(model_name)
    print("load onnx model : ", model_name)
    tf_rep = backend.prepare(model)
    print("input data shepe: ", x.dtype, x.shape)

    # for i in x[:,0,:]:
    #     print (i)

    # [rows, cols, channel] = x.shape
    # for i in range(rows):
    #     for j in range(cols):
    #         print(x[i,j])

    # for i in x:
    #     print(i)
    # print(x)

    # # Run the model on the backend
    # output = backend.run_model(model, x)

    # session = onnxruntime.InferenceSession(OnnxName)
    # input_name = session.get_inputs().name
    # label_name = session.get_outputs().name
    # print(input_name)
    # print(label_name)
    # pred_onx = session.run(None, {input_name:x})

    output = tf_rep.run(x)
    output = np.array(output[0])
    return output


def put_data_to_onnx(onnxmode, input):
    onnx_model = onnx.load(onnxmode)
    onnx.checker.check_model(onnx_model)
    print(onnx_model, ' is checked!')
    # print(onnx_model.graph.initializer[0].float_data[0:])


def conv_test(test_dir):
    # 0. 随机生成proto 以及 featuremap
    param = random_convolution.random_parameters()
    random_convolution.random_result(param, test_dir)
    print("\n--------random data finish----------\n")

    # 1. 创建caffe Model，运行结果保存， 通过caffe-runtime推理，
    # x = get_numpy_from_img(conv_dir + 'conv.png')
    protofile = test_dir + "deploy.prototxt"
    x = np.load(test_dir + "data.npy")
    w = np.load(test_dir + "conv-weight.npy")
    b = np.load(test_dir + "conv-bias.npy")
    [a,b,c] = x.shape
    x = x.reshape(1, a,b,c)
    print("input data shape:", x.dtype, x.shape)
    print(x)
    model, y1 = caffeModel.conv.run(protofile, x, w, b)
    # 保存Model, 和output
    caffe_model = test_dir + "conv.caffemodel"
    model.save(caffe_model)
    print("save model: ", caffe_model)
    np.save(caffe_model+".output", y1)
    print('output shape: ', y1.dtype, y1.shape)
    print(y1) 
    print("\n--------caffe run finish----------\n")

    # 2. 转换caffe model 到onnx, 在onnx-runtime上推理
    cmd = "python3 %s %s %s  conv  %s"%(CONVERTER, protofile, caffe_model, test_dir)
    print(cmd)
    os.system(cmd)
    # convert2onnx.convert(conv_dir+'conv.prototxt', conv_dir+'conv.caffemodel', 'test', conv_dir)
    onnx_model = test_dir + "conv.onnx"
    y2 = onnx_run(onnx_model, x)
    print("output data shape: ", y2.dtype, y2.shape)
    print(y2)
    np.save(onnx_model+".output", y2)
    print("\n--------onnx run finish----------\n")

    # 3. onnx 编译成loadale 文件, 在VP上或者FPGA上运行loadable，获取output
    # todo 添加input数据到onnx model中
    loadable = test_dir + "/conv.nvdla"
    os.system("%s -o %s %s"%(COMPILER, loadable, onnx_model))
    # os.system("cp conv.nvdla  /home/sunqiliang/onnc//home/sunqiliang/onnc/r1.2.0-ubuntu1604/r1.2.0/bin/onnc.nv_fullr1.2.0-ubuntu1604/full_V_riscv64/")
    # os.system("cp conv.nvdla  %s"%(test_dir))
    # todo VP 或者 FPGA测试自动化
    print("\n--------nvdla compile finish----------\n")

    # 4. 对比output
    compare = y1-y2
    compare = np.max(compare)
    print("result:", compare)
    if (compare < 0.1):
        print("convert success. test pass.")
    else:
        print("test not pass. ")
    

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ("Usage:", sys.argv[0], "output_dir")
        sys.exit(-1)

    if (sys.argv[1][-1] != '/'):
        print(sys.argv[1])
        sys.argv[1] = sys.argv[1] + '/'
    
    conv_test(sys.argv[1])