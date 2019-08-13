#!/usr/bin/python3
import numpy as np
import sys, os, time
# from PIL import Image
# import cv2

import caffeTest.conv
import caffeTest.creat_caffe_model
import initData.random_simple_network
import initData.featuremap

import onnx
import onnxruntime
import onnxruntime.backend as backend
from onnx import version_converter, helper, shape_inference, optimizer
from onnx import TensorProto
import coremltools
import onnxmltools


# 编译器路径，用于onnx编译成loadable文件
COMPILER = 'complier/onnc.nv_full.130'
# caffe 转onnx 工具路径， 工具下载 https://github.com/yywbxgl/caffe-to-onnx.git
CONVERTER = 'caffe-to-onnx/convert2onnx.py'


def onnx_run(OnnxName, x):
    model_name = OnnxName
    model = onnx.load(model_name)
    print("load onnx model : ", model_name)
    print("input data shepe: ", x.dtype, x.shape)

    print(onnx.checker.check_model(model))
    
    # # model shape_inference
    # shape_model = shape_inference.infer_shapes(model)
    # onnx.checker.check_model(shape_model)
    # onnx.save(shape_model,  test_dir + "simple_net_shaped.onnx")

    # # model optimizer
    # passes = None
    # opti_mode = onnx.optimizer.optimize(shape_model, passes)
    # onnx.save(shape_model,  test_dir + "simple_net_opt.onnx")
    
    session = backend.prepare(model)

    output = session.run(x)
    output = np.array(output[0])
    return output


def csv2np(csv_file, npy):
    np.save(npy[:-4] if len(npy)>4 and npy[-4:]=='.npy' else npy, np.array(list(map(lambda x: int(x,16),open(csv_file, 'r').read().split(','))), dtype=np.uint8))
    # if (len(npy)>4 and npy[-4:]=='.npy'):
    #     file_name = npy[:-4]
    # else:
    #     file_name = npy
    
    # temp = open(csv_file, 'r').read().split(',')
    # temp = map(lambda x: int(x,16), temp)
    # temp = np.array(list(temp))
    # np.save(file_name, temp)

# def loadCaffeModel(net_path, model_path):
#     # read prototxt
#     net = caffe_pb2.NetParameter()
#     # 把字符串读如message中
#     text_format.Merge(open(net_path).read(), net)
#     # print(net.layer)

#     # read caffemodel
#     model = caffe_pb2.NetParameter()
#     f = open(model_path, 'rb')
#     # 反序列化
#     model.ParseFromString(f.read())
#     f.close()
#     # print(net.layer)
#     print("1.caffe模型加载完成")
#     print(model)
#     # return net,model

def simple_net_test(test_dir):
    # 0. 随机生成proto 以及 featuremap
    param = initData.random_simple_network.random_graph('Network')
    initData.random_simple_network.random_result(param, test_dir)
    initData.featuremap.main(test_dir+'data.npy', test_dir+'data.csv', np.float16)
    csv2np(test_dir+'data.csv', test_dir+'data_featuremap.npy')
    # random_simple_network.inference(test_dir)
    print("\n--------random data finish----------\n")

    # 1. 创建caffe Model，运行结果保存， 通过caffe-runtime推理，
    # x = get_numpy_from_img(conv_dir + 'conv.png')
    protofile = test_dir + 'deploy.prototxt'
    x = np.load(test_dir + "data.npy")

    # x = np.random.rand(3,64,64).astype(np.float32)
    # np.save(test_dir+"data", x)
    # conv_w = np.random.rand(3,4,4).astype(np.float32)
    # np.save(test_dir+"conv1-weight", conv_w)
    # conv_b = np.random.rand(1).astype(np.float32)
    # np.save(test_dir+"conv1-bias", conv_b)
    # fc_w = np.random.rand(10, 64).astype(np.float32)
    # np.save(test_dir+"ip1-weight", fc_w)
    # fc_b = np.random.rand(10).astype(np.float32)
    # np.save(test_dir+"ip1-bias", fc_b)

    [a,b,c] = x.shape
    x = x.reshape(1, a,b,c)
    print("input data shape:", x.dtype, x.shape)
    print(x)

    model , y1 = caffeTest.creat_caffe_model.run(test_dir)
    print(type(y1))
    
    # 保存Model, 和output
    caffe_model = test_dir + "simple_net.caffemodel"
    model.save(caffe_model)
    print("save model: ", caffe_model)
    np.save(caffe_model+".output", y1)
    print('output shape: ', y1.dtype, y1.shape)
    print(y1) 
    print("\n--------caffe run finish----------\n")


    # 2. 转换caffe model 到onnx, 在onnx-runtime上推理
    # 使用自定义的转换工具
    cmd = "python3 %s %s %s  simple_net  %s"%(CONVERTER, protofile, caffe_model, test_dir)
    print(cmd)
    os.system(cmd)

    # #使用官方推荐的转换工具
    # coreml_model  = coremltools.converters.caffe.convert((caffe_model, protofile))
    # onnx_model = onnxmltools.convert_coreml(coreml_model)
    # onnx_model_file = test_dir + "simple_net.onnx"
    # onnxmltools.utils.save_model(onnx_model, onnx_model_file)

    onnx_model_file = test_dir + "simple_net.onnx"
    y2 = onnx_run(onnx_model_file, x)

    print("output data shape: ", y2.dtype, y2.shape)
    print(y2)
    np.save(onnx_model_file +".output", y2)
    print("\n--------onnx run finish----------\n")


    # ----------------------------------------------------------    
    # 3. onnx 编译成loadale 文件, 在VP上或者FPGA上运行loadable，获取output
    # todo 添加input数据到onnx model中
    loadable = test_dir + "/simple_net.nvdla"
    os.system("%s -o %s %s"%(COMPILER, loadable, onnx_model_file))
    # os.system("cp conv.nvdla  /home/sunqiliang/onnc//home/sunqiliang/onnc/r1.2.0-ubuntu1604/r1.2.0/bin/onnc.nv_fullr1.2.0-ubuntu1604/full_V_riscv64/")
    # os.system("cp conv.nvdla  %s"%(test_dir))
    # todo VP 或者 FPGA测试自动化
    print("\n--------nvdla compile finish----------\n")


    # ----------------------------------------------------------
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
        print ("Usage:", sys.argv[0], "[output_dir]")
        sys.exit(-1)

    if (sys.argv[1][-1] != '/'):
        print(sys.argv[1])
        sys.argv[1] = sys.argv[1] + '/'
    
    simple_net_test(sys.argv[1])
