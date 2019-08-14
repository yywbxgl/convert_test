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
import caffe
import cv2
from PIL import Image


# 编译器路径，用于onnx编译成loadable文件
COMPILER = 'complier/onnc.nv_full.130'
# caffe 转onnx 工具路径， 工具下载 https://github.com/yywbxgl/caffe-to-onnx.git
CONVERTER = 'caffe-to-onnx/convert2onnx.py'

# # 图片数据读取为numpy
def get_numpy_from_img(file):

    # img = Image.open(file)
    # x = np.array(img, dtype='float32')

    img = cv2.imread(file)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #cv2默认为 BGR顺序，而其他软件一般使用RGB，所以需要转换
    img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
    x = np.array(img, dtype=np.float32)

    # 矩阵转置换，img读取后的格式为W*H*C 转为model输入格式 C*W*H
    x = np.transpose(x,(2,0,1))
    (c,w,h) = x.shape
    x = x.reshape(1,c,w,h)
    return x


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

def classic_net_test(test_dir, protofile, caffemodel, img):

    net = caffe.Net(protofile, caffemodel, caffe.TEST)
    # (n,c,h,w) = net.blobs['data'].data.shape
    # x = np.random.rand(n,c,h,w).astype(np.float32) * 100
    x = get_numpy_from_img (img)
    print("input shape: ", x.shape, x.dtype)

    # inference
    out = net.forward()

    # print output
    for output in out:
        y1 = net.blobs[output].data

    print("caffe output: \n", y1)
    
    # 保存Model, 和output
    np.save(caffemodel+".output", y1)
    print('output shape: ', y1.dtype, y1.shape)
    print(y1) 
    print("\n--------caffe run finish----------\n")


    # 2. 转换caffe model 到onnx, 在onnx-runtime上推理
    # 使用自定义的转换工具
    cmd = "python3 %s %s %s  simple_net  %s"%(CONVERTER, protofile, caffemodel, test_dir)
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

    print("caffe ouput max:", np.max(y1))
    print("onnx output max:", np.max(y2))
    

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print ("Usage:", sys.argv[0], "[output_dir]  [caffe.prototxt]  [test.caffemodel]  [input_img]")
        sys.exit(-1)

    if (sys.argv[1][-1] != '/'):
        print(sys.argv[1])
        sys.argv[1] = sys.argv[1] + '/'
    
    classic_net_test(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
