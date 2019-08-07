import caffe
import numpy as np
from PIL import Image
import sys, os
# import cv2

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'

# # 图片数据读取为numpy
# def get_numpy_from_img(file):
#     img = cv2.imread(file)
#     # cv2默认为 BGR顺序，而其他软件一般使用RGB，所以需要转换
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2默认为bgr顺序
#     x = np.array(img, dtype=np.float32)
#     # print(x)
#     x = np.reshape(x, (1,5,5,3))
#     # 矩阵转置换，img读取后的格式为W*H*C 转为model输入格式 C*W*H
#     x = np.transpose(x,(0,3,1,2))

#     # img = Image.open("cat.jpg")
#     # x = np.array(img, dtype='float32')
#     # x = x.reshape(net.blobs['data'].data.shape)

#     return x


# # 将numpy 转为图片保存
# def save_numpy_to_img(file):
#     tmp = []
#     channel = 3
#     for i in range(25):
#         for j in range(channel):
#             tmp.append(i)
#     x = np.array(tmp)
#     x = np.reshape(x, (5,5,3))
#     x = x.astype(np.float32)
#     print(x)

#     cv2.imwrite(file, x)
#     print("save img " ,file)

#     # im = Image.fromarray(x)
#     # im.save(file)


def create_input_data():
    # # 创建input data
    total = 5*5
    # x = np.zeros([1,3,5,5] , dtype=np.float32)
    x1 = np.arange(total, dtype=np.float32)
    x2 = np.arange(total, dtype=np.float32)
    x3 = np.arange(total, dtype=np.float32)
    x = np.concatenate((x1, x2, x3))
    x = np.reshape(x, (1,3,5,5))
    return x


def run(protofile, x, w, b):
    # 加载网络结构
    # caffe.set_mode_cpu()

    net = caffe.Net(protofile, caffe.TEST)

    print(net.blobs)

    # 获取input
    print('input shape: ',net.blobs['data'].data.dtype, net.blobs['data'].data.shape)
    # print(x)

    # 添加input数据到model
    net.blobs['data'].data[...] = x

    # 修改 weight 参数 (1,3,3,3)
    print('conv weight shape: ', net.params['conv'][0].data.dtype, net.params['conv'][0].data.shape)
    # w1 = np.arange(9, dtype=np.float32)
    # w2 = np.ones((9), dtype=np.float32)
    # w3 = np.ones((9), dtype=np.float32)*2
    # w = np.concatenate((w1, w2, w3))
    # w = np.reshape(w, net.params['conv'][0].data.shape)
    # w = np.ones(net.params['conv'][0].data.shape, dtype=np.float32)
    net.params['conv'][0].data[:] = w
    # print(w)

    # 修改 bias 参数
    print('conv bias shape: ', net.params['conv'][1].data.dtype, net.params['conv'][1].data.shape)
    # b = np.zeros(net.params['conv'][1].data.shape, dtype=np.float32)
    net.params['conv'][1].data[:] = b
    # print(b)

    # 运行结果
    net.forward()
    # print('output shape: ', net.blobs['conv'].data.dtype, net.blobs['conv'].data.shape)
    # print(net.blobs['conv'].data)   #运行测试

    # 保存Model
    # file_name = save_path + "conv.caffemodel"
    # net.save(file_name)
    # print("save model: ", file_name)

    return net, net.blobs['conv'].data


if __name__ == '__main__':
    save_numpy_to_img('conv.png')
    x = get_numpy_from_img('conv.png')
    run(x)