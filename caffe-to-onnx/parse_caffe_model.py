from proto import caffe_pb2
from google.protobuf import text_format
import onnx


def loadCaffeModel(net_path, model_path):
    # read prototxt
    net = caffe_pb2.NetParameter()
    # 把字符串读如message中
    text_format.Merge(open(net_path).read(), net)
    # print(net.layer)

    # read caffemodel
    model = caffe_pb2.NetParameter()
    f = open(model_path, 'rb')
    # 反序列化
    model.ParseFromString(f.read())
    f.close()
    # print(net.layer)
    print("1.caffe模型加载完成")
    print(model)
    # return net,model


if __name__ == "__main__":
    loadCaffeModel("caffemodel/test/test.prototxt", "caffemodel/test/test.caffemodel")
