## convert_test
根据proto 以及 featuremap 生成caffe model， 转onnx model，编译成loadable文件  
对caffe 以及 Onnx  model 进行推理验证，比较结果

## 环境依赖
1. caffe
2. onnx
3. onnxruntime

## 运行脚本

```
python3  conv_test.py  [output_dir]
```

## docker运行命令
```
 docker run  -it -e PYTHONIOENCODING=utf-8  -v /data/home/qiliang.sun/onnx_test/:/home  onnx_test  bash
```