## convert_test
根据proto 以及 featuremap 生成caffe model， 转onnx model，编译成loadable文件  
对caffe 以及 Onnx  model 进行推理验证，比较结果

## 环境依赖
1. caffe
2. onnxruntime

## 运行脚本
1. 修改conv_test.py脚本中COMPILER 与 CONVERTER的路径

2. 运行命令

```
./conv_test.py  output_dir
```

3. 所有输出结果放在output_dir目录中，包括model以及input和推理后的output