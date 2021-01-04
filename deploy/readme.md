### torch2onnx2tensorRT

#### 环境 1
 - torch == 1.1.0 (torch.onnx.export 暂不支持dynamic_axes)
 - onnx == 1.4.0
 - onnxruntime == 1.4.0
 - tensorRT == 6.0.1.5
 - cuda == 9.0.176
 
#### 环境 2
 - torch == 1.4.0
 - onnx == 1.6.0
 - onnxruntime == 1.6.0
 - tensorRT == 6.0.1.5 (使用了onnx-tensorrt编译的库,因此,严格来说,这并不是标准的6.0.1.5)
 - cuda == 10.1.168

#### 关于torch中的 interpolate(也即Upsample)操作的转换问题

1. torch, onnx, tensorRT 对于interpolate 这个操作的支持度不太一致,不同版本之间也存在兼容问题

2. 对于interpolate mode = 'nearest' 在以下环境是ok的
     
     - torch >=1.1.0 (torch.onnx opset-9)
     - onnx >= 1.5.0
     - tensorRT >= 6.0.1.5 

#### 关于backbone的问题
1. resnet系列都没问题
2. MobilenNetV3 修改 `HardSigmoid()` 的实现后可以转到tensorRT 的engine,但有如下warning(rand输入推理的时候结果一致,但是真实数据推理结果不一致!). 而onnxruntime需要>=1.6.0版本才能运行

```
Unsupported ONNX data type: DOUBLE (11)
Unsupported ONNX data type: DOUBLE (11)
Unsupported ONNX data type: DOUBLE (11)
Unsupported ONNX data type: DOUBLE (11)
Unsupported ONNX data type: DOUBLE (11)
Unsupported ONNX data type: DOUBLE (11)
Unsupported ONNX data type: DOUBLE (11)
Unsupported ONNX data type: DOUBLE (11)

```

#### 关于 tensorRT6.0.1.5 与 onnx-tensorrt 
1. 直接使用原生的tensorRT6.0.1.5 经常会有importmodel error 或者是 input / output 的错误
2. onnx-tensorrt 编译的onnxparser库,解决了这类问题,因此兼容性更好,例如上述的mobilenetV3的 unsupported warning,增加onnx-tensorrt后没有warning,结果正确