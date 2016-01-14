DAN
================

```yaml
   pipeline:
     - svd1
     - prune

   config:
     svd1:
       command: 'svd_tool'
       input_proto: '../VGG16ORI_new.prototxt'
       input_caffemodel: '../VGG16ORI_new.caffemodel'
       output_proto: './try4.prototxt'
       output_caffemodel: './try4.caffemodel'
       pre-validate: True
       layers:
         - 'fc6, rank, 512'
    prune:
      command: 'prune_tool'
      input_proto: './try4.prototxt'
      input_caffemodel: './try4.caffemodel'
      output_caffemodel: './try4_after_prune.caffemodel'
      conditions:
        - [False, 'conv', 0.5]
        - [True, 'fc[0-6]*', 0.75]
```

svd_tool
----------------
svd tool for fc layers of the caffe models.

配置文件示例见 `src/dan/config_example.yaml`

conv_tool
----------------
Decomposition tool for conv layers of the caffe models.


prune_tool
----------------
Prune tool for layers of caffe network model.

配置文件示例见 `src/dan/config_prune.yaml`


quantize_tool
----------------
Quantize tool for layers of caffe network model.

配置文件示例见 `src/dan/config_quan.yaml`

nonmodel_tool
----------------
无模型剪枝与量化工具. 输入npz格式的文件, 不需提供prototxt/caffemodel, 每一层的系数用一个大的一维向量存储. 

配置文件示例见 `src/dan/config_fool.yaml`

与无模型压缩工具配套的 ``caffemodel -> npz`` 以及 ``bin -> caffemodel`` 转换工具见 [dan-tools仓库](https://github.com/angel-eye/dan-tools)


运行示例
----------------

`dan -f <path-to-your-config-file> -c <path-to-your-pycaffe> --quiet-caffe`

`svd_tool --quiet-caffe -c <path to your pycaffe> --input-proto <input prototxt file> --input-caffemodel <input caffemodel file> --output-proto <output prototxt file> --output-caffemodel <output caffemodel file> -l fc6`


下面是将vgg16的网络的fc7这一层做svd分解，保留奇异值最大的400维，构成的新的网络的示例: 


    $ svd_tool -c my/path/to/caffe/python --quiet-caffe --input-proto mypathto/VGG16ORI.prototxt --input-caffemodel mypathto/VGG16ORI.caffemodel --output-proto ./vgg16_rank400_fc7.prototxt --output-caffemodel ./vgg16_rank400_fc7.caffemodel -l fc7,rank,400


使用须知
----------------

并不是所有子工具都提供命令行接口, 尽量使用dan和dan的配置文件形式运行工具.