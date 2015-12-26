DAN
================

配置文件
----------------

配置文件示例 ( `config_example.yaml` ) :

>   pipeline:
>     #- conv1
>     - svd1
>
>   config:
>     svd1:
>       command: 'svd_tool'
>       input_proto: '../VGG16ORI_new.prototxt'
>       input_caffemodel: '../VGG16ORI_new.caffemodel'
>       output_proto: './try4.prototxt'
>       output_caffemodel: './try4.caffemodel'
>       pre-validate: True
>       layers:
>         - 'fc6, rank, 512'
>


svd_tool
----------------
svd tool for fc layers of the caffe models.


conv_tool
----------------
Decomposition tool for conv layers of the caffe models.


运行示例
----------------

`dan -f <path-to-your-config-file> -c <path-to-your-pycaffe> --quiet-caffe`

`svd_tool --input-proto <input prototxt file> --input-caffemodel <input caffemodel file> --output-proto <output prototxt file> --output-caffemodel <output caffemodel file> -c <path to your pycaffe> -l fc6 --quiet-caffe`
