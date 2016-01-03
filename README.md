DAN
================

配置文件
----------------

配置文件示例 ( `config_example.yaml` ) :

>   pipeline:
>
>     #- conv1
>
>     - svd1
>
>   config:
>
>     svd1:
>
>       command: 'svd_tool'
>
>       input_proto: '../VGG16ORI_new.prototxt'
>
>       input_caffemodel: '../VGG16ORI_new.caffemodel'
>
>       output_proto: './try4.prototxt'
>
>       output_caffemodel: './try4.caffemodel'
>
>       pre-validate: True
>
>       layers:
>
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

`svd_tool --quiet-caffe -c <path to your pycaffe> --input-proto <input prototxt file> --input-caffemodel <input caffemodel file> --output-proto <output prototxt file> --output-caffemodel <output caffemodel file> -l fc6`


下面是将vgg16的网络的fc7这一层做svd分解，保留奇异值最大的400维，构成的新的网络的示例: 


    $ svd_tool -c my/path/to/caffe/python --quiet-caffe --input-proto ../VGG16ORI_new.prototxt --input-caffemodel ../VGG16ORI_new.caffemodel --output-proto ./vgg16_rank400_fc7.prototxt --output-caffemodel ./vgg16_rank400_fc7.caffemodel -l fc7,rank,400
