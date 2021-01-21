# Readme
- ## songtype_id:
  - 1: 常见频率  
  - 4: 另一种常见频率  
- ## train_img:   
  - 没有给每个label画框框，直接拿来当样本集  
- ## train_tran:  
  - 给每个label画了框框，验证框框正确性 
- ## coco: 
  - 符合COCO格式的样本集
- ## label:
  - label.txt 自定义label文件
- ## coco_dataset_verify: 
  - 使用COCO API验证label正确性
- ## VOC: 
  - 符合VOC格式的样本集
- ## faster rcnn:  
  - copy https://github.com/facebookresearch/detectron2
  - license: Apache License 2.0
  - environment: 
    - CUDA
    - CuDNN
    - pytorch
    - Visual Studio 2019
    - ninja
  - setup:
    ```console
    python setup.py build develop
    ```
  - verify:
    ```python
    import detectron2
    detectron2.__version__
    ```
  - COCO API 有一个bug，修改后才能运行train_net.py
    ```console
    TypeError: 'numpy.float64' object cannot be interpreted as an integer
    ```
    报错点:  
    ```python
    self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
    self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
    ```
    改成
    ```python
    self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
    ```

- ## darknet:  
  - copy https://github.com/AlexeyAB/darknet
  - license: Do Anything
  - environment: 
    - 10.0 <= CUDA < 11.0 (这玩意儿用的opencv比较老，不支持CUDA 11.0)
  - setup:
    ```console
    设置环境变量 HTTP_PROXY HTTPS_PROXY 方便命令行使用代理（跑完了记得删了）
    cd [随便找个文件夹用来安装vcpkg]
    git clone https://github.com/microsoft/vcpkg
    设置vcpkg环境变量
    bootstrap-vcpkg.bat
    cd [项目文件夹]
    cd darknet
    powershell -ExecutionPolicy Bypass -File .\build.ps1
    ```
  - 如果显示 GPU not used
    ```console
    cd [darknet项目文件夹]
    删除build/*
    打开cmake重新generate
    打开build下的sln解决方案
    设置为release模式，生成解决方案
    将./Release/*文件复制到[darknet项目文件夹]
    ```
    训练: 
    ```console
    ./darknet.exe detector train data/kaggle.data cfg/yolov4-kaggle.cfg yolov4.conv.137 -map -json_port 8070 -mjpeg_port 8090
    ```



# 问题
- RCNN可以在样本集中预定义负样本，但是一般负样本指背景，看看FP样本怎么塞进去
- Faster RCNN 效果非常糟糕, AP = 0.3 
- 感觉给的标记有大量的漏标，还有给的 f 和实际偏差有点大，可能会影响目标检测训练
- 造成Faster RCNN效果糟糕可能的原因:  
  - 标记不完整
  - 标记缺失→标记冲突

# TodoList
- 验证label准确性
- 试试看做多标签分类问题
- 弄一个数据增强器，从原始音频添加白噪声，然后再转成Mel频谱图，放到训练集里头，大概像tensorflow里头的ImageDataGenerator一样？
- 搭一个或者copy一个Faster RCNN或者更新的目标检测网络，回头我会把VOC格式的训练集放上去，最好是全部用python写的，方便改，注意一下开源协议，如果要求你用它的就得全部开源就没法用，至少要等到比赛结束再开源;
- 现在数据集只有随机裁切的10s长度的音频片段，按照原来的想法还需要直接顺序裁切的，混起来做训练集。