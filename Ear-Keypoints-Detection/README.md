# EKD——Ear-Keypoints-Detection using MMDet&MMPose

## 项目简介

EKD是一个利用MMDet（开源目标检测工具）和MMPose（开源多人姿态估计工具）技术实现的耳部关键点识别项目。该项目旨在通过计算机视觉技术，准确地识别人类耳朵上的关键点，为后续的耳部相关应用提供支持。

耳部关键点识别在许多领域具有广泛的应用。例如，在医学领域，耳朵的形态特征可以提供有关个体身体健康状况的重要信息。在安全领域，耳部关键点识别可以用于人脸识别和身份验证系统的辅助功能，提高识别准确性。此外，在人机交互和虚拟现实领域，耳朵的姿态和动作信息可以被用于用户情绪和意图的识别。

本项目采用MMDet和MMPose两个开源工具的组合。MMDet是一个基于深度学习的目标检测框架，能够高效准确地检测图像中的目标物体。MMPose是一个多人姿态估计工具，可以估计图像中多个人的关键点位置和姿态信息。

在项目实施过程中，我们首先使用MMDet进行耳朵的目标检测，从输入图像中准确地定位出耳朵区域。然后，利用MMPose对耳朵区域进行关键点估计，确定耳朵的特定位置。通过这两个工具的结合使用，我们可以实现耳部关键点的准确识别。

### 优化计划

1. 获得更多样本的标注数据集：为了提高模型的性能和泛化能力，我们计划获得更多的标注数据集来训练和验证模型。这可以通过扩大数据收集范围、与合作伙伴合作或利用众包平台等方式实现。更多的样本数据将有助于模型更好地理解不同耳部关键点的变化和多样性。

2. 进一步提升模型性能：我们将继续优化和改进模型以提高其性能。这可以包括使用更先进的深度学习架构、增加网络的深度和宽度、引入更有效的正则化方法等。我们还可以探索数据增强技术，如旋转、缩放、镜像等，以增加模型对不同姿态和角度的适应能力。

3. 为MMPose的visualizer增加show_kpt_name功能：为了更好地展示和可视化耳部关键点的识别结果，我们计划为MMPose的visualizer工具添加show_kpt_name功能（目前只有show_kpt_id）。这将允许我们在可视化结果中显示每个关键点的名称，提高结果的可读性和理解性。

## 创建环境

```shell
# 创建conda环境
conda create -n mmwork python==3.8.16
conda activate mmwork
conda install mamba

# 安装pytorch（根据个人硬件配置，从pytorch选择下载指令）
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 通过openmim安装mmcv和mmdet
mamba install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

# 解压数据集
cd mm/mmhomeworke2
mkdir data
mkdir -p work/result/mmdet
mkdir -p work/result/mmpose
unzip 'data/MMPosehomework.zip' -d 'data/MMPosehomework'
mv 'data/MMPosehomework/样例config配置文件' 'data/MMPosehomework/example'
unzip 'data/MMPosehomework/Ear210_Dataset_coco.zip' -d 'data/MMPosehomework/Ear210_Dataset_coco'

# 准备mmpose和mmdet
cd work
git clone git@github.com:open-mmlab/mmpose.git
git clone git@github.com:open-mmlab/mmdetection.git
cd mmpose
pip install -v -e .
# 以下两行为了避免numpy报错
pip uninstall xtcocotools -y
pip install git+https://github.com/jin-s13/xtcocoapi
# 也是为了解决报错
mamba install scipy
# 同样为了解决报错
pip install -U albumentations --no-binary qudida,albumentations
```

## 修改config文件

打开`rtmdet_tiny_ear.py`和`rtmpose-s-ear.py`

修改以下内容：

```python
data_root = </Absolute/Path/of/Ear210_Keypoint_Dataset_coco/>
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 8
```

## 训练模型

```python
# 移动到work/mmdetection目录
cd /media/ders/mazhiming/mm/mmhomeworke2/work/result/mmdet

# 开始训练，CUDA_VISIBLE_DEVICES根据实际情况修改（选用的显卡的编号）
CUDA_VISIBLE_DEVICES=1 PORT=8082 nohup python \
    </Absolute/Path/of/mmdetection/tools/train.py> \
    </Absolute/Path/of/data/MMPosehomework/example/rtmdet_tiny_ear.py> \
    --work-dir </Absolute/Path/of/work/result/mmdet> > output.log 2>&1 &
```

```python
# 移动到work/mmpose目录
cd /media/ders/mazhiming/mm/mmhomeworke2/work/result/mmpose

# 开始训练，CUDA_VISIBLE_DEVICES根据实际情况修改（选用的显卡的编号）
CUDA_VISIBLE_DEVICES=2 PORT=8083 nohup python \
    </Absolute/Path/of/mmpose/tools/train.py> \
    </Absolute/Path/of/data/MMPosehomework/example/rtmpose-s-ear.py> \
    --work-dir </Absolute/Path/of/work/result/mmpose> > output.log 2>&1 &
```

```python
# 中途断开，调参数再训，CUDA_VISIBLE_DEVICES根据实际情况修改（选用的显卡的编号）
CUDA_VISIBLE_DEVICES=2 PORT=8083 nohup python \
    </Absolute/Path/of/mmpose/tools/train.py> \
    </Absolute/Path/of/data/MMPosehomework/example/rtmpose-s-ear.py> \
    --work-dir </Absolute/Path/of/work/result/mmpose> \
    --resume </Absolute/Path/of/pth> > output.log 2>&1 &
```

## 评估结果

### 日志分析

#### MMdet (epoch200 + epoch200)

>   Average Precision  (AP) @[ IoU=0.50:0.95 | area=  all | maxDets=100 ] = 0.822
>
>   Average Precision  (AP) @[ IoU=0.50    | area=  all | maxDets=100 ] = 0.967
>
>   Average Precision  (AP) @[ IoU=0.75    | area=  all | maxDets=100 ] = 0.967
>
>   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
>
>   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
>
>   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.822
>
>   Average Recall   (AR) @[ IoU=0.50:0.95 | area=  all | maxDets=  1 ] = 0.857
>
>   Average Recall   (AR) @[ IoU=0.50:0.95 | area=  all | maxDets= 10 ] = 0.857
>
>   Average Recall   (AR) @[ IoU=0.50:0.95 | area=  all | maxDets=100 ] = 0.857
>
>   Average Recall   (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
>
>   Average Recall   (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
>
>   Average Recall   (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.857
>
> 06/07 02:36:16 - mmengine - INFO - bbox_mAP_copypaste: 0.822 0.967 0.967 -1.000 -1.000 0.822
>
> 06/07 02:36:16 - mmengine - INFO - Epoch(val) [200][6/6]   coco/bbox_mAP: 0.8220  coco/bbox_mAP_50: 0.9670  coco/bbox_mAP_75: 0.9670  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.8220  data_time: 0.9648  time: 1.0142

![image-20230607113402164](assets/image-20230607113402164.png)

![image-20230607113409977](assets/image-20230607113409977.png)

![image-20230607113417914](assets/image-20230607113417914.png)

#### MMpose (epoch150 + lr调小为80%的epoch150)

>   Average Precision  (AP) @[ IoU=0.50:0.95 | area=  all | maxDets= 20 ] =  0.708
>
>   Average Precision  (AP) @[ IoU=0.50    | area=  all | maxDets= 20 ] =  1.000
>
>   Average Precision  (AP) @[ IoU=0.75    | area=  all | maxDets= 20 ] =  0.860
>
>   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
>
>   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.708
>
>   Average Recall   (AR) @[ IoU=0.50:0.95 | area=  all | maxDets= 20 ] =  0.745
>
>   Average Recall   (AR) @[ IoU=0.50    | area=  all | maxDets= 20 ] =  1.000
>
>   Average Recall   (AR) @[ IoU=0.75    | area=  all | maxDets= 20 ] =  0.881
>
>   Average Recall   (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
>
>   Average Recall   (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.745
>
> 06/07 02:26:24 - mmengine - INFO - Evaluating PCKAccuracy (normalized by ``"bbox_size"``)...
>
> 06/07 02:26:24 - mmengine - INFO - Evaluating AUC...
>
> 06/07 02:26:24 - mmengine - INFO - Evaluating NME...
>
> 06/07 02:26:24 - mmengine - INFO - Epoch(val) [150][6/6]   coco/AP: 0.708495  coco/AP .5: 1.000000  coco/AP .75: 0.859583  coco/AP (M): -1.000000  coco/AP (L): 0.708495  coco/AR: 0.745238  coco/AR .5: 1.000000  coco/AR .75: 0.880952  coco/AR (M): -1.000000  coco/AR (L): 0.745238  PCK: 0.963719  AUC: 0.127494  NME: 0.043727  data_time: 1.753416  time: 1.782141

![image-20230607113437693](assets/image-20230607113437693.png)

![image-20230607113442908](assets/image-20230607113442908.png)

![image-20230607113449343](assets/image-20230607113449343.png)

![image-20230607113457802](assets/image-20230607113457802.png)

### 可视化

<div>
    <div style="float: left; width: 50%;">
        <div style="width: 100%;">
    		<img src="assets/test_ear.jpg" alt="test_ear" style="zoom: 33.3%;" />
        </div>
        <div style="width: 100%;">
            <img src="assets/test_ear_mmdet.jpg" alt="test_ear_mmdet" style="zoom: 33.3%;" />
        </div>
    </div>
    <div style="float: right; width: 50%;">
        <div style="width: 100%;">
            <img src="assets/test_ear_mmpose.jpg" alt="test_ear_mmpose" style="zoom: 33.3%;" />
        </div>
    </div>
</div>



























































> ```shell
> # mmdet单图推理，CUDA_VISIBLE_DEVICES根据实际情况修改（选用的显卡的编号）
> CUDA_VISIBLE_DEVICES=1 PORT=8082 python \
>     <path/to/mmdetection/demo/image_demo.py> \
>     <path/to/test_ear.jpg> \
>     <path/to/rtmdet_tiny_ear.py> \
>     --weights <path/to/best/mmdet/pth> \
>     --out-dir <path/to/output/directory> \
>     --device <cpu or cuda or cuda:0 …… > \
>     > output.log 2>&1
> ```
>
> ```shell
> # mmpose单图推理，CUDA_VISIBLE_DEVICES根据实际情况修改（选用的显卡的编号）
> CUDA_VISIBLE_DEVICES=1 PORT=8082 python \
>     <path/to/mmpose/demo/image_demo.py> \
>     <path/to/rtmdet_tiny_ear.py> \
>     <path/to/best/mmdet/pth> \
>     <path/to/rtmpose-s-ear.py> \
>     <path/to/best/mmpose/pth> \
>     --input <path/to/test_ear.jpg> \
>     --output-root <path/to/output/directory> \
>     --save-predictions \
>     --device <cpu or cuda or cuda:0 …… > \
>     --bbox-thr 0.5 \
>     --kpt-thr 0.5 \
>     --nms-thr 0.3 \
>     --radius 8 \
>     --thickness 7 \
>     --draw-bbox \
>     --draw-heatmap \
>     --show-kpt-idx \
>     > output.log 2>&1
> ```

参考资料：

mmdet

- [mmdet-推理demo](https://mmdetection.readthedocs.io/zh_CN/latest/user_guides/inference.html)

mmpose:

- [mmpose-推理demo](https://mmpose.readthedocs.io/zh_CN/latest/demos.html)
- [mmpose-参数设置](https://mmpose.readthedocs.io/zh_CN/latest/user_guides/inference.html)

## 遇到的坑

### 环境配置

> ```python
> # 以下两行为了避免numpy报错
> pip uninstall xtcocotools -y
> pip install git+https://github.com/jin-s13/xtcocoapi
> # 也是为了解决报错
> mamba install scipy
> # 同样为了解决报错
> pip install -U albumentations --no-binary qudida,albumentations
> ```

从上往下依次来自

1. [Numpy error · Issue #2195 · open-mmlab/mmpose · GitHub](https://github.com/open-mmlab/mmpose/issues/2195)
2. 自行解决
3. https://github.com/open-mmlab/mmpose/pull/1184

### 创建Config的方法

第一次自定义config时，可以从源码的`configs`文件夹中自行选择组件进行组装再使用，使用无论成功与否，都会在工作目录下生成完整的config，可以基于这个config再进行修改

### Config参数设置

MMpose示例config的默认参数中，CosineAnnealingLR开始的epoch偏晚（150epoch），调早一些效果可能更好

### 调试Config的方法

```shell
CUDA_VISIBLE_DEVICES=1 PORT=8082 python \
	<your_command> \
	> output.log 2>&1
```

在终端以此格式调用命令，运行日志打印到当前目录下，并且可以根据终端状态检查命令是否仍存活
