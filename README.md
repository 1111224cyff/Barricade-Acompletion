# Barricade amodal completion

本项目基于poisson图像编辑技术合成图像，并使用yolov8进行实例分割，最终数据输入ASBU网络实现amodal补全。

## 1. Installation

```python
# 搭建环境
conda create -n acomp_barricade python=3.9
conda activate acomp_barricade
# 安装依赖项
pip install -r requirements.txt
```

## 2. Generate synthetic images

根据输入的src，tgt数据，进行图像融合，并生成对应标签

```
# 运行sh文件
bash run_fpie.sh
```

可根据实际需求在run_fpie.sh中调整变量

```
# 示例：迭代2w次，使用源图梯度
python run_fpie.py -n 20000 -g src
```

根据需要的图像上限可设置生成图片数量上限，程序可手动终止

## 3. Preprocess

数据预处理，生成用于amodal补全网络训练和验证的数据集，数据集包含按比例划分的图片及对应的json文件

```
bash preprocess.sh
```

## 4. Train

```
bash train.sh
```

## 5. Amodal completion

```
bash infer.sh
```

## Config

可根据实际需求修改config.yaml中的内容，示例的config.yaml如下：

```
# 项目的根目录路径
preprocess_data_dir: "./data/preprocess"
data_dir: "barricade"

# 图片的俯仰角分类
pitch_angles:
  - pitch0-30
  - pitch30-50
  - pitch50-80

# 涉及的物体类别
categories:
  0: "excavator"
  1: "plastic_fence"
  2: "steel_fence"
  3: "person"

# 划分训练和验证集的比例
split_ratio:
  0.7
```

