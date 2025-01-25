import os
import warnings
import yaml
import shutil
from typing import Tuple

import cv2
import numpy as np


def read_image(name: str) -> np.ndarray:
    # 读单张图像
    img = cv2.imread(name)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif len(img.shape) == 4:
        img = img[..., :-1]
    return img


def write_image(output_dir: str, count: int, image: np.ndarray) -> None:
    # 我的输入是输出的文件目录，与源码不同，所以在此修改
    # 考虑到我还有标签的信息，我需要再保存新的图像名时，把对应的标注文件也用新的名字保存一份
    output_name = f'{count}.png'
    output_img_dir = os.path.join(output_dir, output_name)
    cv2.imwrite(output_img_dir, image)


def read_images(
        src_name: str,
        mask_name: str,
        tgt_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 调用read_image函数，读原图、蒙版、目标图，返回三个参数（ndarray格式）
    src, tgt = read_image(src_name), read_image(tgt_name)
    if os.path.exists(mask_name):
        mask = read_image(mask_name)
    else:
        warnings.warn("No mask file found, use default setting")
        mask = np.zeros_like(src) + 255
    return src, mask, tgt


def copy_label(label_path: str, output_dir: str, count: int) -> None:
    # 将合成后与输出图像同名的标签也放进output_dir
    new_label_name = f'{count}.txt'
    new_label_path = os.path.join(output_dir, new_label_name)
    try:
        shutil.copy(label_path, new_label_path)

    except Exception as e:
        print(f"复制文件时出错：{e}")

def get_file_paths(directory: str, extensions: tuple) -> list:
    """获取指定目录下的所有指定扩展名文件路径"""
    return [
        f for f in os.listdir(directory)
        if f.lower().endswith(extensions)
    ]

def create_mask(img_path: str, label_path: str, output_dir: str) -> None:
    # 读取原图和标签文件路径，生成蒙版图像
    img = read_image(img_path)
    # 读取文件名
    img_file = os.path.basename(img_path)
    img_name = os.path.splitext(img_file)[0]

    # 打开标签文件逐行读取
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 创建蒙版（只有黑白两种颜色，不需要彩色）
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for line in lines:
        label_data = line.strip().split()
        class_id = int(label_data[0])

        # 如果是挖掘机对象
        if class_id == 0:
            # 解析多边形坐标
            polygon_data = list(map(float, label_data[1:]))
            polygon_pts = np.array(
                [[(x * img.shape[1], y * img.shape[0]) for x, y in zip(polygon_data[0::2], polygon_data[1::2])]],
                dtype=np.int32)

            # 在蒙版上绘制填充的多边形
            cv2.fillPoly(mask, polygon_pts, 255)

    # 保存二值蒙版图像
    output_path = os.path.join(output_dir, img_name + '.png')
    cv2.imwrite(output_path, mask)


def load_config(config_path: str):
    """加载配置文件并返回配置字典"""
    with open(config_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def write_labels(tgt, tgt_x, tgt_y, output_dir, count):
    # 获取图像尺寸
    img_height, img_width = tgt.shape[:2]

    # 构建输出文件的路径
    label_name = f'{count}.txt'
    label_path = os.path.join(output_dir, label_name)

    # 创建二值掩码图像
    binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    binary_mask[tgt_x, tgt_y] = 255  # 将目标像素设为白色

    # 使用 cv2.findContours 提取边界轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果找到了轮廓，选择最大的一个作为边界
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # 将轮廓点归一化
        normalized_contour = [
            (pt[0][0] / img_width, pt[0][1] / img_height) for pt in largest_contour
        ]

        # 保存到文件
        with open(label_path, 'w') as file:
            # 开始写入类别编号
            file.write('0 ')
            # 写入归一化后的多边形顶点
            for x_normalized, y_normalized in normalized_contour:
                file.write(f"{x_normalized:.6f} {y_normalized:.6f} ")
            file.write('\n')



def _get_scale_factor(mask: np.ndarray, target: np.ndarray) -> float:
    """
    等比例缩放蒙版图像直到其尺寸不大于目标图像的尺寸。
    """
    # 初始化参数
    scale_factor = 1
    if mask.shape[0] >= target.shape[0] or mask.shape[1] >= target.shape[1]:
        # 初始化 scale_factor，确保蒙版图像等比例缩放到不超过目标图像
        scale_factor = min(target.shape[0] / mask.shape[0], target.shape[1] / mask.shape[1]) * 0.5
        if scale_factor < 0.1:
            print("Scale factor too small!")

    return scale_factor


def resize_mask_to_fit_target(
        cropped_mask: np.ndarray,
        tgt: np.ndarray,
        src: np.ndarray,
        mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将裁剪后的mask与目标图作比较，不要超出范围，超出范围就缩放"""
    # 输出缩放系数
    scale_factor = _get_scale_factor(cropped_mask, tgt)
    if scale_factor == 1:
        return mask, src
    else:
        # 确保蒙版图像是 uint8 类型
        mask = np.clip(mask, 0, 255).astype(np.uint8)
        # 将蒙版缩放
        new_mask_size = (round(mask.shape[1] * scale_factor), round(mask.shape[0] * scale_factor))
        mask = cv2.resize(mask, new_mask_size, interpolation=cv2.INTER_AREA)
        # 将缩放后的图像转换回 int32 类型
        mask = mask.astype(np.int32)
        # 将原图也一起缩放
        new_src_size = (round(src.shape[1] * scale_factor), round(src.shape[0] * scale_factor))
        src = cv2.resize(src, new_src_size, interpolation=cv2.INTER_AREA)
        return mask, src


def merge_txt_to_gt(src_dir: str, tgt_dir: str, merged_dir: str, count: int) -> None:
    """
    合并tgt和src的标注文件
    :param src_dir:
    :param tgt_dir:
    :param merged_dir:
    :param count:
    :return:
    """
    src_path = os.path.join(src_dir, f"{count}.txt")
    tgt_path = os.path.join(tgt_dir, f"{count}.txt")
    merged_path = os.path.join(merged_dir, f"{count}.txt")

    os.makedirs(merged_dir, exist_ok=True)

    with open(src_path, "r") as src_file, \
         open(tgt_path, "r") as tgt_file, \
         open(merged_path, "w") as merged_file:

        # 合并txt
        merged_file.writelines(src_file.readlines())
        merged_file.writelines(tgt_file.readlines())

        print(f"Files merged successfully into {count}.txt")




