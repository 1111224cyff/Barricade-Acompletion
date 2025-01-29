import os
import json
import random
import cv2
import numpy as np
from fpie.io import load_config
import shutil


def calculate_area(points):
    """计算多边形面积"""
    return int(cv2.contourArea(np.array(points, dtype=np.int32)))


def convert_normalized_to_pixel(coords, width, height):
    """将归一化坐标转换为像素坐标"""
    return [(int(coords[i] * width), int(coords[i + 1] * height)) for i in range(0, len(coords), 2)]


def process_annotations(image_dir, labels_dir, categories, output_json=None, mask_output_dir=None, data_dir="barricade", copy_images=False):
    """
    解析图像和标注信息，生成 JSON 格式的标注文件，并可选生成掩码图像。
    只有在 `copy_images=True` 时，才会将 image_dir 中的图片复制到 ./data/barricade/infer/ 目录。

    参数：
        image_dir (str): 图片文件夹路径
        labels_dir (str): 标签文件夹路径
        categories (dict): 类别映射信息
        output_json (str): 生成的 JSON 文件路径 (可选)
        mask_output_dir (str): 掩码文件夹路径 (可选)
        data_dir (str): 数据存放目录
        copy_images (bool): 是否复制图片到 infer 目录，默认为 False

    返回：
        dict: 包含图片和标注信息的字典
    """
    images, annotations = [], []
    image_id, annotation_id = 1, 1

    # 目标存放目录 ../data/{data_dir}/infer/
    infer_dir = os.path.join("../data", data_dir, "infer")
    os.makedirs(infer_dir, exist_ok=True)  # 创建 infer 目录

    if mask_output_dir and not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)

    for file_name in os.listdir(labels_dir):
        if not file_name.endswith('.txt'):
            continue

        # 关联图片文件
        image_file = os.path.join(image_dir, file_name.replace('.txt', '.png'))
        if not os.path.exists(image_file):
            continue

        # 目标图片路径
        target_image_path = os.path.join(infer_dir, os.path.basename(image_file))

        # **只有当 copy_images=True 时才复制图片**
        if copy_images:
            shutil.copy(image_file, target_image_path)
        else:
            target_image_path = image_file  # 直接使用原路径，不复制

        image = cv2.imread(target_image_path)
        height, width, _ = image.shape

        regions = []
        mask = np.zeros((height, width), dtype=np.uint8) if mask_output_dir else None

        with open(os.path.join(labels_dir, file_name), 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = int(parts[0])

                if label not in categories:
                    continue  # 跳过未定义类别

                name = categories[label]
                normalized_coords = list(map(float, parts[1:]))

                if len(normalized_coords) < 6 or len(normalized_coords) % 2 != 0:
                    continue  # 确保至少包含3个点（6个值）

                pixel_coords = convert_normalized_to_pixel(normalized_coords, width, height)
                area = calculate_area(pixel_coords)
                segmentation = [coord for point in pixel_coords for coord in point]

                region = {
                    'segmentation': segmentation,
                    'name': name,
                    'area': area,
                    'isStuff': False,
                    'occlude_rate': 0.0
                }
                regions.append(region)

                # 仅针对特定类别生成掩码
                if mask_output_dir and label in [1, 2]:  # 1=plastic_fence, 2=steel_fence
                    cv2.fillPoly(mask, [np.array(pixel_coords, dtype=np.int32)], 255)

        images.append({
            'file_name': os.path.basename(image_file),
            'height': height,
            'width': width,
            'id': image_id
        })

        annotations.append({
            'regions': regions,
            'image_id': image_id,
            'id': annotation_id,
            'size': len(regions)
        })

        # 保存掩码图像
        if mask_output_dir:
            mask_output_file = os.path.join(mask_output_dir, file_name.replace('.txt', '.png'))
            cv2.imwrite(mask_output_file, mask)

        image_id += 1
        annotation_id += 1

    output_data = {
        'images': images,
        'annotations': annotations
    }

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, separators=(",", ": "))

        print(f"Annotations saved to {output_json}")

    if copy_images:
        print(f"Images copied to {infer_dir}")

    return output_data



def split_and_generate_json(synthetic_data_dir, categories, split_ratio=0.8, data_dir="barricade"):
    """
    根据整体 images 和 labels 文件夹划分数据集，并生成 COCO 格式 JSON 标注文件。
    同时将 train 和 val 的图片文件移动到 ../data/barricade/ 下，并将 JSON 放入 ../data/barricade/annotations/ 目录。

    参数：
        dataset_dir (str): 数据集根目录，包含 `images` 和 `labels` 文件夹。
        config_path (str): 配置文件路径，包含类别信息。
        split_ratio (float): 训练集划分比例，默认为 0.8。
    """

    images_dir = os.path.join(synthetic_data_dir, "images")
    labels_dir = os.path.join(synthetic_data_dir, "labels", "merged_labels")

    # 定义目标目录结构
    data_root_dir = os.path.join("../data", data_dir)
    annotations_dir = os.path.join("../data", data_dir, "annotations")

    # 创建目录
    os.makedirs(data_root_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # 创建 train 和 val 的子文件夹（仍然在原 images 目录下）
    train_dir = os.path.join(images_dir, "train")
    val_dir = os.path.join(images_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 获取所有图片文件名
    image_files = [file for file in os.listdir(images_dir) if file.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files)

    split_index = int(len(image_files) * split_ratio)
    splits = {"train": image_files[:split_index], "val": image_files[split_index:]}

    for split, files in splits.items():
        output_json = os.path.join(annotations_dir, f"{split}.json")

        for image_file in files:
            src_path = os.path.join(images_dir, image_file)
            dst_path = os.path.join(train_dir if split == "train" else val_dir, image_file)
            shutil.move(src_path, dst_path)

        process_annotations(
            image_dir=os.path.join(images_dir, split),
            labels_dir=labels_dir,
            categories=categories,
            output_json=output_json,
            data_dir=data_root_dir
        )

    # 移动 train 和 val 目录
    shutil.move(train_dir, os.path.join(data_root_dir, "train"))
    shutil.move(val_dir, os.path.join(data_root_dir, "val"))

    print(f"Images moved to {data_root_dir}")
    print(f"Annotations saved in {annotations_dir}")



# 示例调用
if __name__ == "__main__":
    synthetic_data_dir = r"..\data\preprocess\synthetic_data"
    config_path = r"..\config.yaml"

    # 读取配置文件，获取 categories
    config = load_config(config_path)
    categories = config.get('categories', {})
    split_ratio = config.get('split_ratio', 0.7)  # 允许配置文件覆盖 split_ratio
    data_dir = config.get('data_dir', 'barricade')

    # 处理训练集和验证集划分
    split_and_generate_json(synthetic_data_dir, categories, split_ratio, data_dir)

    # 生成用于ASBU推理数据（这里才复制图片！）
    image_dir = r"..\data\yolo_segment\images"
    labels_dir = r"..\data\yolo_segment\labels"
    output_path = os.path.join("../data", data_dir, "annotations", "infer.json")
    mask_output_dir = r"..\data\yolo_segment\mask"

    # 生成用于ASBU推测的json，**并且复制图片**
    process_annotations(image_dir,
                        labels_dir,
                        categories,
                        output_json=output_path,
                        mask_output_dir=mask_output_dir,
                        data_dir=data_dir,
                        copy_images=True  # ✅ 只有这里复制图片
                        )


