import time
import os

from fpie.args import get_args
from fpie.io import *
from fpie.process import BaseProcessor, EquProcessor
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetPaths:
    """管理路径的通用类"""
    base_dir: str
    pitch: Optional[str] = None

    def images_dir(self) -> str:
        """返回 images 目录路径"""
        if self.pitch:
            return os.path.join(self.base_dir, self.pitch, "images")
        return os.path.join(self.base_dir, "images")

    def labels_dir(self) -> str:
        """返回 labels 目录路径"""
        if self.pitch:
            return os.path.join(self.base_dir, self.pitch, "labels")
        return os.path.join(self.base_dir, "labels")

    def masking_dir(self) -> str:
        """返回 masking-out 目录路径"""
        if self.pitch:
            return os.path.join(self.base_dir, self.pitch, "masking-out")
        return os.path.join(self.base_dir, "masking-out")

    def create_directories(self):
        """创建必要的目录"""
        os.makedirs(self.images_dir(), exist_ok=True)
        os.makedirs(self.labels_dir(), exist_ok=True)
        if self.pitch:  # 仅在有 pitch 时创建 masking 目录
            os.makedirs(self.masking_dir(), exist_ok=True)


def process_images(
        args,
        src_dir: DatasetPaths,
        tgt_dir: DatasetPaths,
        proc,
        output_images_dir: str,
        output_labels_dir: str,
        count: int):
    """处理图片的逻辑"""

    src_images = get_file_paths(src_dir.images_dir(), (".jpg", ".png", ".jpeg"))
    tgt_images = get_file_paths(tgt_dir.images_dir(), (".jpg", ".png", ".jpeg"))
    output_src_labels_dir = os.path.join(output_labels_dir, "src_labels")
    output_tgt_labels_dir = os.path.join(output_labels_dir, "tgt_labels")
    merged_labels_dir = os.path.join(output_labels_dir, "merged_labels")

    # 在labels文件夹下创建子目录，分别存放src,tgt和合并的标注文件
    os.makedirs(output_src_labels_dir, exist_ok=True)
    os.makedirs(output_tgt_labels_dir, exist_ok=True)
    os.makedirs(merged_labels_dir, exist_ok=True)

    for src_image in src_images:
        # 获取每个图片的绝对路径
        src_img_path = os.path.join(src_dir.images_dir(), src_image)
        img_fn = os.path.splitext(src_image)[0]
        src_label_path = os.path.join(src_dir.labels_dir(), f"{img_fn}.txt")
        src_mask_path = os.path.join(src_dir.masking_dir(), f"{img_fn}.png")

        # 为每一次src对象创建蒙版,输出到masking文件夹可视化
        create_mask(src_img_path, src_label_path, src_dir.masking_dir())

        # 检查蒙版是否全黑
        mask = read_image(src_mask_path)
        if np.all(mask == 0):
            print(f"Skipping all-black mask for image {img_fn}")
            continue

        for tgt_image in tgt_images:
            tgt_image_path = os.path.join(tgt_dir.images_dir(), tgt_image)
            tgt_fn = os.path.splitext(tgt_image)[0]
            tgt_label_path = os.path.join(tgt_dir.labels_dir(), f"{tgt_fn}.txt") #####

            # 读取图像数据
            src, mask, tgt = read_images(src_img_path, src_mask_path, tgt_image_path)
            n, tgt_x, tgt_y = proc.reset(src, mask, tgt, (0, 0))
            print(f"# of vars: {n}")
            proc.sync()

            if proc.root:
                result = tgt
                t = time.time()
            if args.p == 0:
                args.p = args.n

            # 以下循环是输出每一步的图片
            for i in range(0, args.n, args.p):
                if proc.root:
                    result, err = proc.step(args.p)  # type: ignore
                    print(f"Iter {i + args.p}, abs error {err}")
                    if i + args.p < args.n:
                        write_image(f"iter{i + args.p:05d}.png", result)
                else:
                    proc.step(args.p)

            # 这是输出结果图片部分代码
            if proc.root:
                dt = time.time() - t
                print(f"Time elapsed: {dt:.4f}s")

            # 写入输出图像和标签
            write_image(output_images_dir, count, result)
            write_labels(tgt, tgt_x, tgt_y, output_src_labels_dir, count)
            copy_label(tgt_label_path, output_tgt_labels_dir, count)
            merge_txt_to_gt(output_src_labels_dir, output_tgt_labels_dir, merged_labels_dir, count)
            print(f"Successfully processed {count}.png")
            count += 1

    return count


def main() -> None:
    count = 0
    args = get_args("cli")
    config = load_config("config.yaml")
    # 从YAML文件中获取项目根目录路径及pitch_angles
    data_dir = config.get('preprocess_data_dir', "./data/preprocess")
    pitch_list = config.get('pitch_angles')

    if not pitch_list:
        print("Warning: No pitch angles found.")
        return

    proc: BaseProcessor
    if args.method == "equ":
        proc = EquProcessor(
            args.gradient,
            args.backend,
            args.cpu,
            args.mpi_sync_interval,
            args.block_size,
        )
    if proc.root:
        print(
            f"Successfully initialize PIE {args.method} solver "
            f"with {args.backend} backend"
        )

        # 遍历所有文件，重复进行read_images操作
        for pitch_angle in pitch_list:
            # 原图、原图标签、蒙版、目标图、输出、保存挖掘机掩码文件目录
            src_dir = DatasetPaths(os.path.join(data_dir, "src"), pitch_angle)
            tgt_dir = DatasetPaths(os.path.join(data_dir, "tgt"), pitch_angle)
            output_dir = DatasetPaths(os.path.join(data_dir, "synthetic_data"))
            output_dir.create_directories()

            count = process_images(
                args,
                src_dir,
                tgt_dir,
                proc,
                output_dir.images_dir(),
                output_dir.labels_dir(),
                count)

