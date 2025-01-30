import argparse
import yaml
import os
import json
import numpy as np
from PIL import Image
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
sys.path.append('.')
from datasets import reader
import models
import inference as infer
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import torch

import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--load-model', required=True, type=str)
    parser.add_argument('--order-method', required=True, type=str)
    parser.add_argument('--amodal-method', required=True, type=str)
    parser.add_argument('--order-th', default=0.1, type=float)
    parser.add_argument('--amodal-th', default=0.2, type=float)
    parser.add_argument('--annotation', required=True, type=str)
    parser.add_argument('--image-root', required=True, type=str)
    parser.add_argument('--test-num', default=-1, type=int)
    parser.add_argument('--output', default=None, type=str)
    parser.add_argument('--dilate_kernel', default=0, type=int)
    args = parser.parse_args()
    return args

def main(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in config.items():
        setattr(args, k, v)

    if not hasattr(args, 'exp_path'):
        args.exp_path = os.path.dirname(args.config)

    tester = Tester(args)
    tester.run()

class Tester(object):
    def __init__(self, args):
        self.args = args
        self.prepare_data()

    def prepare_data(self):
        config = self.args.data
        dataset = config['dataset']
        self.data_root = self.args.image_root
        if dataset == 'COCOA':
            self.data_reader = reader.COCOADataset(self.args.annotation)
        else:
            if dataset == 'KINSNew':
                self.data_reader = reader.KINSNewDataset(
                    dataset, self.args.annotation)
            else:
                self.data_reader = reader.KINSLVISDataset(
                    dataset, self.args.annotation)
        self.data_length = self.data_reader.get_image_length()
        self.dataset = dataset
        if self.args.test_num != -1:
            self.data_length = self.args.test_num

    def prepare_model(self):
        self.model = models.__dict__[self.args.model['algo']](self.args.model, dist_model=False)
        self.model.load_state(self.args.load_model)
        self.model.switch_to('eval')

    def expand_bbox(self, bboxes):
        new_bboxes = []
        for bbox in bboxes:
            centerx = bbox[0] + bbox[2] / 2.
            centery = bbox[1] + bbox[3] / 2.
            size = max([np.sqrt(bbox[2] * bbox[3] * self.args.data['enlarge_box']),
                        bbox[2] * 1.1, bbox[3] * 1.1])
            new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
            new_bboxes.append(new_bbox)
        return np.array(new_bboxes)

    def run(self):
        self.prepare_model()
        self.infer()

    def infer(self):
        order_th = self.args.order_th  # 获取排序的阈值，用于确定对象间的前后关系。
        amodal_th = self.args.amodal_th  # 获取amodal预测的阈值。

        # 设置图像的转换流程，包括转换为张量和归一化处理。
        self.args.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.args.data['data_mean'], self.args.data['data_std'])
            ])

        segm_json_results = []  # 初始化用于存储结果的列表。
        self.count = 0  # 初始化一个计数器，用于跟踪处理的实例数量。

        # 初始化多个性能评价的计量工具。
        allpair_true_rec = utils.AverageMeter()
        allpair_rec = utils.AverageMeter()
        occpair_true_rec = utils.AverageMeter()
        occpair_rec = utils.AverageMeter()
        intersection_rec = utils.AverageMeter()
        union_rec = utils.AverageMeter()
        target_rec = utils.AverageMeter()
        inv_intersection_rec = utils.AverageMeter()
        inv_union_rec = utils.AverageMeter()

        list_acc, list_iou = [], []
        list_inv_iou = []


        # 遍历数据集中的所有图像进行处理，这里的i代表图片
        for i in range(self.data_length):
            # 这里返回的是一堆列表，即包含这张图中的所有物体的信息
            # 这里返回的category恒为1,数量与每张图里的物体匹配
            modal, category, bboxes, amodal_gt, image_fn = self.data_reader.get_image_instances(i, with_gt=True)
            # modal, category, bboxes, amodal_gt, image_fn = self.data_reader.get_image_instances(i, with_gt=False)
            # print(f'i:{i} category:{category}')

            # 加载和处理图像。
            image = Image.open(os.path.join(self.data_root, image_fn)).convert('RGB')
            if image.size[0] != modal.shape[2] or image.size[1] != modal.shape[1]:
                image = image.resize((modal.shape[2], modal.shape[1]))  # 确保图像尺寸与模态掩码匹配。

            image = np.array(image)  # 转换图像为numpy数组。
            h, w = image.shape[:2]
            bboxes = self.expand_bbox(bboxes)  # 对边界框进行扩展。

            # 计算地面真实的前后关系矩阵。
            gt_order_matrix = infer.infer_gt_order(modal, amodal_gt)


            # 根据指定的方法计算对象的前后关系。
            if self.args.order_method == 'area':
                order_matrix = infer.infer_order_area(modal, above='smaller' if self.args.data['dataset'] == 'COCOA' else 'larger')
            elif self.args.order_method == 'yaxis':
                order_matrix = infer.infer_order_yaxis(modal)
            elif self.args.order_method == 'convex':
                order_matrix = infer.infer_order_convex(modal)
            elif self.args.order_method == 'ours':
                order_matrix = infer.infer_order(self.model, image, modal, category, bboxes,
                    use_rgb=self.args.model['use_rgb'], th=order_th, dilate_kernel=self.args.dilate_kernel,
                    input_size=256, min_input_size=16, interp='nearest', debug_info=False, args=self.args)
            else:
                raise Exception('No such order method: {}'.format(self.args.order_method))

            # 推断amodal掩码。
            if self.args.amodal_method == 'raw':
                amodal_pred = modal.copy()  # 如果方法为'raw'，直接使用模态掩码作为amodal掩码。
            elif self.args.amodal_method in ['ours_nog', 'ours_parents', 'ours', 'sup']:
                # 根据不同的amodal方法来预测amodal掩码，这里返回的都是包含各个物体的列表
                amodal_patches_pred = infer.infer_amodal(
                    self.model, image, modal, category, bboxes, order_matrix,
                    use_rgb=self.args.model['use_rgb'], th=amodal_th, dilate_kernel=self.args.dilate_kernel,
                    input_size=256, min_input_size=16, interp='linear',
                    order_grounded=self.args.amodal_method not in ['ours_nog', 'sup'],
                    debug_info=False, args=self.args)
                amodal_pred = infer.patch_to_fullimage(
                    amodal_patches_pred, bboxes, h, w, interp='linear')
            else:
                raise Exception("No such amodal method: {}".format(self.args.amodal_method))

            print('Loading: ', f"{i+1}/{self.data_length}")

            # 计算排序准确性和IOU等统计数据。
            allpair_true, allpair, occpair_true, occpair, _ = infer.eval_order(order_matrix, gt_order_matrix)
            allpair_true_rec.update(allpair_true)
            allpair_rec.update(allpair)
            occpair_true_rec.update(occpair_true)
            occpair_rec.update(occpair)

            intersection = ((amodal_pred == 1) & (amodal_gt == 1)).sum()
            union = ((amodal_pred == 1) | (amodal_gt == 1)).sum()
            target = (amodal_gt == 1).sum()
            intersection_rec.update(intersection)
            union_rec.update(union)
            target_rec.update(target)

            # 特别为不可见部分计算IOU。
            inv_intersection = ((amodal_pred == 1) & (amodal_gt == 1) & (modal == 0)).sum()
            inv_union = (((amodal_pred == 1) | (amodal_gt == 1)) & (modal == 0)).sum()
            inv_intersection_rec.update(inv_intersection)
            inv_union_rec.update(inv_union)

            # 保存精度和IOU计算结果。
            list_acc.append(occpair_true / (occpair + 1e-6))
            list_iou.append(intersection / (union + 1e-6))
            list_inv_iou.append(inv_intersection / (inv_union + 1e-6))

            # 生成输出结果。
            # 这个i是图像的张数
            # segm_json_results.extend(self.make_KINS_output(i, amodal_pred, category))
             # 生成输出结果。
            segm_json_results.extend(self.make_COCO_output(i, amodal_pred, category, image_fn, image_id=i))

        # 打印总结统计数据。
        acc_allpair = allpair_true_rec.sum / float(allpair_rec.sum)
        acc_occpair = occpair_true_rec.sum / float(occpair_rec.sum)
        miou = intersection_rec.sum / (union_rec.sum + 1e-10)
        pacc = intersection_rec.sum / (target_rec.sum + 1e-10)
        inv_miou = inv_intersection_rec.sum / (inv_union_rec.sum + 1e-10)
        print("Evaluation results. acc_allpair: {:.5g}, acc_occpair: {:.5g}, mIoU: {:.5g}, pAcc: {:.5g}, inv_mIoU: {:.5g}".format(
            acc_allpair, acc_occpair, miou, pacc, inv_miou))

        # 如果指定了输出路径，则保存结果到文件。
        if self.args.output:
            if not os.path.isdir(os.path.dirname(self.args.output)):
                os.makedirs(os.path.dirname(self.args.output))
            # with open(self.args.output, 'w') as f:
                # json.dump(segm_json_results, f)
            self.save_COCO_annotations(self.args.output, segm_json_results, self.args.data['categories'])



    def make_KINS_output(self, idx, amodal_pred, category):
        results = []
        for i in range(amodal_pred.shape[0]):
            data = dict()
            rle = maskUtils.encode(
                np.array(amodal_pred[i, :, :, np.newaxis], order='F'))[0]
            if hasattr(self.data_reader, 'img_ids'):
                data['image_id'] = self.data_reader.img_ids[idx]
            data['category_id'] = category[i].item()
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode()
            data['segmentation'] = rle
            data['bbox'] = utils.mask_to_bbox(amodal_pred[i, :, :])
            data['area'] = float(data['bbox'][2] * data['bbox'][3])
            data['iscrowd'] = 0
            data['score'] = 1.
            data['id'] = self.count
            results.append(data)
            self.count += 1

        return results
    

    def make_COCO_output(self, idx, amodal_pred, category, image_fn, image_id):
        results = []
        # 这个i是每幅图像中的第i个物体
        for i in range(amodal_pred.shape[0]):
            data = dict()
            rle = maskUtils.encode(
                np.array(amodal_pred[i, :, :, np.newaxis], order='F'))[0]
            data['image_id'] = image_id + 1
            data['category_id'] = category[i].item()
            mask = maskUtils.decode(rle)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:       
                print(f'imageID:{image_id + 1} {image_fn} {i} Object No contours found!')
                print(rle)
                # plt.imshow(mask)
                # plt.title('MASK')
                # plt.axis('off')
                # # 显示图形
                # plt.show()
                continue
                

            elif len(contours) == 1:
                data['iscrowd'] = 0
                polygon = [contour.flatten().tolist() for contour in contours if contour.size >= 6]
                data['segmentation'] = polygon
            else:
                data['iscrowd'] = 1
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode()
                data['segmentation'] = rle

            data['bbox'] = utils.mask_to_bbox(amodal_pred[i, :, :])
            data['area'] = float(data['bbox'][2] * data['bbox'][3])
            data['id'] = self.count + 1
            data['file_name'] = image_fn
            data['width'], data['height'] = amodal_pred.shape[2], amodal_pred.shape[1]
            results.append(data)
            self.count += 1

        return results


    def save_COCO_annotations(self, output_file, results, coco_classes):
        # 创建COCO格式的输出结构
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # 添加类别信息
        categories = [{"id": i, "name": name} for i, name in enumerate(coco_classes, 1)]
        coco_output["categories"].extend(categories)

        # 添加图像和注释信息
        for result in results:
            # 添加图像信息
            image_info = {
                "id": result['image_id'],
                "file_name": result['file_name'],
                "width": result['width'],
                "height": result['height']
            }
            if image_info not in coco_output["images"]:
                coco_output["images"].append(image_info)
            
            # 添加注释信息
            annotation_info = {
                "id": result['id'],
                "image_id": result['image_id'],
                "category_id": result['category_id'],
                "segmentation": result['segmentation'],
                "bbox": result['bbox'],
                "area": result['area'],
                "iscrowd": result['iscrowd']
            }
            coco_output["annotations"].append(annotation_info)

        # 保存为JSON文件
        with open(output_file, 'w') as f:
            json.dump(coco_output, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
