import numpy as np
import sys
sys.path.append('.')

import cvbase as cvb
import pycocotools.mask as maskUtils
import utils

def read_KINS(ann):
    modal = maskUtils.decode(ann['inmodal_seg']) # HW, uint8, {0, 1}
    bbox = ann['inmodal_bbox'] # luwh
    category = ann['category_id']
    if 'score' in ann.keys():
        score = ann['score']
    else:
        score = 1.
    return modal, bbox, category, score

def read_LVIS(ann, h, w):
    segm = ann["segmentation"]
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann["segmentation"]
    bbox = ann['bbox'] # luwh
    category = ann['category_id']
    return maskUtils.decode(rle), bbox, category

def read_COCOA(ann, h, w):
    # 这个函数的作用是获取模态掩码及bbox
    # 在COCOA的标注文件中annotation下如果有遮挡的话，会有visible_mask和invisible_mask选项
    # 如果有visible_mask, 则modal是可见部分的掩码
    if 'visible_mask' in ann.keys():
        rle = [ann['visible_mask']]
    # 如果没有visible_mask, 应该是默认了它没有被遮挡
    else:
        rles = maskUtils.frPyObjects([ann['segmentation']], h, w)
        rle = maskUtils.merge(rles)
    # modal应该就是模态掩码，rle应该是包含了原图像的宽高信息和掩码信息的
    # 应该就是全黑的原图，然后掩码区域是白色这种理解
    modal = maskUtils.decode(rle).squeeze()
    if np.all(modal != 1):
        # if the object if fully occluded by others,
        # use amodal bbox as an approximated location,
        # note that it will produce random amodal results.
        amodal = maskUtils.decode(maskUtils.merge(
            maskUtils.frPyObjects([ann['segmentation']], h, w)))
        bbox = utils.mask_to_bbox(amodal)
    else:
    # 这里将mask转换为了bbox
        bbox = utils.mask_to_bbox(modal)
    # # 读取类别信息
    category_name = ann['name']
    if category_name.lower() == "excavator":
        category = 1
    elif category_name.lower() == "plastic_fence" :
        category = 2
    elif category_name.lower() == "steel_fence":
        category = 3
    elif category_name.lower() == "opening":
        category = 4
    else:
        category = 5

    return modal, bbox, category 
    # return modal, bbox, 1 # category as constant 1


class COCOADataset(object):

    def __init__(self, annot_fn):
        data = cvb.load(annot_fn)
        # 这里读取的是.json文件中的images和annotations键值对应的图片和标注信息
        self.images_info = data['images']
        self.annot_info = data['annotations']

        self.indexing = []
        # 注意这里只对标注的列表做遍历
        for i, ann in enumerate(self.annot_info):
            for j in range(len(ann['regions'])):
                # 这里的(i,j)表示第i个图片第j个掩码区域，见get_instance
                self.indexing.append((i, j))

    def get_instance_length(self):
        return len(self.indexing)

    def get_image_length(self):
        return len(self.images_info)

    def get_gt_ordering(self, imgidx):
        num = len(self.annot_info[imgidx]['regions'])
        gt_order_matrix = np.zeros((num, num), dtype=np.int)
        order_str = self.annot_info[imgidx]['depth_constraint']
        if len(order_str) == 0:
            return gt_order_matrix
        order_str = order_str.split(',')
        for o in order_str:
            idx1, idx2 = o.split('-')
            idx1, idx2 = int(idx1) - 1, int(idx2) - 1
            gt_order_matrix[idx1, idx2] = 1
            gt_order_matrix[idx2, idx1] = -1
        return gt_order_matrix # num x num

    def get_instance(self, idx, with_gt=False):
        # 可能看我自己数据的情况，这里with_gt可以设置为True
        # 这里能保证图片和标签对应应该是因为COCOA标注文件本身就是有顺序的


        

        imgidx, regidx = self.indexing[idx]
        # img
        # 获取图片的所有信息
        img_info = self.images_info[imgidx]
        # 获取图片名
        image_fn = img_info['file_name']

        w, h = img_info['width'], img_info['height']
        # region
        # 获取图片的模态掩码，bbox和类别（类别为常数1）
        reg = self.annot_info[imgidx]['regions'][regidx]
        modal, bbox, category = read_COCOA(reg, h, w)
        # 如果有GT，就返回非模态掩码
        if with_gt:
            # 将多边形segmentation RLE编码后合并然后解码为二位掩码数组
            amodal = maskUtils.decode(maskUtils.merge(
                maskUtils.frPyObjects([reg['segmentation']], h, w)))
        else:
            amodal = None
        # print("get_instance内 测试6")
        return modal, bbox, category, image_fn, amodal

    def get_image_instances(self, idx, with_gt=False, with_anns=False, ignore_stuff=False):
        ann_info = self.annot_info[idx]
        img_info = self.images_info[idx]
        image_fn = img_info['file_name']
        w, h = img_info['width'], img_info['height']
        ret_modal = []
        ret_bboxes = []
        ret_category = []
        ret_amodal = []
        for reg in ann_info['regions']:
            if ignore_stuff and reg['isStuff']:
                continue
            # if reg['name'] != 'excavator':
            modal, bbox, category = read_COCOA(reg, h, w)
            ret_modal.append(modal)
            ret_bboxes.append(bbox)
            ret_category.append(category)
            if with_gt:
                amodal = maskUtils.decode(maskUtils.merge(
                    maskUtils.frPyObjects([reg['segmentation']], h, w)))
                ret_amodal.append(amodal)
        if with_anns:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn, ann_info
        else:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn


class KINSLVISDataset(object):

    def __init__(self, dataset, annot_fn):
        self.dataset = dataset
        data = cvb.load(annot_fn)
        self.images_info = data['images']
        self.annot_info = data['annotations']
        self.category_info = data['categories']

        # make dict
        self.imgfn_dict = dict([(a['id'], a['file_name']) for a in self.images_info])
        self.size_dict = dict([(a['id'], (a['width'], a['height'])) for a in self.images_info])
        self.anns_dict = self.make_dict()
        self.img_ids = list(self.anns_dict.keys())

    def get_instance_length(self):
        return len(self.annot_info)

    def get_image_length(self):
        return len(self.img_ids)

    def get_instance(self, idx, with_gt=False):
        ann = self.annot_info[idx]
        # img
        imgid = ann['image_id']
        w, h = self.size_dict[imgid]
        image_fn = self.imgfn_dict[imgid]
        # instance
        if self.dataset == 'KINS':
            modal, bbox, category, _ = read_KINS(ann)
        elif self.dataset == 'LVIS':
            modal, bbox, category = read_LVIS(ann, h, w)
        else:   
            raise Exception("No such dataset: {}".format(self.dataset))
        if with_gt:
            amodal = maskUtils.decode(
                maskUtils.frPyObjects(ann['segmentation'], h, w)).squeeze()
        else:
            amodal = None
        return modal, bbox, category, image_fn, amodal

    def make_dict(self):
        anns_dict = {}
        for ann in self.annot_info:
            image_id = ann['image_id']
            if not image_id in anns_dict:
                anns_dict[image_id] = [ann]
            else:
                anns_dict[image_id].append(ann)
        return anns_dict # imgid --> anns

    def get_image_instances(self, idx, with_gt=False, with_anns=False):
        imgid = self.img_ids[idx]
        image_fn = self.imgfn_dict[imgid]
        w, h = self.size_dict[imgid]
        anns = self.anns_dict[imgid]
        ret_modal = []
        ret_bboxes = []
        ret_category = []
        ret_amodal = []
        #ret_score = []
        for ann in anns:
            if self.dataset == 'KINS':
                modal, bbox, category, score = read_KINS(ann)
            elif self.dataset == 'LVIS':
                modal, bbox, category = read_LVIS(ann, h, w)
            else:
                raise Exception("No such dataset: {}".format(self.dataset))
            ret_modal.append(modal)
            ret_bboxes.append(bbox)
            ret_category.append(category)
            #ret_score.append(score)
            if with_gt:
                amodal = maskUtils.decode(
                    maskUtils.frPyObjects(ann['segmentation'], h, w)).squeeze()
                ret_amodal.append(amodal)
        if with_anns:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn, anns
        else:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn

def mask_to_polygon(mask, tolerance=1.0, area_threshold=1):
    """Convert object's mask to polygon [[x1,y1, x2,y2 ...], [...]]
    Args:
        mask: object's mask presented as 2D array of 0 and 1
        tolerance: maximum distance from original points of polygon to approximated
        area_threshold: if area of a polygon is less than this value, remove this small object
    """
    from skimage import measure
    polygons = []
    # pad mask with 0 around borders
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_mask, 0.5)
    # Fix coordinates after padding
    contours = np.subtract(contours, 1)
    for contour in contours:
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) > 2:
            contour = np.flip(contour, axis=1)
            reshaped_contour = []
            for xy in contour:
                reshaped_contour.append(xy[0])
                reshaped_contour.append(xy[1])
            reshaped_contour = [point if point > 0 else 0 for point in reshaped_contour]

            # Check if area of a polygon is enough
            rle = maskUtils.frPyObjects([reshaped_contour], mask.shape[0], mask.shape[1])
            area = maskUtils.area(rle)
            if sum(area) > area_threshold:
                polygons.append(reshaped_contour)
    return polygons
