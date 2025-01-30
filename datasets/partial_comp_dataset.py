import numpy as np
try:
    import mc
except Exception:
    pass
import cv2
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import utils
from . import reader
import matplotlib.pyplot as plt
import kornia
from torch.nn import functional as F

class PartialCompDataset(Dataset):

    def __init__(self, config, phase):
        self.dataset = config['dataset']
        if self.dataset == 'COCOA':
            # 指向对应的注释文件
            self.data_reader = reader.COCOADataset(config['{}_annot_file'.format(phase)])
        else:
            if self.dataset == 'KINSNew':
                self.data_reader = reader.KINSNewDataset(
                    self.dataset, config['{}_annot_file'.format(phase)])
            else:
                self.data_reader = reader.KINSLVISDataset(
                    self.dataset, config['{}_annot_file'.format(phase)])

        self.indexing = self.data_reader.indexing
        self.annot_info = self.data_reader.annot_info
        self.images_info = self.data_reader.images_info

        # 预先过滤符合条件的索引
        self.valid_indices = [
            i for i in range(len(self.indexing))
            if self.annot_info[self.indexing[i][0]]['regions'][self.indexing[i][1]]['name'] in {'plastic_fence', 'steel_fence'}
        ]
        if not self.valid_indices:
            raise ValueError("No valid 'plastic_fence' or 'steel_fence' items found in the dataset.")
        # print(f"Number of valid 'plastic_fence' or 'steel_fence' items: {len(self.valid_indices)}")

        # 预先过滤符合条件的擦除器索引
        self.eraser_indices = [
            i for i in range(len(self.indexing))
            if self.annot_info[self.indexing[i][0]]['regions'][self.indexing[i][1]]['name'] in {'person', 'excavator'}
        ]
        
        if not self.eraser_indices:
            raise ValueError("No valid 'person' or 'excavator' items found for eraser.")
        # print(f"Number of valid 'person' or 'excavator' eraser items: {len(self.eraser_indices)}")

        self.use_rgb = config['load_rgb']
        if self.use_rgb:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(config['data_mean'], config['data_std'])
            ])
        self.eraser_setter = utils.EraserSetterRGB(config['eraser_setter'])
        self.sz = config['input_size']
        self.eraser_front_prob = config['eraser_front_prob']
        self.phase = phase
        self.use_default = config['use_default']
        # 这里的get表明，即使在yaml配置文件中没找到，也能正常运行，逗号后面是没找到的默认值
        self.use_matting = config.get('use_matting', False)
        self.border_width = config.get('border_width', 5)
        self.occluded_only = config.get('occluded_only', False)
        self.boundary_label = config.get('boundary_label', False) 

        self.config = config

        self.memcached = config.get('memcached', False)
        self.initialized = False
        self.memcached_client = config.get('memcached_client', None)
        self.edge_detection = kornia.filters.Sobel()

    def __len__(self):
        # 返回预先过滤的有效索引数量
        return len(self.valid_indices)

    def _init_memcached(self):
        if not self.initialized:
            assert self.memcached_client is not None, "Please specify the path of your memcached_client"
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _load_image(self, fn):
        if self.memcached:
            try:
                img_value = mc.pyvector()
                self.mclient.Get(fn, img_value)
                img_value_str = mc.ConvertBuffer(img_value)
                img = utils.pil_loader(img_value_str)
                return img
            except Exception as e:
                print(f"Read image failed ({fn}): {e}.")
                raise FileNotFoundError(f"File not found: {fn}")
        else:
            try:
                return Image.open(fn).convert('RGB')
            except FileNotFoundError:
                fallback_fn = fn.replace('val', 'train')
                # print(f"File not found: {fn}. Trying fallback path: {fallback_fn}.")
                try:
                    return Image.open(fallback_fn).convert('RGB')
                except FileNotFoundError as e:
                    print(f"Fallback file not found: {fallback_fn}.")
                    raise FileNotFoundError(f"Fallback file not found: {fallback_fn}") from e

    def _get_inst(self, idx, load_rgb=False, randshift=False):
        # 利用数据增强的办法获取标注文件里的模态掩码和RGB图像
        # 这里获取的一定是模态掩码，但是我这边数据原因，这里可能都是未被遮挡的模态掩码

        modal, bbox, category, imgfn, _ = self.data_reader.get_instance(idx)

        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        # size是一个近似放大的边长
        size = max([np.sqrt(bbox[2] * bbox[3] * self.config['enlarge_box']), bbox[2] * 1.1, bbox[3] * 1.1])
        if size < 5 or np.all(modal == 0):
            # 不再递归调用，而是随机选择一个新的索引并重试
            new_idx = np.random.choice(len(self.indexing))
            print(f"Invalid instance at index {idx} with size {size} or empty modal. Selecting new index {new_idx}.")
            return self._get_inst(new_idx, load_rgb=load_rgb, randshift=randshift)

        # shift & scale aug
        if self.phase == 'train':
            # 如果 randshift 为真，则对bbox中心点进行随机平移
            if randshift:
                centerx += np.random.uniform(*self.config['base_aug']['shift']) * size
                centery += np.random.uniform(*self.config['base_aug']['shift']) * size
            # 对裁剪框大小进行随机缩放
            size /= np.random.uniform(*self.config['base_aug']['scale'])

        # crop，计算新的裁剪边界框
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        # resize 到指定大小
        modal = cv2.resize(utils.crop_padding(modal, new_bbox, pad_value=(0,)),
            (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

        # flip，水平翻转
        if self.config['base_aug']['flip'] and np.random.rand() > 0.5:
            flip = True
            modal = modal[:, ::-1]
        else:
            flip = False

        if load_rgb:
            # 获取图像路径
            rgb = np.array(self._load_image(os.path.join(
                self.config['{}_image_root'.format(self.phase)], imgfn)))  # uint8
            rgb = cv2.resize(utils.crop_padding(rgb, new_bbox, pad_value=(0,0,0)),
                (self.sz, self.sz), interpolation=cv2.INTER_CUBIC)
            if flip:
                rgb = rgb[:, ::-1, :]

        if load_rgb:
            return modal, category, rgb
        else:
            return modal, category, None

    def __getitem__(self, idx):
        if self.memcached:
            self._init_memcached()

        # 获取预先过滤的有效索引
        valid_idx = self.valid_indices[idx]
        imgidx, regidx = self.indexing[valid_idx]
        region_name = self.annot_info[imgidx]['regions'][regidx]['name']
        # print(f"Processing index {idx} with region '{region_name}'.")

        # 随机选择一个符合条件的擦除器索引
        randidx = np.random.choice(self.eraser_indices)
        randimgidx, randregidx = self.indexing[randidx]
        rand_region_name = self.annot_info[randimgidx]['regions'][randregidx]['name']
        # print(f"Selected eraser index {randidx} with region '{rand_region_name}'.")

        # 获取当前索引对应的实例
        try:
            modal, category, rgb = self._get_inst(valid_idx, load_rgb=True, randshift=True)
        except FileNotFoundError as e:
            print(f"FileNotFoundError when processing valid index {valid_idx}: {e}. Selecting another valid index.")
            # 选择另一个有效索引来替代
            new_valid_idx = np.random.choice(len(self.valid_indices))
            return self.__getitem__(new_valid_idx)

        if not self.config.get('use_category', True):
            category = 1

        # 获取擦除器的实例
        try:
            eraser, _, eraser_rgb = self._get_inst(randidx, load_rgb=True, randshift=False)
        except FileNotFoundError as e:
            print(f"FileNotFoundError when processing eraser index {randidx}: {e}. Selecting another eraser index.")
            # 选择另一个擦除器索引来替代
            new_eraser_idx = np.random.choice(len(self.eraser_indices))
            eraser, _, eraser_rgb = self._get_inst(self.eraser_indices[new_eraser_idx], load_rgb=True, randshift=False)

        # 根据yaml里面的参数对擦除器的掩码和 RGB 图像做调整
        eraser, eraser_rgb = self.eraser_setter(modal, eraser, eraser_rgb)  # uint8 {0, 1}

        border_width = self.border_width

        # erase
        erased_modal = modal.copy()
        # 随机决定擦除器是否在模态掩码上方，返回布尔值
        # 如果小于设置的阈值（0.8）,则返回1
        # 就是说，有0.8的概率擦除器在模态掩码上方
        eraser_above = np.random.rand() < self.eraser_front_prob

        if eraser_above:
            eraser_mask = eraser
            # 擦除模态掩码中的相应部分
            erased_modal[eraser == 1] = 0  # eraser above modal
            if self.occluded_only or self.boundary_label:
                # 获取被擦除的部分
                occluded = (eraser == 1) & (modal == 1)

            if self.boundary_label:
                occluded_extend = F.max_pool2d(torch.from_numpy(occluded[None, None, ...]).float(), border_width, stride=1, padding=border_width//2)
                complement_extend = F.max_pool2d(torch.from_numpy((1-modal)[None, None, ...]).float(), border_width, stride=1, padding=border_width//2)
                gt_boundary = ((occluded_extend == 1) & (complement_extend == 1))[0, 0].float()

        else:  # 如果擦除器在下方
            # eraser_mask就是擦除器的模态掩码
            eraser_mask = (eraser == 1) & (modal == 0)  # B \ A
            # 令擦除器被遮挡部分的掩码为0
            eraser[modal == 1] = 0  # eraser below modal

            if self.occluded_only:
                occluded = np.zeros_like(erased_modal)

            if self.boundary_label:
                gt_boundary = torch.zeros_like(torch.from_numpy(erased_modal)).float()

        # 下面部分主要是转换为张量
        eraser_mask = eraser_mask.astype('float')  # just used for matting RGB image

        eraser_tensor = torch.from_numpy(eraser.astype(np.float32)).unsqueeze(0)  # 1HW

        # 这里的use_default属性为True
        if self.use_default:
            # 保持边界
            keep_boundary = eraser_tensor  
        else:
            # 如果设置这里的use_default属性为False，则进行边界交集计算
            eraser_extend = F.max_pool2d(torch.from_numpy(eraser[None, None, ...]).float(), border_width, stride=1, padding=border_width//2)
            modal_extend = F.max_pool2d(torch.from_numpy(erased_modal[None, None, ...]).float(), border_width, stride=1, padding=border_width//2)
            keep_boundary = ((eraser_extend == 1) & (modal_extend == 1))[0].float()

            # image matting boundary
            eraser_mask[keep_boundary[0].numpy() == 1] = 0.5

            eraser_mask[eraser_mask == 1] = 0.8  # matting almost for other pixels, change from 0.8

        # 合并保持边界的擦除器张量
        eraser_tensor = torch.cat([keep_boundary, eraser_tensor])  # HW
        # 生成擦除后的模态掩码
        erased_modal = erased_modal.astype(np.float32) * category

        # erase rgb
        # 如果 rgb 存在且使用 RGB
        if rgb is not None and self.use_rgb:
            # 如果使用抠图
            if self.use_matting:
                # 为擦除器掩码添加通道
                eraser_mask = eraser_mask[..., None]
                # 生成抠图后的 RGB
                rgb = rgb * (1 - eraser_mask) + eraser_rgb * eraser_mask
            else:
                # 生成非抠图后的 RGB
                rgb = rgb * (1 - eraser_tensor[1, ..., None].numpy())

            # 转换 RGB
            rgb = self.img_transform(rgb).float()  # CHW
        else:
            # 生成空的 RGB
            rgb = torch.zeros((3, self.sz, self.sz), dtype=torch.float32)  # 3HW

        # 生成模态掩码张量
        erased_modal_tensor = torch.from_numpy(
            erased_modal.astype(np.float32)).unsqueeze(0)  # 1HW

        if self.occluded_only:
            # target是被擦除器遮挡部分的掩码
            target = torch.from_numpy(occluded.astype(int))
        else:
            # target是idx对应物体的模态掩码
            target = torch.from_numpy(modal.astype(int))  # HW

        if self.boundary_label:
            # 
            target = torch.stack([target, gt_boundary.long()])

        # print(f"Processed index {idx} with region '{region_name}' and eraser index {randidx}.")
        return rgb, erased_modal_tensor, eraser_tensor, target
