# coding:utf-8

import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS
from collections import defaultdict
import json,time,copy,os

class CoCo(COCO):
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        instance_ids = []
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for _, ann in enumerate(self.dataset['annotations']):
                #ann['image_id'] = ann['id']
                ann['id'] = _#ann['image_id']
                ann['area'] = ann['bbox'][2] * ann['bbox'][3]
                imgToAnns[ann['image_id']].append(ann)
                inid = ann.get('instance_id', 0)
                instance_ids.append(inid)
                anns[ann['id']] = ann     # 这里存在错误
                if _ == 0:
                    print("ann:", ann)  #
                ##{'area': 81909.20209190401, 'iscrowd': 0, 'image_id': 11003, 'bbox': [636.08832, 305.77176, 298.5888, 274.32108000000005], 'category_id': 2}

        if 'images' in self.dataset:
            for _, img in enumerate(self.dataset['images']):
                img['image_id'] = str(img['id'])
                img['id'] = _#img['image_id']
                #img.pop('id')
                imgs[img['image_id']] = img
                if _ == 0:
                    print("img:", img)
                # {'file_name': 'images_withoutrect/11003.png', 'height': 1080, 'width': 1920, 'image_id': 11003}

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.instance_ids = list(set(instance_ids))
        self.nids = len(self.instance_ids)
        del instance_ids
        self.anns = anns  # 每个 ann 进行标记
        self.imgToAnns = imgToAnns  # 每个图像 id 包含的 ann
        self.catToImgs = catToImgs  # 每个类别包含的 图像 id
        self.imgs = imgs  # 对图像进行 id 标记
        self.cats = cats  # 对类别进行标记

    def getinstance_ids(self):
        return self.instance_ids

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = CoCo()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile) == unicode):
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
            'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id + 1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1 - x0) * (y1 - y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res


@DATASETS.register_module
class CocoDataset(CustomDataset):

    CLASSES = ('短外套','古风','短裤','短袖上衣','长半身裙','背带裤','长袖上衣','长袖连衣裙',
               '短马甲','短裙','背心上衣','短袖连衣裙','长袖衬衫','中等半身裙','无袖上衣','长外套',
               '无袖连衣裙','连体衣','长马甲','长裤','吊带上衣','中裤','短袖衬衫')


    def getnum_class(self):
        return len(self.inids)

    def load_annotations(self, ann_file):
        self.coco = CoCo(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.inids = self.coco.getinstance_ids()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        tid = 1
        self.ids2label = {}
        for cat_id in self.inids:
            if cat_id == 0:
                self.ids2label[0] = 0
            else:
                self.ids2label[cat_id] = tid
                tid += 1

        self.img_ids = self.coco.getImgIds() # ids = self.imgs.keys()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0] # [self.imgs[ids]]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_video_name(self):
        res = []
        for info in self.img_infos:
            name = os.path.basename(info['filename'])[:-4]
            name = name.split('_')[1]
            res.append(name)
        return res


    def get_instance_ids(self):
        res = []
        for info in self.img_infos:
            img_id = info['image_id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            res.append(ann_info[0]['instance_id'])
        return res

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['image_id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id]) # 选出 img_id 所有的 anns 并：[ann['id'] for ann in anns]
        ann_info = self.coco.loadAnns(ann_ids) #  [self.anns[id] for id in ids]
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_inids = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label.get(ann['category_id'],0))
                inid = ann.get('instance_id', 0)
                gt_inids.append(self.ids2label.get(inid,0))
                #gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_inids = np.array(gt_inids, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_inids = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            inids=gt_inids,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
