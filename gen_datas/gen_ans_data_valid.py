# coding:utf-8

import json
import glob
import os
import cv2
import collections
import numpy as np
import albumentations
from multiprocessing import Pool
# 导入路径
data_rpath = '/tcdata_train/'
data_paths = glob.glob(data_rpath + '*')

print(data_paths)

img_paths = []  # 所有图片的路径
for p in data_paths:
    if os.path.isdir(p) and 'validation' in p:
        img_paths.extend(glob.glob(p + '/image/*/*.jpg'))

# 设置保存路径
myspace = "/myspace/"
img_spath = '/myspace/images/'
anns_spath = '/myspace/annotations/'
video_img_spath = '/myspace/video_images/'
if not os.path.exists(img_spath):
    os.makedirs(img_spath)
if not os.path.exists(anns_spath):
    os.makedirs(anns_spath)
if not os.path.exists(video_img_spath):
    os.makedirs(video_img_spath)

# 23类类别信息（其中'古风'与'古装'同为第2类）
CLASS_DICT = collections.OrderedDict({
    '短外套': 1,
    '古风': 2, '古装': 2,
    '短裤': 3,
    '短袖上衣': 4,
    '长半身裙': 5,
    '背带裤': 6,
    '长袖上衣': 7,
    '长袖连衣裙': 8,
    '短马甲': 9,
    '短裙': 10,
    '背心上衣': 11,
    '短袖连衣裙': 12,
    '长袖衬衫': 13,
    '中等半身裙': 14,
    '无袖上衣': 15,
    '长外套': 16,
    '无袖连衣裙': 17,
    '连体衣': 18,
    '长马甲': 19,
    '长裤': 20,
    '吊带上衣': 21,
    '中裤': 22,
    '短袖衬衫': 23})

# =======1. 对图像库数据集进行标注文件的准备=======
# {
# "images":[
#       {"file_name": i_img_name_item_id,
#       "id": 1,
#       "height": h,
#       "width": w}, ...
#       ]
#  "annotations":[
#       {"image_id":1, "bbox":[xmin,ymin,w,h], "category_id":1}, ...
#       ]
#  "categories":[
#       {"id":1, "name":'短外套'}, ...
#       ]
# }
# ============================================
print("starting annotating:")
images = []
annotations = []
categories = []
img_id = 0

height, width = 960, 540

# 更新categories
for k in list(CLASS_DICT.keys()):
    if '古装' == k:
        continue
    categories.append({"id": CLASS_DICT[k], "name": k})

def get_img_ann(ip):
    img = cv2.imread(ip)
    h, w, _ = img.shape

    r = min(height / h, width / w)
    h_n, w_n = min(int(r * h), height), min(int(r * w), width)

    aug = albumentations.Compose([
        albumentations.Resize(h_n, w_n, p=1),
        albumentations.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['category_id']}, p=1
    )
    # 获取图像路径ip对应的标注路径ap
    ap = ip.replace('image', 'image_annotation')
    ap = ap.replace('jpg', 'json')
    res = {'images':[],
           'anns':[]}
    with open(ap, 'r') as json_f:
        img_ann = json.load(json_f)

    # 若标注为空
    if len(img_ann['annotations']) == 0:
        del img
        pass
    # 若存在标注
    else:
        # 更新images
        file_name = 'images/i_' + str(img_ann['img_name'][:-4]) + '_' + str(img_ann['item_id'] + '.jpg')

        # # 保存图片至images文件夹
        # cv2.imwrite(img_spath + file_name, img)
        # del img

        img_id = 'i_' + str(img_ann['img_name'][:-4]) + '_' + str(img_ann['item_id'])
        res['images'].append({'file_name': file_name,
                       'id': img_id,
                       'height': height,
                       'width': width})
        # 更新annotations
        boxs = []
        cats = []
        in_ids = []
        for ann in img_ann['annotations']:
            box = ann['box']
            cls_id = CLASS_DICT[ann['label']]

            x1, y1 = max(0, box[0]), max(0, box[1])
            x1, y1 = min(w - 1, x1), min(h - 1, y1)
            x2, y2 = min(w - 1, box[2]), min(h - 1, box[3])
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            boxs.append([x1, y1, x2, y2])
            cats.append(cls_id)
            in_ids.append(ann['instance_id'])
        data = {'image': img, 'bboxes': boxs, "category_id": cats}
        data = aug(**data)
        img = data['image']
        boxs = data['bboxes']
        cats = data["category_id"]
        cv2.imwrite(myspace + file_name, img)
        del img

        for c, box, in_id in zip(cats, boxs, in_ids):
            # box = [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            res['anns'].append({'image_id': img_id,
                                'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                                'category_id': c,
                                'instance_id': in_id})
        return res


pool = Pool(32)
anns = pool.map(get_img_ann,img_paths)
pool.close()
pool.join()

for ani in anns:
    if len(ani['anns']) > 0:
        annotations.extend(ani['anns'])
    if len(ani['images']) > 0:
        images.extend(ani['images'])

print("images",len(images),images[-10:])
print("annotations",len(annotations),annotations[-10:])
print('Finish preparing item images!')
print("Frame image starts id from ", img_id)
# exit(0)
del img_paths
# =======2. 对视频库直播切片进行标注文件的准备=======
# 在图像库标注基础上追加内容即可
# {
# "images":[
#       {"file_name": v_video_id_frame_index,
#       "id": 1,
#       "height": h,
#       "width": w}, ...
#       ]
#  "annotations":[
#       {"image_id":1, "bbox":[xmin,ymin,w,h], "category_id":1}, ...
#       ]
#  "categories":[
#       {"id":1, "name":'短外套'}, ...
#       ]
# }
# ============================================
video_paths = []  # 所有视频的路径
# video_ann_paths = [] # 所有视频标注的路径
for p in data_paths:
    if os.path.isdir(p) and 'validation' in p:
        video_paths.extend(glob.glob(p + '/video/*.mp4'))

print("starting deal video")


def get_IoU(pred_bboxes, gt_bboxs):
    """
    return iou score between pred / gt bboxes
    :param pred_bbox: predict bbox coordinate
    :param gt_bbox: ground truth bbox coordinate
    :return: iou score
    """
    overlaps = []
    for gt_bbox in gt_bboxs:
        ixmin = np.maximum(pred_bboxes[:, 0], gt_bbox[0])
        iymin = np.maximum(pred_bboxes[:, 1], gt_bbox[1])
        ixmax = np.minimum(pred_bboxes[:, 2], gt_bbox[2])
        iymax = np.minimum(pred_bboxes[:, 3], gt_bbox[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

        # -----1----- intersection
        inters = iw * ih

        # -----2----- union, uni = S1 + S2 - inters
        uni = ((gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.) +
               (pred_bboxes[:, 2] - pred_bboxes[:, 0] + 1.) * (pred_bboxes[:, 3] - pred_bboxes[:, 1] + 1.) -
               inters)

        # -----3----- iou, get max score and max iou index
        overlap = inters / uni
        overlaps.append(overlap.max())
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
    return overlaps


def get_frame_img(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    _, frame_img = cap.read()
    cap.release()
    return frame_img


def get_keeplist(anns):
    gt_boxs, gt_in_ids = [], []
    ans = []
    for frame1 in anns:
        ans.append(frame1)
        bbox = frame1["annotations"]

        gt_box, gt_id = [], []
        for ann in bbox:
            x1 = int(ann['box'][0])
            y1 = int(ann['box'][1])
            x2 = int(ann['box'][2])
            y2 = int(ann['box'][3])

            gt_box.append([x1, y1, x2, y2])
            gt_id.append(int(ann['instance_id']))
        if len(gt_id) > 0:
            gt_in_ids.append(max(gt_id))
        else:
            gt_in_ids.append(0)
        gt_boxs.append(gt_box)  # 每一帧包含的 box

    rm_list = []
    keep_list = []
    n = len(gt_boxs)
    for i in range(n):
        if len(rm_list) > 0 and i in rm_list:
            continue
        boxi = np.array(gt_boxs[i])
        if boxi.shape[0] == 0:
            continue

        for j in range(i + 1, n):
            boxj = np.array(gt_boxs[j])

            if boxj.shape[0] == 0:
                keep_list.append(i)
                rm_list.append(j)
                continue
            if boxi.shape[0] == boxj.shape[0]:
                iou = get_IoU(boxi, boxj)
                if min(iou) >= 0.85:
                    rm_list.append(j)
                else:
                    if gt_in_ids[i] > 0:
                        keep_list.append(i)
                    break
    return ans, keep_list


def get_video_ann(vp):
    # 获取视频路径p对应的标注路径vap
    vap = vp.replace('video', 'video_annotation')
    vap = vap.replace('mp4', 'json')

    with open(vap, 'r',encoding='utf-8') as json_f2:
        video_ann = json.load(json_f2)
    anns = video_ann["frames"]
    ans, keep_list = get_keeplist(anns)
    keep_list = keep_list[:3]

    cap = cv2.VideoCapture(vp)
    timeF, c = 40, 0
    success = True
    res = {'images':[],
           'anns':[]}
    while (cap.isOpened()):
        success, frame = cap.read()
        # print(frame.shape)
        # exit()
        if success:
            if (c % timeF == 0):
                if c // timeF not in keep_list:
                    continue
                ani = anns[c // timeF]["annotations"]
                vh, vw = frame.shape[:2]
                # 更新images
                img_id = 'v_' + str(video_ann['video_id']) + '_' + str(anns[c // timeF]['frame_index'])

                vfile_name = 'video_images/v_' + str(video_ann['video_id']) + '_' + str(
                    anns[c // timeF]['frame_index']) + '.jpg'
                cv2.imwrite(myspace + vfile_name, frame)

                res['images'].append({'file_name': vfile_name,
                               'id': img_id,
                               'height': vh,
                               'width': vw})
                # 更新annotations
                for fann in ani:
                    box = fann['box']
                    x1, y1 = max(0, box[0]), max(0, box[1])
                    x1, y1 = min(width - 1, x1), min(height - 1, y1)
                    x2, y2 = min(width - 1, box[2]), min(height - 1, box[3])

                    fxmin = float(x1)
                    fymin = float(y1)
                    fbox_w = float(x2 - x1 + 1)
                    fbox_h = float(y2 - y1 + 1)
                    try:
                        #print(CLASS_DICT)
                        fcls_id = CLASS_DICT[fann['label']]
                        res['anns'].append({'image_id': img_id,
                                            'bbox': [fxmin, fymin, fbox_w, fbox_h],
                                            'category_id': fcls_id,
                                            'instance_id': fann['instance_id']})
                    except:
                        print(fann['label'])
                        print(vp)
            c += 1
        else:
            break
    cap.release()
    return res


pool = Pool(32)
anns = pool.map(get_video_ann,video_paths)
pool.close()
pool.join()

for ani in anns:
    if len(ani['anns']) > 0:
        annotations.extend(ani['anns'])
    if len(ani['images']) > 0:
        images.extend(ani['images'])

print("images",len(images),images[-10:])
print("annotations",len(annotations),annotations[-10:])
print('Finish preparing frame images!')

# ‘古装’和‘古风’合为‘古风’
new_categories = [categories[i] for i, cat in enumerate(categories) if cat['name'] != '古装']

# 保存标注至annotations文件夹
print("images",len(images),"annotations",len(annotations))
all_anns = {"images": images, "annotations": annotations, "categories": new_categories}
with open(anns_spath + 'val.json', 'w') as json_f3:
    json.dump(all_anns, json_f3)

print('Finish saving val.json')