# coding:utf-8

import json
import glob
import os
import cv2
import collections
import numpy as np
import albumentations
from multiprocessing import Pool
from collections import defaultdict

# import imaplib
#
# imaplib.reload(sys)
myspace = "/myspace/"
img_spath = '/myspace/images_train/'
anns_spath = '/myspace/annotations/'
video_img_spath = '/myspace/video_images_train/'
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

def get_img_ann(ip):
    # 从同一批 img_id 中，筛选 box >=2 的图片作为目标检测的数据
    # 对于同一批 img_id 中，box 都只有 1 个的数据，只选其中一个作为训练数据
    nboxs = []
    for fname in os.listdir(ip):
        ap = os.path.join(ip,fname)
        with open(ap, 'r') as json_f:
            img_ann = json.load(json_f)
        nbox = len(img_ann.get('annotations',[]))
        nboxs.append((fname,nbox))

    ## 按照 box 的数量倒序排序
    res = {'images': [],
           'anns': []}
    if len(nboxs) == 0:
        return res
    nboxs = sorted(nboxs,key=lambda x:x[1],reverse=True)
    if nboxs[0][1] == 0:
        return res
    tmp = []
    for t in nboxs:
        fram_index, nbox = t
        if len(tmp) >2 and nbox < 2:  ## an 中 box 的数量少于 2 个，只保留 3 张图片
            break
        tmp.append(t)

    nboxs = tmp

    for v in nboxs:
        fname, nbox = v
        ap = os.path.join(ip, fname)
        with open(ap, 'r') as json_f:
            img_ann = json.load(json_f)
        imp = ap.replace('image_annotation','image')
        imp = imp.replace('json','jpg')

        img = cv2.imread(imp)
        h, w, _ = img.shape
        height, width = 960, 540
        r = min(height / h, width / w)
        h_n, w_n = min(int(r * h), height), min(int(r * w), width)

        img = cv2.resize(img,(w_n, h_n),interpolation=cv2.INTER_LINEAR)

        file_name = 'images/i_' + str(img_ann['img_name'][:-4]) + '_' + str(img_ann['item_id'] + '.jpg')
        img_id = 'i_' + str(img_ann['img_name'][:-4]) + '_' + str(img_ann['item_id'])
        res['images'].append({'file_name': file_name,
                              'id': img_id,
                              'height': h_n,
                              'width': w_n})

        # 更新annotations
        boxs, v_points = [], []
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
            x1, y1, x2, y2 = map(lambda x: r * x, [x1, y1, x2, y2])
            boxs.append([x1, y1, x2, y2])
            cats.append(cls_id)
            v_points.append(ann['viewpoint'])
            in_ids.append(ann['instance_id'])

        cv2.imwrite(myspace + file_name, img)
        del img

        for c, box, in_id, v_p in zip(cats, boxs, in_ids, v_points):
            # box = [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            res['anns'].append({'image_id': img_id,
                                'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                                'v_point':v_p,
                                'category_id': c,
                                'instance_id': in_id})
    return res


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

def get_keeplist(anns):
    gt_boxs, gt_in_ids = [], []
    ans = []
    for frame1 in anns:
        ans.append(frame1)
        bbox = frame1["annotations"]

        gt_box,gt_id = [],[]
        for ann in bbox:
            x1, y1, x2, y2 = map(int,ann['box'])
            gt_box.append([x1, y1, x2, y2])
            gt_id.append(int(ann['instance_id']))
        if len(gt_id) > 0:
            gt_in_ids.append(max(gt_id))
        else:
            gt_in_ids.append(0)
        gt_boxs.append(gt_box) # 每一帧包含的 box

    rm_list = []
    keep_list = []
    n = len(gt_boxs)
    for i in range(n):
        if len(rm_list) > 0 and i in rm_list:
            continue
        boxi = np.array(gt_boxs[i])
        if boxi.shape[0] == 0:
            continue
        keep_list.append(i)
        for j in range(i+1,n):
            boxj = np.array(gt_boxs[j])
            if boxj.shape[0] == 0:
                keep_list.append(i)
                rm_list.append(j)
                break
            if boxi.shape[0] == boxj.shape[0]:
                iou = get_IoU(boxi,boxj)
                if min(iou) >= 0.8:
                    rm_list.append(j)

    return ans,list(set(keep_list))


def get_video_ann(vap):

    with open(vap, 'r') as json_f:
        video_ann = json.load(json_f)
    res = {'images': [],
           'anns': []}
    if len(video_ann['frames']) == 0:
        return res

    ans, keep_list = get_keeplist(video_ann["frames"])  ## 把连续帧，iou > 0.85 的删除

    nboxs = []
    anns = video_ann["frames"]
    for ani in anns:
        nbox = len(ani['annotations'])
        fram_index = ani['frame_index']
        nboxs.append((fram_index, nbox))

    nboxs = sorted(nboxs, key=lambda x: x[1], reverse=True)
    tmp = []
    for ti,t in enumerate(nboxs):
        fram_index, nbox = t
        if nbox < 2:
            tmp.extend(nboxs[ti:ti+3]) # 取 box 多于 1 个
            break
        tmp.append(t)

    nboxs = tmp

    # 获取视频路径p对应的标注路径vap
    vp = vap.replace('/video_annotation','/video')
    vp = vp.replace('json','mp4')

    height, width = 960, 540
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
                fs = False
                for t in nboxs:
                    fram_index, nbox = t
                    if c == fram_index:
                        fs = True
                        break
                if not fs:
                    c = c + 1
                    continue

                if c // timeF not in keep_list:
                    c = c + 1
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
                                            'v_point': fann['viewpoint'],
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


if __name__=='__main__':
    # 导入路径
    data_rpath = '/tcdata_train/'
    data_paths = glob.glob(data_rpath + '*')

    print(data_paths)

    img_paths = []  # 所有图片的路径
    for p in data_paths:
        if os.path.isdir(p) and 'train_dataset' in p:
            img_paths.extend(glob.glob(p + '/image_annotation/*'))

    # 设置保存路径

    if not os.path.exists(img_spath):
        os.makedirs(img_spath)
    if not os.path.exists(anns_spath):
        os.makedirs(anns_spath)
    if not os.path.exists(video_img_spath):
        os.makedirs(video_img_spath)

    # ============================================
    print("starting annotating:")
    images = []
    annotations = []
    categories = []
    img_id = 0

    # 更新categories
    for k in list(CLASS_DICT.keys()):
        if '古装' == k:
            continue
        categories.append({"id": CLASS_DICT[k], "name": k})

    pool = Pool(32)
    anns = pool.map(get_img_ann,img_paths)
    pool.close()
    pool.join()

    instance_list = []
    keep_res = []  ## 保存符合条件的 res，instance_id 次数不能超过 3 次，instance_id=0 除外
    all_images = []
    all_annotations = []
    ind_dct = defaultdict(lambda:0)
    for ani in anns:
        if len(ani['anns']) > 0:
            for annj in ani['anns']:
                if annj['v_point'] != 0:
                    continue
                ind_dct[annj['instance_id']] += 1
            all_annotations.extend(ani['anns'])
            all_images.extend(ani['images'])
    # 如果 instance_id 没有在 3 个图片中出现，则删除
    tmp_dct = {}
    for key in ind_dct:
        if key == 0:
            continue
        if ind_dct[key] >= 3:
            tmp_dct[key] = ind_dct[key]
    ind_dct = tmp_dct
    img_anns = anns
    # exit(0)
    del img_paths

    # ============================================
    video_paths = []  # 所有视频的路径
    # video_ann_paths = [] # 所有视频标注的路径
    for p in data_paths:
        if os.path.isdir(p) and 'train_dataset' in p:
            video_paths.extend(glob.glob(p + '/video_annotation/*'))

    print("starting deal video")

    pool = Pool(32)
    anns = pool.map(get_video_ann,video_paths)
    pool.close()
    pool.join()

    ## 需要考虑在 images 中存在，video 中也存在的图片


    ## 需要考虑在 images 中存在，video 中也存在的图片

    v_instance_list = []
    means, stds = [], []
    com_keys = []
    com_dct = defaultdict(lambda:0)
    ## 筛选 instance_id 在 query 中出现过，且次数多于 2 张图片的 id
    for ani in anns:
        if len(ani['anns']) > 0:
            for annj in ani['anns']:
                if annj['v_point'] != 0:
                    continue
                if annj['instance_id'] in ind_dct:
                    com_dct[annj['instance_id']] += 1
            all_annotations.extend(ani['anns'])
            all_images.extend(ani['images'])
    tmp_dct = {}
    for key in com_dct.keys():
        if com_dct[key] >= 2:
            tmp_dct[key] = com_dct[key]
    com_dct = tmp_dct
    # 构建图像检索 json
    ir_images, ir_ann = [], []
    for ani in anns:
        if len(ani['anns']) > 0:
            img_ids = []
            for annj in ani['anns']:
                if annj['instance_id'] in com_dct:
                    ir_ann.append(annj)
                    img_ids.append(annj['image_id'])
            img_ids = set(img_ids)
            for annj in ani['images']:
                if annj['id'] in img_ids:
                    ir_images.append(annj)

    for ani in img_anns:
        if len(ani['anns']) > 0:
            img_ids = []
            for annj in ani['anns']:
                if annj['instance_id'] in com_dct:
                    ir_ann.append(annj)
                    img_ids.append(annj['image_id'])
            img_ids = set(img_ids)
            for annj in ani['images']:
                if annj['id'] in img_ids:
                    ir_images.append(annj)

    print("images", len(all_images), all_images[-10:])
    print("annotations", len(all_annotations), all_annotations[-10:])
    print('Finish preparing frame images!')

    # ‘古装’和‘古风’合为‘古风’
    new_categories = [categories[i] for i, cat in enumerate(categories) if cat['name'] != '古装']

    # 保存标注至annotations文件夹
    print("images", len(all_images), "annotations", len(all_annotations))
    all_anns = {"images": all_images, "annotations": all_annotations, "categories": new_categories}
    with open(anns_spath + 'train_detect.json', 'w') as json_f3:
        json.dump(all_anns, json_f3)

    # 保存标注至annotations文件夹
    print("images", len(ir_images), "annotations", len(ir_ann))
    all_anns = {"images": ir_images, "annotations": ir_ann, "categories": new_categories}
    with open(anns_spath + 'train_ir.json', 'w') as json_f3:
        json.dump(all_anns, json_f3)

    print('Finish saving train.json')