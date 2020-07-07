import cv2
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

plt.rcParams['font.sans-serif']=['SimHei']



root_path = r"./datas/train_dataset_part1/video"#视频存放路径
imageDirPath = r"./datas/train_dataset_part1/images"# 切片存放路径
indexes = [f for f in os.listdir(os.path.join(root_path))]

height,width = 960, 540


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



for i in indexes[3:]:#[180:190]: # 16
    videoPath = os.path.join(root_path, i)

    name= os.path.basename(videoPath)[:-4]# str(videoPath).split('/')[-1].split('.')[0]
    vc = cv2.VideoCapture(videoPath)
    cap = vc

    with open(r"./datas/train_dataset_part1/video_annotation/{}.json".format(name), 'r') as load_f:
        f = json.load(load_f)

    anns = f["frames"]
    index = 0
    timeF = 40  # 采样间隔，每隔timeF帧提取一张图片
    c = 0
    s = 0
    success = True
    boxs = {}
    #print('anns',len(anns))
    gt_boxs,gt_in_ids = [], []
    for frame1 in anns:
        frame_index = frame1["frame_index"]
        bbox = frame1["annotations"]

        b = []
        gt_box,gt_id = [],[]
        for ann in bbox:
            x1 = int(ann['box'][0])
            y1 = int(ann['box'][1])
            x2 = int(ann['box'][2])
            y2 = int(ann['box'][3])
            gt_box.append([x1, y1, x2, y2])
            b.append([(x1, y1), (x2, y2),str(ann['instance_id']),ann['label']])
            gt_id.append(int(ann['instance_id']))
        boxs[frame_index] = b
        if len(gt_id) > 0:
            gt_in_ids.append(max(gt_id))
        else:
            gt_in_ids.append(0)
        gt_boxs.append(gt_box)

    rm_list = []
    keep_list = []
    n = len(boxs.keys())
    for i in range(n):
        if len(rm_list) > 0 and i in rm_list:
            continue
        boxi = np.array(gt_boxs[i])
        if boxi.shape[0] == 0:
            rm_list.append(i)
            break

        for j in range(i+1,n):
            boxj = np.array(gt_boxs[j])

            if boxj.shape[0] == 0:
                rm_list.append(j)
                continue
            if boxi.shape[0] == boxj.shape[0]:
                iou = get_IoU(boxi,boxj)
                if min(iou) >= 0.85:
                    rm_list.append(j)
                else:
                    if gt_in_ids[i] > 0:
                        keep_list.append(i)
                    break
    print(name,rm_list)
    #exit()
    fontpath = "./simsun.ttc"  # <== 这里是宋体路径
    font = ImageFont.truetype(fontpath, 32)
    keep_list = keep_list[:3]
    while (cap.isOpened()):
        success, frame = cap.read()
        #print(frame.shape)
        #exit()
        if success:
            if (c % timeF == 0):
                if c // timeF not in keep_list:
                    continue
                #print(c,boxs[c])
                for bi in boxs[c]:
                    cv2.rectangle(frame, bi[0], bi[1], (0, 255, 0), 4)

                    img_pil = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img_pil)
                    b, g, r = 0, 0, 255
                    draw.text(bi[0], bi[3], font=font, fill=(b, g, r))
                    frame = np.array(img_pil)

                    cv2.putText(frame, bi[2], (bi[0][0],bi[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
                cv2.imshow(name, frame)
                cv2.waitKey(500)
            c = c + 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()











