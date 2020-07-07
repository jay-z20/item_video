# coding:utf-8

import json
import cv2
import os
import numpy as np

anns_spath = '/myspace/annotations/'

with open('/myspace/result50.json', 'r') as f:
    test_box = json.load(f)

height, width = 960, 540

with open(anns_spath + 'img_info.json', 'r') as f:
    img_info = json.load(f)




img_ids = test_box.keys()

for im_id in img_ids:

    item_id = test_box[im_id].get('item_id',None)
    if item_id:
        img_name = test_box[im_id]['result'][0]["img_name"]
        item_box = test_box[im_id]['result'][0]["item_box"]

        img_item_id = 'i_%s_%s'%(img_name, item_id)
        for i_info in img_info:
            if img_item_id in i_info:
                info = i_info[img_item_id]
                break
        #info = img_info[img_id]

        x1, y1, x2, y2 = item_box
        # pad_x = (width - info['w_n']) // 2
        # x1 = x1 - pad_x
        # pad_y = (height - info['h_n']) // 2
        # y1 = y1 - pad_y

        x1, y1, x2, y2 = map(lambda x: x / info['r'], [x1, y1, x2, y2])
        if x2 + x1 > info['w']:
            x2 = info['w'] - x1
        if y2 + y1 > info['h']:
            y2 = info['h'] - y1

        x2 = x1 + x2
        y2 = y1 + y2
        test_box[im_id]['result'][0]["item_box"] = [x1, y1, x2, y2]

        frame_box = test_box[im_id]['result'][0]["frame_box"]
        x1, y1, x2, y2 = frame_box
        if x2 + x1 > width:
            x2 = width - x1
        if y2 + y1 > height:
            y2 = height - y1
        x2 = x1 + x2
        y2 = y1 + y2
        test_box[im_id]['result'][0]["frame_box"] = [x1, y1, x2, y2]


with open('/mmdetection/result.json','w',encoding='utf-8') as f:
    json.dump(test_box,f)

with open('/myspace/result.json', 'r') as f:
    res = json.load(f)

img_ids = res.keys()
for im_id in img_ids:
    item_id = res[im_id].get('item_id', None)
    if item_id:
        if item_id not in test_box:
            res_dct = {}
            res_dct['item_id'] = item_id
            res_dct['frame_index'] = int(res[im_id]['frame_index'])
            tmp_dct = {}
            tmp_dct["img_name"] = res[im_id]['result'][0]['img_name']

            img_name = res[im_id]['result'][0]["img_name"]
            item_box = res[im_id]['result'][0]["item_box"]

            img_item_id = 'i_%s_%s' % (img_name, item_id)
            for i_info in img_info:
                if img_item_id in i_info:
                    info = i_info[img_item_id]
                    break

            x1, y1, x2, y2 = item_box
            x1, y1, x2, y2 = map(lambda x: x / info['r'], [x1, y1, x2, y2])
            if x2 + x1 > info['w']:
                x2 = info['w'] - x1
            if y2 + y1 > info['h']:
                y2 = info['h'] - y1

            x2 = x1 + x2
            y2 = y1 + y2

            tmp_dct["item_box"] = [x1, y1, x2, y2]

            frame_box = res[im_id]['result'][0]["frame_box"]
            x1, y1, x2, y2 = frame_box
            if x2 + x1 > width:
                x2 = width - x1
            if y2 + y1 > height:
                y2 = height - y1
            x2 = x1 + x2
            y2 = y1 + y2

            tmp_dct["frame_box"] = [x1, y1, x2, y2]

            res_dct['result'] = [tmp_dct]
            test_box[im_id] = res_dct

with open('/mmdetection/result.json','w',encoding='utf-8') as f:
    json.dump(test_box,f)







