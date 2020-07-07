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
data_rpath = '/tcdata/'
data_paths = glob.glob(data_rpath + '*')

print(data_paths)

img_paths = []  # 所有图片的路径
for p in data_paths:
    if os.path.isdir(p):
        img_paths.extend(glob.glob(p + '/image/*/*.jpg'))


# 设置保存路径
myspace = "/myspace/"
img_spath = '/myspace/'
anns_spath = '/myspace/annotations/'
video_img_spath = '/myspace/'
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


# ‘古装’和‘古风’合为‘古风’
new_categories = [categories[i] for i, cat in enumerate(categories) if cat['name'] != '古装']


def get_images(ip):
    img = cv2.imread(ip)
    h, w, _ = img.shape

    r = min(height / h, width / w)
    h_n, w_n = min(int(r * h), height), min(int(r * w), width)

    aug = albumentations.Compose([
        albumentations.Resize(h_n, w_n, p=1),
    ], p=1
    )
    # 获取图像路径ip对应的标注路径ap

    img_name = os.path.basename(ip)[:-4]
    item_id = ip.split('/')[-2]

    res = {'images':[]}
    # 若标注为空

    # 更新images
    file_name = 'images/i_' + img_name + '_' + str(item_id + '.jpg')

    # # 保存图片至images文件夹
    # cv2.imwrite(img_spath + file_name, img)
    # del img

    img_id = 'i_' + img_name + '_' + item_id

    res['img_info'] = {img_id:{
        'r': r,
        'h_n': h_n,
        'w_n': w_n,
        'h': h,
        'w': w
    }}


    res['images'].append({'file_name': file_name,
                   'id': img_id,
                   'height': height,
                   'width': width})

    data = {'image': img}
    data = aug(**data)
    img = data['image']

    cv2.imwrite(myspace + file_name, img)
    del img
    return res

pool = Pool(32)
anns = pool.map(get_images,img_paths)
pool.close()
pool.join()

img_info = []
for ani in anns:
    if len(ani['images']) > 0:
        images.extend(ani['images'])

    img_info.append(ani['img_info'])


with open(anns_spath + 'img_info.json', 'w') as json_f3:
    json.dump(img_info, json_f3)

print("images",len(images),images[-10:])
print('Finish preparing item images!')
print("Frame image starts id from ", img_id)

all_anns = {"images": images, "categories": new_categories}
with open(anns_spath + 'test_gallery.json', 'w') as json_f3:
    json.dump(all_anns, json_f3)



video_paths = []  # 所有视频的路径
# video_ann_paths = [] # 所有视频标注的路径
for p in data_paths:
    if os.path.isdir(p):
        video_paths.extend(glob.glob(p + '/video/*.mp4'))

print("starting deal video")

def get_video(vp):
    # 获取视频路径p对应的标注路径vap

    video_id = os.path.basename(vp)[:-4]
    res = {'images':[]}
    cap = cv2.VideoCapture(vp)
    timeF, c = 40, 0
    success = True
    while (cap.isOpened()):
        success, frame = cap.read()
        # print(frame.shape)
        # exit()
        if success:
            if (c % timeF == 0):
                vh, vw = frame.shape[:2]
                # 更新images
                vfile_name = 'video_images/v_' + video_id + '_' + str(c) + '.jpg'
                cv2.imwrite(myspace + vfile_name, frame)

                res['images'].append({'file_name': vfile_name,
                               'id': 'v_'+ video_id + '_' + str(c),
                               'height': vh,
                               'width': vw})
            c += 1
        else:
            break

    cap.release()
    return res

pool = Pool(32)
anns = pool.map(get_video,video_paths)
pool.close()
pool.join()

images = []
for ani in anns:
    if len(ani['images']) > 0:
        images.extend(ani['images'])

print("images",len(images),images[-10:])
print('Finish preparing frame images!')

print('Finish preparing frame images!')


# 保存标注至annotations文件夹
print("images",len(images))
all_anns = {"images": images, "categories": new_categories}
with open(anns_spath + 'test_query.json', 'w') as json_f3:
    json.dump(all_anns, json_f3)

print('Finish saving test.json')





























