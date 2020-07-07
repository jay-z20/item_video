# coding:utf-8

import json


anns_spath = '/myspace/annotations/'

## 从 test_gallery 筛选 100 个 images
with open(anns_spath + 'test_gallery.json', 'r') as json_f3:
    test_gallery = json.load(json_f3)

images = test_gallery['images']

images = images[:100]
test_gallery['images'] = images
with open(anns_spath + 'test_gallery_mini.json', 'w') as json_f3:
    json.dump(test_gallery, json_f3)


## 从 img_info 筛选 100 个 images 的图片信息
with open(anns_spath + 'img_info.json', 'r') as json_f3:
    img_info = json.load(json_f3)

images_ids = [im['id'] for im in images]
mini_img_info = [info for info in img_info if list(info.keys())[0] in images_ids]
with open(anns_spath + 'img_info_mini.json', 'w') as json_f3:
    json.dump(mini_img_info, json_f3)



## 从 test_query 筛选 100 个 images
with open(anns_spath + 'test_query.json', 'r') as json_f3:
    test_query = json.load(json_f3)

images = test_query['images']
images = images[:100]
test_query['images'] = images

with open(anns_spath + 'test_query_mini.json', 'w') as json_f3:
    json.dump(test_query, json_f3)





