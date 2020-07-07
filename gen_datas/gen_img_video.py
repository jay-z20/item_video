# coding:utf-8

import json


with open('/myspace/annotations/train.json','r') as f:
    t = json.load(f)


images = []
image_video = []

for img in t['images']:
    if 'i_' in img['file_name']:
        images.append(img)
    else:
        image_video.append(img)


a = {'images':images,'categories':t['categories']}

with open('/myspace/annotations/test_gallery.json','w',encoding='utf-8') as f:
    json.dump(a,f)



a = {'images':image_video,'categories':t['categories']}

with open('/myspace/annotations/test_query.json','w',encoding='utf-8') as f:
    json.dump(a,f)


