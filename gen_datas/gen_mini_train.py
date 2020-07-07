# coding:utf-8


import json


anns_spath = '/myspace/annotations/'

with open(anns_spath + 'train.json','r') as f:
    train = json.load(f)

anns = train['annotations']


#annotations = []
instance_ids = []
img_ids = []
n = 0
for anni in anns:
    if anni['instance_id'] not in instance_ids:
        if n < 1000:
            instance_ids.append(anni['instance_id'])
            #annotations.append(anni)
            img_ids.append(anni['image_id'])
            n += 1
    else:
        #annotations.append(anni)
        img_ids.append(anni['image_id'])

instance_ids.append(0)

## 把一张图片中包含两个 instance 的，图片其中一个没包含在 annotation 中的，修改 instance id
img_ids = list(set(img_ids))

new_ans = []
for anni in anns:
    if anni['image_id'] in img_ids:
        if anni['instance_id'] not in instance_ids:
            anni['instance_id'] = 0
        new_ans.append(anni)

images = train['images']
new_images = []
for im in images:
    if im['id'] in img_ids:
        new_images.append(im)

# 保存标注至annotations文件夹
new_categories = train['categories']
print("images",len(new_images),"annotations",len(new_ans))
all_anns = {"images": images, "annotations": new_ans, "categories": new_categories}
with open(anns_spath + 'train_1000.json', 'w',encoding='utf-8') as json_f3:
    json.dump(all_anns, json_f3)

print('Finish saving train_1000.json')


