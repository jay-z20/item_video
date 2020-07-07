# coding:utf-8

import json



with open('/myspace/test_gallery_pred.bbox.json','r') as f:
    test_anns = json.load(f)

with open('/myspace/annotations/test_gallery.json','r') as f:
    test_gallery = json.load(f)

anns = []
image_ids = []
for ani in test_anns:
    if 'i_' in ani['image_id']:
        ani['category_id'] = 0
        anns.append(ani)
        image_ids.append(ani['image_id'])

image_ids = list(set(image_ids))


images = test_gallery['images']

new_images = []

for im in images:
    if im['id'] in image_ids:
        new_images.append(im)

test_gallery['annotations'] = anns
test_gallery['images'] = new_images

print('len test_gallery,',len(new_images))

with open('/myspace/annotations/test_gallery_anns.json','w',encoding='utf-8') as f:
    json.dump(test_gallery,f)




with open('/myspace/test_query_pred.bbox.json','r') as f:
    test_anns = json.load(f)

with open('/myspace/annotations/test_query.json','r') as f:
    test_query = json.load(f)


image_ids = []
anns = []
for ani in test_anns:
    if 'v_' in ani['image_id']:
        ani['category_id'] = 0
        anns.append(ani)
        image_ids.append(ani['image_id'])


image_ids = list(set(image_ids))


images = test_query['images']

new_images = []

for im in images:
    if im['id'] in image_ids:
        new_images.append(im)


test_query['annotations'] = anns
test_query['images'] = new_images
print('len test_query,',len(new_images))
with open('/myspace/annotations/test_query_anns.json','w',encoding='utf-8') as f:
    json.dump(test_query,f)



print('test_query_anns finished')

