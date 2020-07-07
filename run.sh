echo "runing run.sh"

## 数据太多需要先删除之前生成的数据
#rm -f /myspace/images/*
#rm -f /myspace/video_images/*
#rm -rf /myspace/work_dirs/*
#ls  /myspace/mask/
#rm /myspace/mask/*.npy
#mkdir /myspace/mask/
## 生成训练数据
#python gen_datas/gen_ans_data_train_new.py


## 生成验证数据
#python gen_datas/gen_ans_data_valid.py
#python gen_datas/gen_img_video.py  ## 把训练数据拆分成 gallery 和 query，用于测试



## 生成 test_query 和 test_gallery
#python gen_datas/gen_ans_test_no_pad.py


## 从上一步中筛选 mini 数据进行测试
#python gen_datas/gen_min_test.py


## 目标检测训练
#python tools/train.py --config configs/cascade_rcnn_r101_fpn_1x.py --gpus=1
#python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --config configs/cascade_rcnn_r101_fpn_1x.py --gpus=4 --autoscale-lr --launcher pytorch
#python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --config configs/cascade_rcnn_r101_fpn_1x_2.py --gpus=4 --autoscale-lr --launcher pytorch

#./tools/dist_train.sh configs/cascade_rcnn_r101_fpn_1x.py 4
#./tools/dist_train.sh configs/cascade_rcnn_r101_fpn_1x_2.py 4

# 重新训练的模型还没预测
## 目标检测 test, 分开测试 gallery 和 query
#python tools/test.py configs/cascade_rcnn_r101_fpn_2x.py /myspace/work_dirs/cascade_rcnn_r101_fpn_1x/latest.pth --json_out /myspace/test_gallery_pred
#python -m torch.distributed.launch --nproc_per_node=4 tools/test.py configs/cascade_rcnn_r101_fpn_2x.py /myspace/work_dirs/cascade_rcnn_r101_fpn_1x/latest.pth --json_out /myspace/test_gallery_pred --launcher pytorch

#python tools/test.py configs/cascade_rcnn_r101_fpn_3x.py /myspace/work_dirs/cascade_rcnn_r101_fpn_1x/latest.pth --json_out /myspace/test_query_pred
#python -m torch.distributed.launch --nproc_per_node=4 tools/test.py configs/cascade_rcnn_r101_fpn_3x.py /myspace/work_dirs/cascade_rcnn_r101_fpn_1x/latest.pth --json_out /myspace/test_query_pred --launcher pytorch


## 把 mmdetection 生成的 box 和 上面的 test_query、test_gallery 合并
#python gen_datas/merge_test_ans.py


## 为 train_ir.py 生成 1000 个类别的数据进行训练
#python gen_datas/gen_mini_train.py

#ls /myspace/work_dirs/resnet50_ibn_a/

## 训练 ir 模型
#python tools/train_ir.py --config configs_ir/resnet_x101.py
#cp /myspace/work_dirs/resnet50_ibn_a/epoch_60.pth /myspace/work_dirs/resnet50_ibn_a/epoch_bn_iter60.pth
#python -m torch.distributed.launch --nproc_per_node=4 tools/train_ir.py --config configs_ir/resnet50_ibn_a_1x.py --gpus=4  --launcher pytorch
#python -m torch.distributed.launch --nproc_per_node=4 tools/train_ir.py --config configs_ir/resnet101_ibn_a_1x.py --gpus=4  --launcher pytorch
#python -m torch.distributed.launch --nproc_per_node=4 tools/train_ir.py --config configs_ir/resnet101_ibn_b_1x.py --gpus=4  --launcher pytorch



#ls /myspace/work_dirs/resnet_x101

## 使用 ir 预测结果
python -m torch.distributed.launch --nproc_per_node=4 tools/test_ir.py configs_ir/resnet50_ibn_a_1x.py /myspace/work_dirs/resnet50_ibn_a_ws/epoch_50.pth --json_out /myspace/test_res --launcher pytorch


#python tools/test_ir.py  configs_ir/resnet50_ibn_a_1x.py /myspace/work_dirs/resnet50_ibn_a/epoch_20.pth --json_out /myspace/test_res

## 后处理
#cp /myspace/result.json /mmdetection/result.json
python gen_datas/get_sub.py


ls /myspace/work_dirs/cascade_rcnn_r101_fpn_1x

ls /myspace/work_dirs/resnet50_ibn_a_ws/

ls /myspace/work_dirs/resnet50_ibn_a/



