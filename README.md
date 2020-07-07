
# [淘宝直播商品识别大赛](https://tianchi.aliyun.com/competition/entrance/231772/introduction)


- [x] 数据预处理 (gen_datas/gen_ans_data_train_no_pad.py)
    - [x] 只保留视频片段和 gallery 都出现过的 id
    - [x] 对视频片段中相隔40帧的图片按照iou去重

- [x] model and loss (mmdet/models/ir/ir.py)
    - [x] circle loss
    - [x] 保存前面4个 iter 的 gallery 图像作为 circle loss 中的 anchor
    - [x] 弱监督数据增强
    - [x] FP16

- [x] test (tools/test_ir.py)
    - [x] 多卡预测









