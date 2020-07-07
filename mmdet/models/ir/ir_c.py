from __future__ import division

import torch
import torch.nn as nn

from .. import builder
from mmdet.models.backbones import resnet50_ibn_a, resnet101_ibn_a, resnet101_ibn_b
from mmdet.core import auto_fp16
from mmdet.models.losses import CircleLoss,make_loss_with_center
from mmdet.core import bbox2roi
import torch.nn.functional as F
import os
import torch.distributed as dist

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class IRNet(nn.Module):

    def __init__(self, backbone, loss_cfg, roi_extractor, num_class, pretrained=None):
        super(IRNet, self).__init__()
        self.fp16_enabled = False
        #print("backbone", "+" * 20, backbone)
        if backbone.type == 'resnet50_ibn_a':
            self.backbone = resnet50_ibn_a(last_stride=backbone.last_stride)
        elif backbone.type == 'resnet101_ibn_a':
            self.backbone = resnet101_ibn_a(last_stride=backbone.last_stride)
        elif backbone.type == 'resnet101_ibn_b':
            self.backbone = resnet101_ibn_b(last_stride=backbone.last_stride)
        else:
            self.backbone = builder.build_backbone(backbone)
        self.num_classes = num_class
        self.bbox_roi_extractor = builder.build_roi_extractor(roi_extractor)

        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        self._init_fc(self.classifier)

        self.memory_size = 288
        embedding_memory = torch.zeros(self.memory_size, 2048,requires_grad=False)
        label_memory = torch.zeros(self.memory_size,requires_grad=False).long()
        self.register_buffer('queue_idx', torch.tensor([0]))
        self.register_buffer('has_been_filled', torch.tensor([0]))

        self.register_buffer('embedding_memory', embedding_memory)
        self.register_buffer('label_memory', label_memory)

        self.epoch = 0
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.init_weights(pretrained=pretrained)
        self.f = 0
        self.loss_func = CircleLoss()

    def set_epoch(self,epoch):
        #print(self.loss_func)
        self.epoch = epoch

    def add_to_memory(self, embeddings, labels):
        batch_size = labels.size(0)
        with torch.no_grad():
            out_ids = torch.fmod(torch.arange(batch_size, dtype=torch.long).cuda() + self.queue_idx, self.memory_size)
            self.embedding_memory.index_copy_(0, out_ids, embeddings.detach())
            self.label_memory.index_copy_(0, out_ids, labels.detach())
        
        # end_idx = ((self.queue_idx + batch_size - 1) % (self.memory_size)) + 1

        # if end_idx > self.queue_idx:
        #     self.embedding_memory[self.queue_idx:end_idx] = embeddings.detach()
        #     self.label_memory[self.queue_idx:end_idx] = labels.detach()
        # else:
        #     se = self.memory_size - self.queue_idx
        #     self.embedding_memory[self.queue_idx:] = embeddings[:se].detach()
        #     self.embedding_memory[:end_idx] = embeddings[se:].detach()
        #     self.label_memory[self.queue_idx:] = labels[:se].detach()
        #     self.label_memory[:end_idx] = labels[se:].detach()

            prev_queue_idx = self.queue_idx
            self.queue_idx = (self.queue_idx + batch_size) % self.memory_size

            if (not self.has_been_filled) and (self.queue_idx <= prev_queue_idx):
                self.has_been_filled = self.has_been_filled + 1

    def merge_momery(self,embeddings, labels):
        batch_size = embeddings.size(0)
        if self.epoch >= 20:
            index = torch.arange(end=batch_size)
            save_embeddings = embeddings[index % 4 == 0]
            save_labels = labels[index % 4 == 0]
            if self.has_been_filled or self.queue_idx > 0:
                if self.has_been_filled:
                    embeddings = torch.cat([embeddings, self.embedding_memory], dim=0)
                    labels = torch.cat([labels, self.label_memory], dim=0)
                else:
                    embeddings = torch.cat([embeddings, self.embedding_memory[:self.queue_idx]], dim=0)
                    labels = torch.cat([labels, self.label_memory[:self.queue_idx]], dim=0)

            return embeddings, labels, save_embeddings, save_labels
        return embeddings, labels, None, None



    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.constant_(fc.bias, 0.)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=False`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=True`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        # print('img','='*20,img.size())
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.simple_test(img, img_meta, **kwargs)

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        # print(img)
        #print(img.size())  # Nx3x160x96
        x = self.backbone(img)[-1]  # 2x2048x5x3
        #print('x', x.size())
        rois = bbox2roi(gt_bboxes)
        #print('rois:', rois.size())  #
        #print('gt_inids', gt_labels)
        gt_labels = torch.cat(gt_labels)
        # if self.f == 0:
        #     print('gt_inids', gt_labels.size() ,gt_labels)
        #     self.f += 1

        global_feat = self.bbox_roi_extractor([x], rois)  # kx2048x7x7
        #global_feat = self.gap(bbox_feats) + self.gmp(bbox_feats)
        #print('bbox_feats', bbox_feats.size())
        global_feat = global_feat.view(global_feat.size(0), -1)  # kx2048
        #print('bbox_feats view', bbox_feats.size())

        feat = self.bottleneck(global_feat)
        global_feat = normalize(global_feat, axis=-1)

        cls_score = self.classifier(feat)
        loss = {}
        batch_size = gt_labels.size(0)
        embeddings, labels, save_embeddings, save_labels = self.merge_momery(global_feat, gt_labels)

        loss['loss'] = F.cross_entropy(cls_score, gt_labels) \
                       + 0.1 * self.loss_func(embeddings, labels, batch_size)

        if (save_embeddings is not None) and self.epoch >= 20:
            save_embeddings = save_embeddings.contiguous()
            out_list = [torch.zeros_like(save_embeddings, device=save_embeddings.device, dtype=save_embeddings.dtype)
                        for _ in range(dist.get_world_size())]
            dist.all_gather(out_list, save_embeddings)
            save_embeddings = torch.cat(out_list, dim=0)

            save_labels = save_labels.contiguous()
            out_list = [torch.zeros_like(save_labels, device=save_labels.device, dtype=save_labels.dtype)
                        for _ in range(dist.get_world_size())]
            dist.all_gather(out_list, save_labels)
            save_labels = torch.cat(out_list, dim=0)

            self.add_to_memory(save_embeddings, save_labels)
        #print('loss', loss)
        return loss

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_meta, gt_bboxes,gt_labels, proposals=None, rescale=False):
        """Run inference on a single image.

        Args:
            img (Tensor): must be in shape (N, C, H, W)
            img_meta (list[dict]): a list with one dictionary element.
                See `mmdet/datasets/pipelines/formatting.py:Collect` for
                details of meta dicts.
            proposals : if specified overrides rpn proposals
            rescale (bool): if True returns boxes in original image space

        Returns:
            dict: results
        """

        name = [os.path.basename(meta['filename'])[:-4] for meta in img_meta]
        #print('img_meta',img_meta)
        #print('gt_bboxes',gt_bboxes.size(),gt_bboxes)
        name = name * gt_bboxes.size(1)
        #exit()
        #print(name)
        #print(img.size())
        x = self.backbone(img)[-1]
        #print('x', x.size())
        rois = bbox2roi(gt_bboxes)
        #print('rois:', rois.size())
        #print('gt_inids', gt_labels)
        global_feat = self.bbox_roi_extractor([x], rois)
        #global_feat = self.gap(bbox_feats) + self.gmp(bbox_feats)
        #print('bbox_feats', bbox_feats.size())
        global_feat = global_feat.view(global_feat.size(0), -1)
        #print('bbox_feats view', bbox_feats.size())
        global_feat = normalize(global_feat, axis=-1)
        #feat = self.bottleneck(global_feat)
        return global_feat, name,  gt_bboxes

    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        pass
