from __future__ import division
import math

import numpy as np
import torch
from mmcv.runner.utils import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler
import copy, random
from collections import defaultdict


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class DistSameIdentityCateSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

    def __init__(self, data_source, batch_size, num_replicas=None,  # world_size
                 rank=None, num_instances=4):
        if isinstance(data_source, (list, tuple)):
            self.gallery_ind, self.query_ind, self.cate_ind = data_source[0].get_query_gallery_cate()
        else:
            self.gallery_ind, self.query_ind, self.cate_ind = data_source.get_query_gallery_cate()
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        ## 筛选 cate_ind 中，包含数据最多的 cate
        self.max_cate = 0
        tmp = defaultdict(list)
        ind_list = []
        for ci in self.cate_ind:
            for inid in self.cate_ind[ci]:
                if inid not in ind_list:
                    ind_list.append(inid)
                    tmp[ci].append(inid)
        self.cate_ind = tmp

        self.inid = list(self.gallery_ind.keys())
        self.num_identities = 0
        for ind in self.inid:
            self.num_identities += len(self.gallery_ind[ind]) * 4

    def __iter__(self):
        # indices = torch.range(0,self.num_identities)
        batch_idxs_dict = defaultdict(list)
        random.seed(self.epoch)
        np.random.seed(self.epoch)
        for ind in self.inid:
            g_idxs = copy.deepcopy(self.gallery_ind[ind])
            q_idxs = copy.deepcopy(self.query_ind[ind])
            random.shuffle(g_idxs)

            if len(q_idxs) < (self.num_instances - 1) * len(g_idxs):
                q_idxs = np.random.choice(q_idxs, size=(self.num_instances - 1) * len(g_idxs), replace=True)

            random.shuffle(q_idxs)
            batch_idxs = []
            for i, idx in enumerate(g_idxs):
                batch_idxs.append(idx)
                batch_idxs.extend(q_idxs[i * (self.num_instances - 1):(i + 1) * (self.num_instances - 1)])
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[ind].append(batch_idxs)
                    batch_idxs = []

        avai_cat = copy.deepcopy(self.cate_ind)
        final_idxs = []
        cate_key = list(self.cate_ind.keys())
        #print(self.rank, 'cate_key', cate_key)

        while len(cate_key) >= 2:
            random.shuffle(cate_key)
            # print(avai_cat[12], '*+' * 20)
            # print(self.rank,'cate_key', cate_key)
            n_left = self.num_pids_per_batch // 2
            selected_inids = []
            for ci in cate_key:
                nsample = min(len(avai_cat[ci]), n_left)
                selected_inids.extend([(ci, ii) for ii in random.sample(avai_cat[ci],nsample)])
                n_left -= nsample
                if n_left == 0:
                    break
            if n_left > 0:
                break

            # cate = random.sample(cate_key,1)[0]
            # selected_inids = []
            # nsample = min(len(avai_cat[cate]), self.num_pids_per_batch // 2)
            # selected_inids.extend([(cate,ii) for ii in random.sample(avai_cat[cate], nsample)])
            # if nsample < self.num_pids_per_batch//2:
            #    selected_inids.extend([(self.max_cate, ii) for ii in random.sample(avai_cat[self.max_cate], self.num_pids_per_batch // 2 - nsample)])
            for ind in selected_inids:
                ci, ind = ind
                batch_idxs = batch_idxs_dict[ind].pop(0)

                final_idxs.extend(batch_idxs)
                # if ind == 20119601:
                #     print(20119601,len(batch_idxs_dict[ind]),'='*20)
                if len(batch_idxs_dict[ind]) == 0:
                    # print(ind)
                    # if ind == 20119601:
                    #     print(avai_cat[ci],'-'*20)
                    avai_cat[ci].remove(ind)
                    # if ind == 20119601:
                    #     print(avai_cat[ci],ci,'*'*20)
                    if len(avai_cat[ci]) == 0:
                        cate_key.remove(ci)

        self.num_identities = len(final_idxs) - len(final_idxs) % (self.batch_size * self.num_replicas)
        final_idxs = final_idxs[:self.num_identities]
        #print(self.rank, len(final_idxs), final_idxs[:10])
        self.num_identities = len(final_idxs) // self.num_replicas
        offset = self.num_identities * self.rank
        final_idxs = final_idxs[offset:offset + self.num_identities]

        return iter(final_idxs)

    def __len__(self):
        return self.num_identities

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistSameIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

    def __init__(self, data_source, batch_size, num_replicas=None,  # world_size
                 rank=None, num_instances=4):
        if isinstance(data_source, (list, tuple)):
            self.gallery_ind, self.query_ind = data_source[0].get_query_gallery()
        else:
            self.gallery_ind, self.query_ind = data_source.get_query_gallery()
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.inid = list(self.gallery_ind.keys())
        self.num_identities = 0
        for ind in self.inid:
            self.num_identities += len(self.gallery_ind[ind]) * 4

    def __iter__(self):
        # indices = torch.range(0,self.num_identities)
        batch_idxs_dict = defaultdict(list)

        for ind in self.inid:
            g_idxs = copy.deepcopy(self.gallery_ind[ind])
            q_idxs = copy.deepcopy(self.query_ind[ind])
            random.shuffle(g_idxs)

            if len(q_idxs) < (self.num_instances - 1) * len(g_idxs):
                q_idxs = np.random.choice(q_idxs, size=(self.num_instances - 1) * len(g_idxs), replace=True)

            random.shuffle(q_idxs)
            batch_idxs = []
            for i, idx in enumerate(g_idxs):
                batch_idxs.append(idx)
                batch_idxs.extend(q_idxs[i * (self.num_instances - 1):(i + 1) * (self.num_instances - 1)])
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[ind].append(batch_idxs)
                    batch_idxs = []

        avai_inids = copy.deepcopy(self.inid)
        final_idxs = []

        while len(avai_inids) >= self.num_pids_per_batch:
            selected_inids = random.sample(avai_inids, self.num_pids_per_batch)
            for ind in selected_inids:
                batch_idxs = batch_idxs_dict[ind].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[ind]) == 0:
                    avai_inids.remove(ind)
        self.num_identities = len(final_idxs) - len(final_idxs) % (self.batch_size * self.num_replicas)
        final_idxs = final_idxs[:self.num_identities]
        self.num_identities = len(final_idxs) // self.num_replicas

        offset = self.num_identities * self.rank
        final_idxs = final_idxs[offset:offset + self.num_identities]

        return iter(final_idxs)

    def __len__(self):
        return self.num_identities

    def set_epoch(self, epoch):
        self.epoch = epoch


class SameIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

    def __init__(self, data_source, batch_size, num_instances=4):
        if isinstance(data_source, (list, tuple)):
            self.gallery_ind, self.query_ind = data_source[0].get_query_gallery()
        else:
            self.gallery_ind, self.query_ind = data_source.get_query_gallery()
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.inid = list(self.gallery_ind.keys())
        self.num_identities = 0
        for ind in self.inid:
            self.num_identities += len(self.gallery_ind[ind]) * 4

    def __iter__(self):
        # indices = torch.range(0,self.num_identities)
        batch_idxs_dict = defaultdict(list)

        for ind in self.inid:
            g_idxs = copy.deepcopy(self.gallery_ind[ind])
            q_idxs = copy.deepcopy(self.query_ind[ind])
            random.shuffle(g_idxs)

            if len(q_idxs) < (self.num_instances - 1) * len(g_idxs):
                q_idxs = np.random.choice(q_idxs, size=(self.num_instances - 1) * len(g_idxs), replace=True)

            random.shuffle(q_idxs)
            batch_idxs = []
            for i, idx in enumerate(g_idxs):
                batch_idxs.append(idx)
                batch_idxs.extend(q_idxs[i * (self.num_instances - 1):(i + 1) * (self.num_instances - 1)])
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[ind].append(batch_idxs)
                    batch_idxs = []

        avai_inids = copy.deepcopy(self.inid)
        final_idxs = []

        while len(avai_inids) >= self.num_pids_per_batch:
            selected_inids = random.sample(avai_inids, self.num_pids_per_batch)
            for ind in selected_inids:
                batch_idxs = batch_idxs_dict[ind].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[ind]) == 0:
                    avai_inids.remove(ind)

        self.num_identities = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.num_identities


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

    def __init__(self, data_source, num_instances, imgs_per_gpu):
        self.data_source = data_source.get_instance_ids()
        self.imgs_per_gpu = imgs_per_gpu
        print('get_instance_ids:', self.data_source[:4])
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, inid in enumerate(self.data_source):
            self.index_dic[inid].append(index)

        # print('self.index_dic',self.index_dic.get(0,'0'))
        self.inid = list(self.index_dic.keys())
        self.num_identities = len(self.inid)
        nn = 0
        for ki in self.inid:
            if ki != 0:
                nn += self.num_instances
        self.n = nn - nn % self.imgs_per_gpu  # (self.num_identities - 1) * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            inid = self.inid[i]
            if inid == 0:
                continue
            t = self.index_dic[inid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t.astype(np.int64).tolist())
        # self.n = len(ret)
        ret = ret[:self.n]
        # assert len(ret) == self.num_identities * self.num_instances
        return iter(ret)

    def __len__(self):
        return self.n


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate(
                [indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,  # world_size
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
