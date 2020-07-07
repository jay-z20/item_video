import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

torch.multiprocessing.set_sharing_strategy('file_system')

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.models.ir import IRNet
from collections import defaultdict
import json
import numpy as np



class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results, names, gt_bboxes = [], [], []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, name, gt_bbox = model(return_loss=False, rescale=not show, **data)
        results.append(result)
        names.extend(name)
        gt_bbox = gt_bbox.squeeze(0)
        gt_bboxes.append(gt_bbox)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results, names, gt_bboxes


def multi_gpu_test(model, data_loader, show=False, tmpdir=None, gpu_collect=False):
    model.eval()
    results, names, gt_bboxes = [], [], []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    res = {}
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, name, gt_bbox = model(return_loss=False, rescale=not show, **data)
        results.append(result)
        name = name[0]
        names.append(name)
        #gt_bbox = gt_bbox[0]
        if isinstance(gt_bbox, np.ndarray):
            if len(gt_bbox.shape) == 1:
                gt_bbox = gt_bbox.tolist()
            else:
                gt_bbox = gt_bbox[0].tolist()
        elif isinstance(gt_bbox, list):
            if len(gt_bbox) == 1:
                gt_bbox = gt_bbox[0]
        assert len(gt_bbox) == 4
        assert isinstance(gt_bbox, list)
        #gt_bbox = gt_bbox.squeeze(0)
        gt_bboxes.append(gt_bbox)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    res['results'] = results
    res['names'] = names
    res['gt_bboxes'] = gt_bboxes

    results, names, gt_bboxes = collect_results_cpu(res, len(dataset), tmpdir)
    return results,names,gt_bboxes


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 2048
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))  ###????????????
    dist.barrier()
    # collect all parts
    # if rank != 0:
    #     return None
    # else:
    # load results of all parts from tmp dir
    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
        part_list.append(mmcv.load(part_file))
    # sort the results
    results, names, gt_bboxes = [], [], []
    for res in part_list:
        results.extend(res['results'])
        names.extend(res['names'])
        #print(res['names'][0])
        gt_bboxes.extend(res['gt_bboxes'])
    # the dataloader may pad some samples
    results = results[:size]
    names = names[:size]
    gt_bboxes = gt_bboxes[:size]

    # remove tmp dir
    # if os.path.exists(tmpdir):
    #     shutil.rmtree(tmpdir)
    return results, names, gt_bboxes


def multi_gpu_test_video(model, data_loader, gf, gnames, gboxs):
    model.eval()

    dataset = data_loader.dataset
    n = gf.size(0)
    #print(gboxs.size(),len(gnames),gnames)
    prog_bar = mmcv.ProgressBar(len(dataset))
    res = defaultdict(list)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            qf, name, gt_bbox = model(return_loss=False, **data)
        name = name[0]
        #print('name:', name)
        #gt_bbox = gt_bbox[0]
        if isinstance(gt_bbox, np.ndarray):
            if len(gt_bbox.shape) == 1:
                gt_bbox = gt_bbox.tolist()
            else:
                gt_bbox = gt_bbox[0].tolist()
        elif isinstance(gt_bbox, list):
            if len(gt_bbox) == 1:
                gt_bbox = gt_bbox[0]
        assert len(gt_bbox) == 4
        assert isinstance(gt_bbox, list)

        v_name = name.split('_')[1]         # video name
        v_index = str(name.split('_')[2])  # video index

        distmat = torch.sum(qf * gf,dim=1)
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        #print('distmat',distmat.size())
        distmat = distmat.cpu().numpy()
        index = np.argsort(distmat)  #### import

        index = index[::-1]
        ## 1.统计每个 query box 周围存在相同 gallery id 的次数
        tmp = []
        n = 0
        dct = defaultdict(lambda: 0)
        dct_gallery = defaultdict(list)
        for idx in index:
            if n >= 5:
                break
            gn = gnames[idx]
            if gn in tmp:
                continue
            else:
                tmp.append(gn)
            n += 1
            dct[gn.split('_')[2]] += 1  # 相同 gallery id 不同 file 的次数
            dct_gallery[gn.split('_')[2]].append(
                {
                    'item_box': gboxs[idx],
                    'img_name': gn,
                    'distance': distmat[idx]
                }
            )

        g_n = sorted(dct.items(), key=lambda x: x[1], reverse=True)  ## gallery 出现的次数
        ## 2. 同一个 query id 出现在 query box 的最大次数  >= 3 次
        # print(g_n[0],g_n[0][0])
        if g_n[0][1] >= 3:
            dct = {
                'frame_name': name,
                'frame_index': v_index,
                'frame_box': gt_bbox,
                'match_item': g_n[0][0],
                'item_box': dct_gallery[g_n[0][0]][0]['item_box'],
                'img_name': dct_gallery[g_n[0][0]][0]['img_name'],
                'distance': dct_gallery[g_n[0][0]][0]['distance']
            }
        else:
            dct = {
                'frame_name': name
            }

        res[v_name].append(dct.copy())
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    tmpdir = None
    if tmpdir is None:
        MAX_LEN = 2048
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(res, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        res = part_list[0]
        for pi in part_list[1:]:
            for key in pi:
                if key in res:
                    res[key].extend(pi[key])
                else:
                    res[key] = pi[key]

        result = {}
        for key in res.keys():
            v = res[key]
            tmp, frame_name = [], []
            ## 3.统计最近 gallery id 在 query video 中出现的次数，如果出现次数大于 video frame 的 1/2 则为结果，否则判定为不存在
            match_item_dct = defaultdict(lambda: 0)
            dct_gallery = defaultdict(list)
            for dct in v:
                gname = dct.get('img_name', None)
                if gname:
                    s = str(dct['frame_index']) + '_' + dct['match_item']
                    if s not in tmp:
                        match_item_dct[dct['match_item']] += 1
                        tmp.append(s)

                    dct_gallery[dct['match_item']].append(
                        {
                            'item_box': dct['item_box'],
                            'img_name': dct['img_name'],
                            'distance': dct['distance'],
                            'frame_index': dct['frame_index'],
                            'frame_box': dct['frame_box'],
                        }
                    )
            try:
                gname_list_cnt = sorted(match_item_dct.items(), key=lambda x: x[1], reverse=True)
                if len(gname_list_cnt) < 1:
                    continue
                if gname_list_cnt[0][1] >= 5:
                    ## 能匹配到确定结果
                    res_dct = {}
                    # print('v'*20,v)
                    tmp_list = sorted(dct_gallery[gname_list_cnt[0][0]], key=lambda x: x['distance'], reverse=True)
                    #print('tmp_list', tmp_list)
                    res_dct['item_id'] = gname_list_cnt[0][0]
                    res_dct["frame_index"] = int(tmp_list[0]['frame_index'])
                    tmp_dct = {}
                    tmp_dct["img_name"] = tmp_list[0]['img_name'].split('_')[1]
                    item_box = tmp_list[0]['item_box']
                    if isinstance(item_box,np.ndarray):
                        if len(item_box.shape) == 1:
                            tmp_dct["item_box"] = item_box.tolist()
                        else:
                            tmp_dct["item_box"] = item_box[0].tolist()
                    else:
                        tmp_dct["item_box"] = item_box

                    frame_box = tmp_list[0]['frame_box']
                    if isinstance(frame_box,np.ndarray):
                        if len(frame_box.shape) == 1:
                            tmp_dct["frame_box"] = frame_box.tolist()
                        else:
                            tmp_dct["frame_box"] = frame_box[0].tolist()
                    else:
                        tmp_dct["frame_box"] = frame_box

                    res_dct['result'] = [tmp_dct]
                    result[key] = res_dct
            except:
                pass

        print('len key',len(result.keys()))
        with open('/myspace/result50.json','w',encoding='utf-8') as f:
                json.dump(result,f)




def single_gpu_test_video(model, data_loader, gf, gnames, gboxs):
    model.eval()

    dataset = data_loader.dataset
    gf = gf.cuda()
    n = gf.size(0)
    #print(gboxs.size(),len(gnames),gnames)
    prog_bar = mmcv.ProgressBar(len(dataset))
    res = defaultdict(list)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            qf, name, gt_bbox = model(return_loss=False, **data)
        name = name[0]
        print('name:', name)
        gt_bbox = gt_bbox.squeeze(0).cpu().numpy().tolist()
        v_name = name.split('_')[1]         # video name
        v_index = str(name.split('_')[2])  # video index

        distmat = torch.sum(qf*gf,dim=1)
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        print('distmat',distmat.size())


        distmat = distmat.cpu().numpy()
        index = np.argsort(distmat) #### import

        index = index[::-1]
        ## 1.统计每个 query box 周围存在相同 query id 的次数
        tmp = []
        n = 0
        dct = defaultdict(lambda:0)
        dct_gallery = defaultdict(list)
        for idx in index:
            if n >=5:
                break
            gn = gnames[idx]
            if gn in tmp:
                continue
            else:
                tmp.append(gn)
            n += 1
            dct[gn.split('_')[2]] += 1
            dct_gallery[gn.split('_')[2]].append(
                {
                    'item_box':gboxs[idx],
                    'img_name':gn,
                    'distance':distmat[idx]
                }
            )

        g_n = sorted(dct.items(), key=lambda x: x[1], reverse=True) ## gallery 出现的次数
        ## 2. 同一个 query id 出现在 query box 的最大次数  >= 3 次
        if g_n[0][1] >= 3:
            dct = {
                'frame_name':name,
                'frame_index': v_index,
                'frame_box': gt_bbox,
                'match_item':g_n[0][0],
                'item_box': dct_gallery[0]['item_box'],
                'img_name': dct_gallery[0]['img_name'],
                'distance': dct_gallery[0]['distance']
            }
        else:
            dct = {
                'frame_name': name
            }

        res[v_name].append(dct.copy())

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    result = {}
    for key in res.keys():
        v = res[key]
        tmp, frame_name = [], []
        ## 3.统计最近 gallery id 在 query video 中出现的次数，如果出现次数大于 video frame 的 1/2 则为结果，否则判定为不存在
        match_item_dct = defaultdict(lambda:0)
        dct_gallery = defaultdict(list)
        for dct in v:
            gname = dct.get('img_name',None)
            if gname:
                if dct['frame_index'] in tmp:
                    tmp.append(dct['frame_index'])
                    match_item_dct[dct['match_item']] += 1

                dct_gallery[dct['match_item']].append(
                    {
                        'item_box': dct['item_box'],
                        'img_name': dct['img_name'],
                        'distance': dct['distance'],
                        'frame_index': dct['frame_index'],
                        'frame_box': dct['frame_box'],
                    }
                )

        gname_list_cnt = sorted(match_item_dct.items(), key=lambda x: x[1], reverse=True)
        if gname_list_cnt[0][1] >= 5:
            ## 能匹配到确定结果
            res_dct = {}
            #print('v'*20,v)
            tmp_list = sorted(dct_gallery[gname_list_cnt[0][0]],key=lambda x:x['distance'],reverse=True)
            #print(vt,vt[0])
            res_dct['item_id'] = gname_list_cnt[0][0]
            res_dct["frame_index"] = tmp_list[0]['frame_index']
            tmp_dct = {}
            tmp_dct["img_name"] = tmp_list[0]['img_name']
            tmp_dct["item_box"] = tmp_list[0]['item_box']
            tmp_dct["frame_box"]= tmp_list[0]['frame_box']
            res_dct['result'] = [tmp_dct]
            result[key] = res_dct
        else:
            result[key] = {}


    print('len key',len(result.keys()))
    with open('/myspace/result.json','w',encoding='utf-8') as f:
            json.dump(result,f)


def single_gpu_test_video_c(model, data_loader, gf, gnames, gboxs):
    model.eval()

    dataset = data_loader.dataset
    n = gf.size(0)
    #print(gboxs.size(),len(gnames),gnames)
    prog_bar = mmcv.ProgressBar(len(dataset))
    res = defaultdict(list)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            qf, name, gt_bbox = model(return_loss=False, **data)
        name = name[0]
        #print('name:', name)
        gt_bbox = gt_bbox.squeeze(0)
        v_name = name.split('_')[1]         # video name
        v_index = str(name.split('_')[2])  # video index

        m = qf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        #print('distmat',distmat.size())
        v, dim = torch.min(distmat, 1, keepdim=True)
        #print('min',v,dim)
        qv, qdim = torch.min(v,0)
        #print(qv,qdim)
        # 提取 box
        qbox = gt_bbox[qdim]        # qurey video box
        gbox = gboxs[dim[qdim]]     # gallery images box
        # 提取 name, qname 是同一张图片，不需要特殊处理
        #print('dim[qdim]',dim[qdim])
        gname = gnames[dim[qdim].item()]
        #print('gname',gname)

        if '/' in gname or '\\' in gname:
            gname = os.path.basename(gname)
        if '.' in gname:
            gname = gname[:-4]

        dct = {
            'frame_index':v_index,
            'frame_box':qbox,
            'item_box':gbox,
            'img_name':gname,
            'distance':qv.item()
        }  #[v_index,min_di]
        #print('dict','+'*20,dct)
        res[v_name].append(dct.copy())

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    result = {}
    for key in res.keys():
        v = res[key]
        res_dct = {}
        #print('v'*20,v)
        vt = sorted(v,key=lambda x:x['distance'])
        #print(vt,vt[0])
        name = vt[0]['img_name']
        res_dct['item_id'] = name.split('_')[2]
        res_dct["frame_index"] = int(vt[0]['frame_index'])
        tmp_dct = {}
        tmp_dct["img_name"] = name.split('_')[1]
        tmp_dct["item_box"] = vt[0]['item_box'][0].cpu().numpy().tolist()[0]
        tmp_dct["frame_box"]= vt[0]['frame_box'][0].cpu().numpy().tolist()
        res_dct['result'] = [tmp_dct]
        result[key] = res_dct

    print('len key',len(result.keys()))
    with open('/root/code/myspace/result.json','w',encoding='utf-8') as f:
            json.dump(result,f)




def multi_gpu_test1(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu1(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    #model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    model = IRNet(cfg.backbone, cfg.loss, cfg.bbox_roi_extractor, 100, cfg.pretrained)


    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs, names, gt_bboxes = single_gpu_test(model, data_loader, args.show)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs, names, gt_bboxes = multi_gpu_test(model, data_loader, args.show, args.tmpdir,
                                 args.gpu_collect)

    outputs = torch.cat(outputs, dim=0).cuda(non_blocking=True)
    #gt_bboxes = torch.cat(gt_bboxes, dim=0).cuda(non_blocking=True)
    # tmp = []
    # for ni in names:
    #     tmp.extend(ni)
    # names = tmp
    print('gallery:',outputs.size())
    print('gt_bboxes',len(gt_bboxes))
    print('names',len(names))

    dataset = build_dataset(cfg.data.test_video)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        single_gpu_test_video(model, data_loader, outputs, names,gt_bboxes)
    else:
        model = MMDistributedDataParallel(model.cuda())
        multi_gpu_test_video(model, data_loader, outputs, names,gt_bboxes)




if __name__ == '__main__':
    main()
