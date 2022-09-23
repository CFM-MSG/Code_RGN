import collections
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from models.loss import ivc_loss, cal_nll_loss, rec_loss, localize_loss
from utils import TimeMeter, AverageMeter

def info(msg):
    print(msg)
    logging.info(msg)


class MainRunner:
    def __init__(self, args):
        self.args = args
        self._build_dataset()

        self.args['model']['config']['vocab_size'] = self.train_set.vocab_size
        self.args['model']['config']['max_epoch'] = self.args['train']['max_num_epochs']

        self._build_model()
        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0

        self.rec_miou = []

    def train(self):
        best_results = None
        for epoch in range(1, self.args['train']['max_num_epochs']+1):
            info('Start Epoch {}'.format(epoch))
            self.model_saved_path = self.args['train']['model_saved_path']
            os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
            save_path = os.path.join(self.model_saved_path, 'model-{}.pt'.format(epoch))
            self._train_model(epoch=epoch)
            self._save_model(save_path)
            results = self._eval_model(epoch=epoch)
            if best_results is None or results['R@1,mIoU'].avg > best_results['R@1,mIoU'].avg:
                best_results = results
                os.system('cp %s %s'%(save_path, os.path.join(self.model_saved_path, 'model-best.pt')))
                info('Best results have been updated.')
            info('=' * 60)


        info(self.rec_miou)

        msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in best_results.items()])
        info('Best results:')
        info('|'+msg+'|')
 
    def _train_model(self, epoch, **kwargs):
        self.model.train()

        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, bid, curr_lr)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            info(msg)

        display_n_batches, bid = 50, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())

        for bid, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training epoch {}".format(epoch)):
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            bsz = net_input["glance"].shape[0]
            output = self.model(epoch=epoch, **net_input)
            

            ## Semantic Loss
            loss, loss_dict = rec_loss(**output, num_props=self.model.num_props, **self.args['loss'])

            # Localization Loss
            props_glance = net_input["glance"].unsqueeze(1).expand(bsz, self.model.num_props).reshape(-1)
            term3_gl_loss, gl_loss_dict = localize_loss(**output, props_glance=props_glance, num_props=self.model.num_props, **self.args['loss'])
            loss_dict.update(gl_loss_dict)
            loss += term3_gl_loss

            ## Intra-video Loss
            rnk_loss, rnk_loss_dict = ivc_loss(**output, num_props=self.model.num_props, use_div_loss=True, **self.args['loss'])
            loss_dict.update(rnk_loss_dict)
            loss = loss + rnk_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()
            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)

            time_meter.update() 

            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()




    def _eval_model(self, save=None, epoch=0):
        self.model.eval()
        with torch.no_grad():
            metrics_logger = collections.defaultdict(lambda: AverageMeter())

            with torch.no_grad():
                import time
                time_start = time.time()
                pred_res_rec = []
                reglance_res = {}
                for bid, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="Eval epoch {}".format(epoch)):
                    durations = np.asarray([i[1] for i in batch['raw']])
                    gt = np.asarray([i[2] for i in batch['raw']])

                    net_input = move_to_cuda(batch['net_input'])

                    bsz = len(durations)
                    num_props = self.model.num_props
                    # num_props = 1
                    k = min(num_props, 5)

                    output = self.model(epoch=epoch, pseudo_labels=None, **net_input)
                    

                    words_mask = output['words_mask'].unsqueeze(1) \
                        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
                    words_id = output['words_id'].unsqueeze(1) \
                        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

                    nll_loss, acc = cal_nll_loss(output['words_logit'], words_id, words_mask)
                    idx = nll_loss.view(bsz, num_props).argsort(dim=-1)

                    width = output['width'].view(bsz, num_props).gather(index=idx, dim=-1)
                    center = output['center'].view(bsz, num_props).gather(index=idx, dim=-1)


                    selected_props = torch.stack([torch.clamp(center-width/2, min=0), torch.clamp(center+width/2, max=1)], dim=-1)

                    
                    selected_props = selected_props.cpu().numpy()
                    gt = gt / durations[:, np.newaxis]


                    r1_res = selected_props[:,0,:]*durations[:, np.newaxis]

                    r1_gt = gt * durations[:, np.newaxis]
                    r1_iou = calculate_IoU_batch((r1_res[:, 0], r1_res[:, 1]), (r1_gt[:, 0], r1_gt[:, 1]))

                    res = top_1_metric(selected_props[:, 0], gt)
                    
                    for key, v in res.items():
                        metrics_logger['R@1,'+key].update(v, bsz)
                    
                    res = top_n_metric(selected_props[:, :k].transpose(1, 0, 2), gt)
                    for key, v in res.items():
                        metrics_logger['R@%d,'%(k)+key].update(v, bsz)

            msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in metrics_logger.items()])
            info('|'+msg+'|')
            return metrics_logger


    def _build_dataset(self):
        import datasets as da
        import pickle
        from torch.utils.data import DataLoader
        args = self.args['dataset']
        cls = getattr(da, args['dataset'], None)
        with open(args['vocab_path'], 'rb') as fp:
            vocab = pickle.load(fp)
        self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, is_training=True, split='train')
        self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args, split='test')
        self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args, split='val') if args['val_data'] else None
        info('train: {} samples, test: {} samples'.format(len(self.train_set), len(self.test_set)))
        batch_size = self.args['train']['batch_size']

        def worker_init_fn(worker_id):
            def set_seed(seed):
                import random
                import numpy as np
                import torch

                random.seed(seed)
                np.random.seed(seed + 1)
                torch.manual_seed(seed + 3)
                torch.cuda.manual_seed(seed + 4)
                torch.cuda.manual_seed_all(seed + 4)

            set_seed(8 + worker_id)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                       collate_fn=self.train_set.collate_data, num_workers=4,
                                       worker_init_fn=worker_init_fn)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.test_set.collate_data,
                                      num_workers=4)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False,
                                     collate_fn=self.val_set.collate_data,
                                     num_workers=4) if args['val_data'] else None

    def _build_model(self):
        model_config = self.args['model']
        import models

        self.model = getattr(models, model_config['name'], None)(model_config['config'])
        self.model = self.model.cuda()
        print(self.model)
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total:', total_num, 'Trainable:', trainable_num)

    def _build_optimizer(self):
        from optimizers import AdamOptimizer
        from optimizers.lr_schedulers import InverseSquareRootSchedule
        
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        args = self.args['train']["optimizer"]
        self.optimizer = AdamOptimizer(args, parameters)
        self.lr_scheduler = InverseSquareRootSchedule(args, self.optimizer)

    def _save_model(self, path):
        state_dict = {
            'num_updates': self.num_updates,
            'config': self.args,
            'model_parameters': self.model.state_dict(),
            'model_parameters': self.model.state_dict(),
            'num_updates': self.num_updates,

        }
        torch.save(state_dict, path)
        info('save model to {}, num_updates {}.'.format(path, self.num_updates))

    def _load_model(self, path):
        state_dict = torch.load(path)
        self.num_updates = state_dict['num_updates']
        self.lr_scheduler.step_update(self.num_updates)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)

        info('load model from {}, num_updates {}.'.format(path, self.num_updates))


def calculate_IoU_batch(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


# [nb, 2], [nb, 2]
def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result

def top_1_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)
