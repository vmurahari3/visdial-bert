import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import options
from models.visual_dialog_encoder import VisualDialogEncoder
import torch.optim as optim
from utils.visualize import VisdomVisualize
import pprint
from time import gmtime, strftime
from timeit import default_timer as timer
from pytorch_transformers.optimization import AdamW
import os
from utils.visdial_metrics import SparseGTMetrics, NDCG, scores_to_ranks
from pytorch_transformers.tokenization_bert import BertTokenizer
from utils.data_utils import sequence_mask, batch_iter
from utils.optim_utils import WarmupLinearScheduleNonZero
import json
import logging
from dataloader.dataloader_dense_annotations import VisdialDatasetDense
from dataloader.dataloader_visdial import VisdialDataset

from train import forward, visdial_evaluate

if __name__ == '__main__':

    params = options.read_command_line()
    os.makedirs('checkpoints', exist_ok=True)
    if not os.path.exists(params['save_path']):
        os.mkdir(params['save_path'])
    viz = VisdomVisualize(
        enable=bool(params['enable_visdom']),
        env_name=params['visdom_env'],
        server=params['visdom_server'],
        port=params['visdom_server_port'])
    pprint.pprint(params)
    viz.addText(pprint.pformat(params, indent=4))

    dataset = VisdialDatasetDense(params)

    num_images_batch = 1

    dataset.split = 'train'
    dataloader = DataLoader(
        dataset,
        batch_size= num_images_batch,
        shuffle=False,
        num_workers=params['num_workers'],
        drop_last=True,
        pin_memory=False)

    eval_dataset = VisdialDataset(params)
    eval_dataset.split = 'val'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['device'] = device
    dialog_encoder = VisualDialogEncoder(params['model_config'])

    param_optimizer = list(dialog_encoder.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    langauge_weights = None
    with open('config/language_weights.json') as f:
        langauge_weights = json.load(f)

    optimizer_grouped_parameters = []
    for key, value in dict(dialog_encoder.named_parameters()).items():
        if value.requires_grad:
            if key in langauge_weights:
                lr = params['lr'] 
            else:
                lr = params['image_lr']

            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0}
                ]

            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=params['lr'])
    scheduler = WarmupLinearScheduleNonZero(optimizer, warmup_steps=10000, t_total=200000)
    startIterID = 0

    if params['start_path']:

        pretrained_dict = torch.load(params['start_path'])

        if not params['continue']:
            if 'model_state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['model_state_dict']

            model_dict = dialog_encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print("number of keys transferred", len(pretrained_dict))
            assert len(pretrained_dict.keys()) > 0
            model_dict.update(pretrained_dict)
            dialog_encoder.load_state_dict(model_dict)
        else:
            model_dict = dialog_encoder.state_dict()
            optimizer_dict = optimizer.state_dict()
            pretrained_dict_model = pretrained_dict['model_state_dict']
            pretrained_dict_optimizer = pretrained_dict['optimizer_state_dict']
            pretrained_dict_scheduler = pretrained_dict['scheduler_state_dict']
            pretrained_dict_model = {k: v for k, v in pretrained_dict_model.items() if k in model_dict}
            pretrained_dict_optimizer = {k: v for k, v in pretrained_dict_optimizer.items() if k in optimizer_dict}
            model_dict.update(pretrained_dict_model)
            optimizer_dict.update(pretrained_dict_optimizer)
            dialog_encoder.load_state_dict(model_dict)
            optimizer.load_state_dict(optimizer_dict)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            scheduler = WarmupLinearScheduleNonZero(optimizer, warmup_steps=10000, \
                 t_total=200000, last_epoch=pretrained_dict["iterId"])
            scheduler.load_state_dict(pretrained_dict_scheduler)
            startIterID = pretrained_dict['iterId']

            del pretrained_dict, pretrained_dict_model, pretrained_dict_optimizer, pretrained_dict_scheduler, \
                model_dict, optimizer_dict
            torch.cuda.empty_cache()

    num_iter_epoch = dataset.numDataPoints['train'] // num_images_batch if not params['overfit'] else 1
    print('\n%d iter per epoch.' % num_iter_epoch)

    dialog_encoder = nn.DataParallel(dialog_encoder)
    dialog_encoder.to(device)

    start_t = timer()
    optimizer.zero_grad()
    # kl div reduces to ce if the target distribution is fixed
    ce_loss_fct =  nn.KLDivLoss(reduction='batchmean')
    
    for epoch_id, idx, batch in batch_iter(dataloader, params):
        iter_id = startIterID + idx + (epoch_id * num_iter_epoch)
        dialog_encoder.train()
        # expand image features, 
        features = batch['image_feat'] 
        spatials = batch['image_loc'] 
        image_mask = batch['image_mask']
        image_label = batch['image_label']
        image_target = batch['image_target']

        num_rounds = batch["tokens"].shape[1]
        num_samples = batch["tokens"].shape[2]

        # sample 80 options including the gt option due to memory constraints
        assert num_images_batch == 1
        gt_option_ind = batch['gt_option'].item()
        all_inds_minus_gt = torch.cat([torch.arange(gt_option_ind), torch.arange(gt_option_ind + 1,100)],0)
        all_inds_minus_gt = all_inds_minus_gt[torch.randperm(99)[:79]]
        option_indices = torch.cat([batch['gt_option'].view(-1), all_inds_minus_gt] , 0)

        features = features.unsqueeze(1).unsqueeze(1).expand(features.shape[0], num_rounds, 80, features.shape[1], features.shape[2])
        spatials = spatials.unsqueeze(1).unsqueeze(1).expand(spatials.shape[0], num_rounds, 80, spatials.shape[1], spatials.shape[2])
        image_mask = image_mask.unsqueeze(1).unsqueeze(1).expand(image_mask.shape[0], num_rounds, 80, image_mask.shape[1])
        image_label = image_label.unsqueeze(1).unsqueeze(1).expand(image_label.shape[0], num_rounds, 80, image_label.shape[1])
        image_target = image_target.unsqueeze(1).unsqueeze(1).expand(image_target.shape[0], num_rounds, 80, image_target.shape[1],image_target.shape[2])

        features = features.view(-1, features.shape[-2], features.shape[-1])
        spatials = spatials.view(-1, spatials.shape[-2], spatials.shape[-1])
        image_mask = image_mask.view(-1, image_mask.shape[-1])
        image_label = image_label.view(-1, image_label.shape[-1])
        image_target = image_target.view(-1, image_target.shape[-2], image_target.shape[-1])

        # reshape text features
        tokens = batch['tokens']
        segments = batch['segments']
        sep_indices = batch['sep_indices'] 
        mask = batch['mask']
        hist_len = batch['hist_len']
        nsp_labels = batch['next_sentence_labels']

        # select 80 options from the 100 options including the GT option
        tokens = tokens[:, :, option_indices, :]
        segments = segments[:, :, option_indices, :]
        sep_indices = sep_indices[:, :, option_indices, :]
        mask = mask[:, :, option_indices, :]
        hist_len = hist_len[:, :, option_indices]
        nsp_labels = nsp_labels[:, :, option_indices]

        tokens = tokens.view(-1, tokens.shape[-1])
        segments = segments.view(-1, segments.shape[-1])
        sep_indices = sep_indices.view(-1, sep_indices.shape[-1])
        mask = mask.view(-1, mask.shape[-1])
        hist_len = hist_len.view(-1)
        nsp_labels = nsp_labels.view(-1)
        nsp_labels = nsp_labels.to(params['device'])

        batch['tokens'] = tokens
        batch['segments'] = segments
        batch['sep_indices'] = sep_indices
        batch['mask'] = mask
        batch['hist_len'] = hist_len
        batch['next_sentence_labels'] = nsp_labels

        batch['image_feat'] = features.contiguous()
        batch['image_loc'] = spatials.contiguous()
        batch['image_mask'] = image_mask.contiguous()
        batch['image_target'] = image_target.contiguous()
        batch['image_label'] = image_label.contiguous()

        print("token shape", tokens.shape)
        loss = 0
        nsp_loss = 0
        _, _, _, _, nsp_scores = forward(dialog_encoder, batch, \
        params, sample_size=None, output_nsp_scores=True, evaluation=True)
        logging.info("nsp scores: {}".format(nsp_scores))        
        # calculate dense annotation ce loss        
        nsp_scores = nsp_scores.view(-1, 80, 2)
        nsp_loss = F.cross_entropy(nsp_scores.view(-1,2), nsp_labels.view(-1)) 
        nsp_scores = nsp_scores[:, :, 0]
        
        gt_relevance = batch['gt_relevance'].to(device)
        # shuffle the gt relevance scores as well
        gt_relevance = gt_relevance[:, option_indices]
        ce_loss = ce_loss_fct(F.log_softmax(nsp_scores, dim=1), F.softmax(gt_relevance, dim=1))
        loss = ce_loss +  params['nsp_loss_coeff'] * nsp_loss
        loss /= params['batch_multiply']
        loss.backward()        
        scheduler.step()

        if iter_id % params['batch_multiply'] == 0 and iter_id > 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if iter_id % 10 == 0:
            # Update line plots
            viz.linePlot(iter_id, loss.item(), 'loss', 'tot loss')
            viz.linePlot(iter_id, nsp_loss.item(), 'loss', 'nsp loss')
            viz.linePlot(iter_id, ce_loss.item(), 'loss', 'ce loss')
        
        old_num_iter_epoch = num_iter_epoch
        if params['overfit']:
            num_iter_epoch = 100
        if iter_id % num_iter_epoch == 0:
            torch.save({'model_state_dict' : dialog_encoder.module.state_dict(),'scheduler_state_dict':scheduler.state_dict() \
                 ,'optimizer_state_dict': optimizer.state_dict(), 'iter_id':iter_id}, os.path.join(params['save_path'], 'visdial_dialog_encoder_%d.ckpt'%iter_id))

        if iter_id % num_iter_epoch == 0 and iter_id > 0:
            viz.save()
        # fire evaluation
        print("num iteration for eval", num_iter_epoch)
        if iter_id % num_iter_epoch == 0 and iter_id > 0:
            eval_batch_size = 2
            if params['overfit']:
                eval_batch_size = 5
            
            # each image will need 1000 forward passes, (100 at each round x 10 rounds).
            dataloader = DataLoader(
                eval_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=params['num_workers'],
                drop_last=True,
                pin_memory=False)
            all_metrics = visdial_evaluate(dataloader, params, eval_batch_size, dialog_encoder)
            for metric_name, metric_value in all_metrics.items():
                print(f"{metric_name}: {metric_value}")
                if 'round' in metric_name:
                    viz.linePlot(iter_id, metric_value, 'Retrieval Round Val Metrics Round -' + metric_name.split('_')[-1], metric_name)
                else:
                    viz.linePlot(iter_id, metric_value, 'Retrieval Val Metrics', metric_name)
        
        num_iter_epoch = old_num_iter_epoch