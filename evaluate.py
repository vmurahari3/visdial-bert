import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from dataloader.dataloader_visdial import VisdialDataset
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
from train import forward

def eval_ai_generate(dataloader, params, eval_batch_size, split='test'):
    ranks_json = []
    dialog_encoder.eval()
    batch_idx = 0
    with torch.no_grad():
        batch_size = 500 * (params['n_gpus']/8)
        batch_size = min([1, 2, 4, 5, 100, 1000, 200, 8, 10, 40, 50, 500, 20, 25, 250, 125], \
             key=lambda x: abs(x-batch_size) if x <= batch_size else float("inf"))
        print("batch size for evaluation", batch_size)
        for epochId, _, batch in batch_iter(dataloader, params):
            if epochId == 1:
                break

            tokens = batch['tokens']
            num_rounds = tokens.shape[1]
            num_options = tokens.shape[2]
            tokens = tokens.view(-1, tokens.shape[-1])                       
            segments = batch['segments']
            segments = segments.view(-1, segments.shape[-1])
            sep_indices = batch['sep_indices']
            sep_indices = sep_indices.view(-1, sep_indices.shape[-1])
            mask = batch['mask']
            mask = mask.view(-1, mask.shape[-1])
            hist_len = batch['hist_len']
            hist_len = hist_len.view(-1)
            
            # get image features
            features = batch['image_feat'] 
            spatials = batch['image_loc'] 
            image_mask = batch['image_mask']

            # expand the image features to match those of tokens etc.
            max_num_regions = features.shape[-2]
            features = features.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options, max_num_regions, 2048).contiguous()
            spatials = spatials.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options, max_num_regions, 5).contiguous()
            image_mask = image_mask.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options, max_num_regions).contiguous()

            features = features.view(-1, max_num_regions, 2048)
            spatials = spatials.view(-1, max_num_regions, 5)
            image_mask = image_mask.view(-1, max_num_regions)

            assert tokens.shape[0] == segments.shape[0] == sep_indices.shape[0] == mask.shape[0] == \
                hist_len.shape[0] == features.shape[0] == spatials.shape[0] == \
                    image_mask.shape[0] == num_rounds * num_options * eval_batch_size

            output = []
            assert (eval_batch_size * num_rounds * num_options)//batch_size == (eval_batch_size * num_rounds * num_options)/batch_size
            for j in range((eval_batch_size * num_rounds * num_options)//batch_size):
                # create chunks of the original batch
                item = {}
                item['tokens'] = tokens[j*batch_size:(j+1)*batch_size,:]
                item['segments'] = segments[j*batch_size:(j+1)*batch_size,:]
                item['sep_indices'] = sep_indices[j*batch_size:(j+1)*batch_size,:]
                item['mask'] = mask[j*batch_size:(j+1)*batch_size,:]
                item['hist_len'] = hist_len[j*batch_size:(j+1)*batch_size]

                item['image_feat'] = features[j*batch_size:(j+1)*batch_size, : , :]
                item['image_loc'] = spatials[j*batch_size:(j+1)*batch_size, : , :]
                item['image_mask'] = image_mask[j*batch_size:(j+1)*batch_size, :]

                _, _, _, _, nsp_scores = forward(dialog_encoder, item, params, output_nsp_scores=True, evaluation=True)
                # normalize nsp scores
                nsp_probs = F.softmax(nsp_scores, dim=1)
                assert nsp_probs.shape[-1] == 2
                output.append(nsp_probs[:,0])

            # print("output shape",torch.cat(output,0).shape)
            output = torch.cat(output,0).view(eval_batch_size, num_rounds, num_options)
            ranks = scores_to_ranks(output)
            ranks = ranks.squeeze(1)
            for i in range(eval_batch_size):
                ranks_json.append(
                    {
                        "image_id": batch["image_id"][i].item(),
                        "round_id": int(batch["round_id"][i].item()),
                        "ranks": [
                            rank.item()
                            for rank in ranks[i][:]
                        ],
                    }
                    )

            batch_idx += 1
    return ranks_json
if __name__ == '__main__':

    params = options.read_command_line()
    pprint.pprint(params)
    dataset = VisdialDataset(params)
    eval_batch_size = 5
    split = 'test'
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=params['num_workers'],
        drop_last=False,
        pin_memory=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['device'] = device
    dialog_encoder = VisualDialogEncoder(params['model_config'])

    if params['start_path']:
        pretrained_dict = torch.load(params['start_path'])

        if 'model_state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['model_state_dict']

        model_dict = dialog_encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("number of keys transferred", len(pretrained_dict))
        assert len(pretrained_dict.keys()) > 0
        model_dict.update(pretrained_dict)
        dialog_encoder.load_state_dict(model_dict)

    dialog_encoder = nn.DataParallel(dialog_encoder)
    dialog_encoder.to(device)
    ranks_json = eval_ai_generate(dataloader, params, eval_batch_size, split=split)

    json.dump(ranks_json, open(params['save_name'] + '_predictions.txt', "w"))
