import os
import argparse
from six import iteritems
from itertools import product
from time import gmtime, strftime

def read_command_line(argv=None):
    parser = argparse.ArgumentParser(description='Large Scale Pretraining for Visual Dialog')

    #-------------------------------------------------------------------------
    # Data input settings
    parser.add_argument('-visdial_processed_train', default='data/visdial/visdial_1.0_train_processed.json', \
                                 help='json file containing train split of visdial data')
    parser.add_argument('-visdial_processed_val', default='data/visdial/visdial_1.0_val_processed.json',
                            help='json file containing val split of visdial data')
    parser.add_argument('-visdial_processed_test', default='data/visdial/visdial_1.0_test_processed.json',
                            help='json file containing test split of visdial data')
    parser.add_argument('-visdial_image_feats', default='data/visdial/visdial_img_feat.lmdb',
                            help='json file containing image feats for train,val and splits of visdial data')
    parser.add_argument('-visdial_processed_train_dense', default='data/visdial/visdial_1.0_train_dense_processed.json',
                            help='samples on the train split for which dense annotations are available')
    parser.add_argument('-visdial_processed_train_dense_annotations', default='data/visdial/visdial_1.0_train_dense_annotations_processed.json',
                            help='json file containing dense annotations on some instance of the train split')
    parser.add_argument('-visdial_processed_val_dense_annotations', default='data/visdial/visdial_1.0_val_dense_annotations_processed.json',
                            help='JSON file with dense annotations')
    parser.add_argument('-start_path', default='', help='path of starting model checkpt')
    parser.add_argument('-model_config', default='config/bert_base_6layer_6conect.json', help='model definition of the bert model')
    #-------------------------------------------------------------------------
    # Logging settings
    parser.add_argument('-enable_visdom', type=int, default=0,
                            help='Flag for enabling visdom logging')
    parser.add_argument('-visdom_env', type=str, default='',
                            help='Name of visdom environment for plotting')
    parser.add_argument('-visdom_server', type=str, default='http://asimo.cc.gatech.edu',
                            help='Address of visdom server instance')
    parser.add_argument('-visdom_server_port', type=int, default=7777,
                            help='Port of visdom server instance')
    #-------------------------------------------------------------------------
    # Optimization / training params
    # Other training environmnet settings
    parser.add_argument('-num_workers', default=8, type=int,
                            help='Number of worker threads in dataloader')  
    parser.add_argument('-batch_size', default=80, type=int,
                            help='size of mini batch')
    parser.add_argument('-num_epochs', default=20, type=int,
                            help='total number of epochs')
    parser.add_argument('-batch_multiply', default=1, type=int,
                            help='amplifies batch size in mini-batch training')    
    parser.add_argument('-lr',default=2e-5,type=float,help='learning rate')
    parser.add_argument('-image_lr',default=2e-5,type=float,help='learning rate for vision params')

    parser.add_argument('-overfit', action='store_true', help='overfit for debugging')
    parser.add_argument('-continue', action='store_true', help='continue training')

    parser.add_argument('-num_train_samples',default=0,type=int, help='number of train samples, set 0 to include all')
    parser.add_argument('-num_val_samples',default=0, type=int, help='number of val samples, set 0 to include all')
    parser.add_argument('-num_options',default=100, type=int, help='number of options to use. Max: 100 Min: 2')
    parser.add_argument('-n_gpus',default=8, type=int, help='number of gpus running the job')
    parser.add_argument('-sequences_per_image',default=8, type=int, help='number of sequences sampled from an image during training')
    parser.add_argument('-visdial_tot_rounds',default=11, type=int,  \
               help='number of rounds to use in visdial,caption is counted as a separate round, therefore a maximum of 11 rounds possible')
    parser.add_argument('-max_seq_len',default=256, type=int, help='maximum sequence length for the dialog sequence')
    parser.add_argument('-num_negative_samples',default=1, type=int, help='number of negative samples for every positive sample for the nsp loss')

    parser.add_argument('-lm_loss_coeff',default=1,type=float,help='Coeff for lm loss')
    parser.add_argument('-nsp_loss_coeff',default=1,type=float,help='Coeff for nsp loss')
    parser.add_argument('-img_loss_coeff',default=1,type=float,help='Coeff for img masked loss')

    parser.add_argument('-mask_prob',default=0.15,type=float,help='prob used to sample masked tokens')

    parser.add_argument('-save_path', default='checkpoints/',
                            help='Path to save checkpoints')
    parser.add_argument('-save_name', default='',
                            help='Name of save directory within savePath')

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    try:
        parsed = vars(parser.parse_args(args=argv))
        if parsed['save_name']:
            # Custom save file path
            parsed['save_path'] = os.path.join(parsed['save_path'],
                                            parsed['save_name'])
        else:
            # Standard save path with time stamp
            import random
            timeStamp = strftime('%d-%b-%y-%X-%a', gmtime())
            parsed['save_path'] = os.path.join(parsed['save_path'], timeStamp)
            parsed['save_path'] += '_{:0>6d}{}'.format(random.randint(0, 10e6),parsed['visdom_env'])
   
        assert parsed['sequences_per_image'] <= 8 
        assert parsed['visdial_tot_rounds'] <= 11

    except IOError as msg:
        parser.error(str(msg))

    return parsed