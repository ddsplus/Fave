import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import datetime
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
import pickle
from utils import Data_Train, Data_Val, Data_Test, Data_CHLS
from model import create_model_Fave, Att_Fave_model
from auto_trainer import model_train

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='ml-100k', help='Dataset name')        # amazon_beauty, ml-100k, yelp, steam
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('--random_seed', type=int, default=1997, help='Random seed')
parser.add_argument('--max_len', type=int, default=50, help='The max length of sequence')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')
parser.add_argument("--hidden_size", default=128, type=int, help="hidden size of model")
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
parser.add_argument('--emb_dropout', type=float, default=0.0, help='Dropout of item embedding')
parser.add_argument("--hidden_act", default="gelu", type=str)
parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight')
parser.add_argument('--eval_interval', type=int, default=1, help='eval interval')
parser.add_argument('--patience', type=int, default=5, help='early stop patience')
parser.add_argument('--eps', type=float, default=0.001, help='start step')
parser.add_argument('--sample_N', type=int, default=30, help='Euler calculate steps')
parser.add_argument('--lambda_t', type=float, default=0.001, help='scale')
parser.add_argument('--dropout_c', type=float, default=0.1, help='scale')
parser.add_argument('--eps_reverse', type=float, default=0.001, help='reverse start step')
parser.add_argument('--m_logNorm', type=float, default=1.0, help='Logit-Normal Sampling mean')
parser.add_argument('--s_logNorm', type=float, default=0.6, help='Logit-Normal Sampling Variance')
parser.add_argument('--s_modsamp', type=float, default=1.0, help='Mode_sample_timestep scale parameter')
parser.add_argument('--last', type=int, default=2, help='last H Get')
parser.add_argument('--mask_ratio', type=float, default=1.0, help='Balanced positive-negative class ratio')
parser.add_argument('--sampling_method', type=str, default='mode', choices=['uniform', 'logit_normal', 'mode', 'cosmap'])
parser.add_argument('--Loss_Alpha', type=float, default=0.5, help='Loss parameter')
parser.add_argument('--Loss_Beta', type=float, default=0.4, help='Loss parameter')

parser.add_argument('--stage_mode', type=str, default='finetune', choices=['pretrain', 'finetune'], help='Training stage')
parser.add_argument('--ckpt_path', type=str, default='./best for ml-100k copy/pretrain_best_stage1.pt', help='Path to save/load checkpoint')
parser.add_argument('--finetune_lr_scale', type=float, default=0.1, help='Learning rate scale for finetuning')
parser.add_argument('--drop_path_rate', type=float, default=0.2, help='dropout path rate')

parser.add_argument('--loss_aux_weight', type=float, default=0.5, help='Weight for auxiliary decoder loss')
parser.add_argument('--loss_straight_weight', type=float, default=0.1, help='Weight for straightness loss')
parser.add_argument('--train_mask_ratio', type=float, default=0.5, help='Mask ratio during finetune training')
parser.add_argument('--infer_mask_ratio', type=float, default=0.6, help='Mask ratio during inference')
parser.add_argument('--gamma_init', type=float, default=1e-2, help='Initial value for LayerScale gamma')
parser.add_argument('--mask_end', type=float, default=0.75, help='Probability threshold for end masking')
parser.add_argument('--loss_pretrain_weight', type=float, default=0.3, help='Weight for decoder loss in pretrain stage')

args = parser.parse_args()

# Logger setup
if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)
if not os.path.exists(args.log_file + args.dataset):
    os.makedirs(args.log_file + args.dataset)

logging.basicConfig(level=logging.INFO, filename=args.log_file + args.dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)
logger.info(args)

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def item_num_create(args, item_num):
    args.item_num = item_num + 1
    return args

def user_num_create(args, user_num):
    args.user_num = user_num + 1
    return args

def main(args, logger):
    fix_random_seed_as(args.random_seed)
    path_data = os.path.join('.', 'datasets', 'data', args.dataset, 'dataset.pkl')
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)

    args = item_num_create(args, len(data_raw['smap']))
    args = user_num_create(args, len(data_raw['train'].items()))

    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()

    Fave_rec = create_model_Fave(args)
    rec_fm_joint_model = Att_Fave_model(Fave_rec, args)

    override_lr = args.lr
    override_params = None

    if args.stage_mode == 'pretrain':
        print("=== STAGE 1: PRETRAINING ===")
        rec_fm_joint_model.stage = 'pretrain'
        if not os.path.exists(os.path.dirname(args.ckpt_path)):
            os.makedirs(os.path.dirname(args.ckpt_path))

    elif args.stage_mode == 'finetune':
        print("=== STAGE 2: FINETUNING ===")

        # Load pretrained weights
        if not os.path.exists(args.ckpt_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found at {args.ckpt_path}")

        print(f"Loading pretrained weights from {args.ckpt_path}...")
        checkpoint = torch.load(args.ckpt_path, map_location=args.device)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        rec_fm_joint_model.load_state_dict(state_dict, strict=False)

        # Switch to finetune mode (freeze layers)
        if hasattr(rec_fm_joint_model, 'switch_to_finetune'):
            rec_fm_joint_model.switch_to_finetune()
        else:
            rec_fm_joint_model.stage = 'finetune'
            rec_fm_joint_model.item_embeddings.weight.requires_grad = False
            rec_fm_joint_model.position_embeddings.weight.requires_grad = False
            print("Manually froze embeddings.")

        # Set LR and optimizer parameters
        override_lr = args.lr * args.finetune_lr_scale
        override_params = filter(lambda p: p.requires_grad, rec_fm_joint_model.parameters())
        print(f"Finetune LR: {override_lr}")

    best_model, test_results = model_train(
        tra_data_loader, val_data_loader, test_data_loader,
        rec_fm_joint_model, args, logger,
        override_lr=override_lr,
        override_params=override_params
    )

if __name__ == '__main__':
    main(args, logger)
