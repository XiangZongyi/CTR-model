import argparse
from data_utils import LoadDataset, create_dataloader
from utils import init_seed

from model import FM, DeepFM, GraphCTR1, GraphCTR2
from trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='')
parser.add_argument('--dataset_path', type=str, default='/home/dou/xiangzy/CTR-model/data')
parser.add_argument('--dataset_name', type=str, default='ml-100k')
parser.add_argument('--data_split_ratio', type=list, default=[0.8, 0.2], help="")
parser.add_argument('--neg_sample_num', type=int, default=1, help="")
parser.add_argument('--seed', type=int, default=2020, help="")
parser.add_argument('--batch_size', type=int, default=2048, help="")
parser.add_argument('--test_batch_size', type=int, default=2048, help='')
parser.add_argument('--device', type=str, default='cuda', help="")
parser.add_argument('--embedding_size', type=int, default=10, help="the latent vector embedding size")
parser.add_argument('--n_layers', type=int, default=3, help="graphctr:3, deepfm:5")
parser.add_argument('--dropout', type=float, default=0.7, help="graphctr:, deepfm:0.2, DIN:0")
parser.add_argument('--learning_rate', type=float, default=0.001, help="")
parser.add_argument('--weight_decay', type=float, default=0.001, help="the weight decay of optimizer")
parser.add_argument('--optimizer', type=str, default='adam', help="")
parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard_log', help='the path to save tensorboard')
parser.add_argument('--model_name', type=str, default='graphctr2', help='graphctr1, graphctr2, fm ,deepfm')

args = parser.parse_args()

# init random seed
init_seed(args.seed)

# load dataset
dataset = LoadDataset(args)
train_dataset, inter_mat = dataset.get_train_dataset()
test_dataset = dataset.get_test_dataset()
user_feat_mat, item_feat_mat = dataset.get_feature_mat()

# load dataloader
train_data = create_dataloader(train_dataset, args.batch_size, training=True)
test_data = create_dataloader(test_dataset, args.test_batch_size, training=False)  # only user id for test dataloader

# get model
if args.model_name == 'graphctr1':
    model = GraphCTR1(args, dataset, inter_mat, user_feat_mat, item_feat_mat).to(args.device)
elif args.model_name == 'graphctr2':
    model = GraphCTR2(args, dataset, inter_mat, user_feat_mat, item_feat_mat).to(args.device)
elif args.model_name == 'fm':
    model = FM(args, dataset, user_feat_mat, item_feat_mat).to(args.device)
elif args.model_name == 'deepfm':
    model = DeepFM(args, dataset, user_feat_mat, item_feat_mat).to(args.device)
else:
    raise ValueError('The model name in args is error!')
print('The model is {}.'.format(args.model_name))

trainer = Trainer(args, model)

AUC_best = 0
epoch_best = 0
for epoch in range(args.epoch):
    trainer.train_an_epoch(train_data, epoch_id=epoch + 1)
    AUC = trainer.evaluate(test_data, epoch_id=epoch + 1)
    if AUC > AUC_best:
        AUC_best = AUC
        epoch_best = epoch + 1
    print("AUC: {:.4f}, Best AUC: {:.4f}, Best Epoch: {}".format(AUC, AUC_best, epoch_best))
