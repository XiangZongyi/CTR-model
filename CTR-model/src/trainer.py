from tqdm import tqdm
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

from utils import set_color


class Trainer():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay

        self.optimizer = self._build_optimizer(name=args.optimizer, params=self.model.parameters())

        self._writer = SummaryWriter(log_dir=args.tensorboard_dir)

    def _build_optimizer(self, name, params):
        r"""Init the Optimizer
        Returns:
            torch.optim: the optimizer
        """
        if name.lower() == 'adam':
            optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            print('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=self.learning_rate)
        return optimizer

    def train_an_epoch(self, train_data, epoch_id):
        self.model.train()
        total_loss = 0
        iter_data = tqdm(train_data, total=len(train_data), ncols=100, desc=set_color(f"Train {epoch_id:>5}", 'pink'))
        for batch_id, interaction in enumerate(iter_data):
            self.optimizer.zero_grad()
            loss = self.model.calculate_loss(interaction)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            iter_data.set_postfix(Batch_Loss=loss.item())
        self._writer.add_scalar('model/loss', total_loss, epoch_id)
    
    def evaluate(self, test_data, epoch_id):
        self.model.eval()
        AUC = 0
        batch_num = 0
        with torch.no_grad():
            iter_data = tqdm(test_data, total=len(test_data), ncols=100, desc=set_color(f"Evaluate   ", 'pink'))
            for batch_idx, interaction in enumerate(iter_data):
                user, item, rating = interaction[0], interaction[1], interaction[2]
                user = user.to(self.args.device)
                item = item.to(self.args.device)
                rating_pred = self.model.predict(user, item)
                auc = roc_auc_score(rating.numpy(), rating_pred.cpu().numpy())
                batch_num += 1
                AUC += auc
            AUC /= batch_num
            self._writer.add_scalar('model/AUC', AUC, epoch_id)
        return AUC
