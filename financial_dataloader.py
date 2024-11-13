import random
random.seed(42)  # always sample the same batches
import torch
from torch.utils import data
import pandas as pd
# disable chained assignments
pd.options.mode.chained_assignment = None

"""parent code"""
from utils import default_device, set_locals_in_self                       # if gpu is available, use cuda:0
from train import Losses
"""our code"""
from data_utils import get_financial_data    # data is loaded


"""
Dataloader for real data
"""

class DataLoader(data.DataLoader):
    """
    rewrite the dataloader with a validation method
    """
    # Caution, you might need to set self.num_features manually if it is not part of the args.
    def __init__(self, num_steps, fuse_x_y=False, train=True, **get_batch_kwargs):
        set_locals_in_self(locals())
        
        # The stuff outside the or is set as class attribute before instantiation.
        self.num_features = get_batch_kwargs.get('num_features') or self.num_features
        self.num_outputs = get_batch_kwargs.get('num_outputs') or self.num_outputs
        print('DataLoader.__dict__', self.__dict__)
        
        # load train dataset
        self.train_X, self.train_Y = get_financial_data(train, **get_batch_kwargs)
        self.all_batches = range(self.train_X.shape[0])
        print('Train =', train, ', X shape:', self.train_X.shape, ', Y shape:', self.train_Y.shape)
        
        # make sure reload dataset before all data used up
        self.num_steps = min(self.train_X.shape[0]//get_batch_kwargs.get('batch_size'), self.num_steps)
        print('Number of steps =', self.num_steps)

    def gbm(self, fuse_x_y, train, step, **kwargs):
        """
        For every step, we randomly sample from some two dates of training data to get batches (batch_size).
        After many steps in an epoch, we do not repeat sample and use up all datapoints available (total batches > batch_size * step_per_epoch).
        Only train_X, train_Y will be modified in the method
        :return: x, y, y. 
              x.shape = (seq_len=L stocks trading on two dates, batch_size=resample L stocks m times, num_features=30 features)
        """

        if train:  # training: random sampling
            sample_batches = random.sample(self.all_batches, kwargs["batch_size"])  # sampled indexes, without replacement
        else:  # validation: fixed sample first batch_size data
            sample_batches = self.all_batches[:kwargs["batch_size"]]
            
        result_X = self.train_X[sample_batches,:,:]
        result_Y = self.train_Y[sample_batches,:]

        # for train/validation dataset: remove already used samples
        if not kwargs["sample_replacement"]:  # sample_replacement=False, no repeated data for every step in the same epoch
            self.all_batches = sorted(list(set(self.all_batches) - set(sample_batches)))  # remaining batches indexes
            
        # for next epoch: get new train dataloader if data used up in one epoch
        if (step == 1 or self.train_X.shape[0] < kwargs["batch_size"]):
            self.all_batches = range(self.train_X.shape[0])
            print('Reload data for next epoch... Train =', train, ', X shape:', self.train_X.shape, ', Y shape:', self.train_Y.shape)

        # X.shape: (batch_size,seq_len,num_features)->(seq_len,batch_size,num_features)
        x = torch.tensor(result_X, device=default_device, dtype=torch.float32).swapaxes(1, 0)  
        # targets: float for binary, .long() for multiclass in crossentropyloss()
        y = torch.tensor(result_Y, device=default_device, dtype=torch.float32).swapaxes(1, 0)
        target_y = y
        
        if fuse_x_y:
            return torch.cat([x, torch.cat([torch.zeros_like(y[:1]), y[:-1]], 0).unsqueeze(-1).float()], -1), target_y
        else:
            return (x, y), target_y

    def __len__(self):
        return self.num_steps

    def __iter__(self):  # pass in "get_batch_kwargs", fuse_x_y and train in self.variables
        return iter(self.gbm(**self.get_batch_kwargs,
                             fuse_x_y=self.fuse_x_y, train=self.train, step=step) for step in range(self.num_steps, 0, -1))

    @torch.no_grad()
    def validate(self, finetuned_model):
        """
        validation data used to tune hyperparameters: learning rate, etc
        validate every 10 epochs (if validation_period=10)
        :param finetuned_model: fine tuned model after every 10 epochs
        :param eval_pos: default is to evaluate score of the last date
        DO 100 BOOTSTRAPS OF VALIDATION DATASET AND CALCULATE VALIDATION SCORE AS THE MEAN AND STD
        """
        finetuned_model.eval()
        device = next(iter(finetuned_model.parameters())).device
        
        # reload valid dataloader for every epoch
        # one dataloader for training, another for valid, valid dataset: train=False
        if not hasattr(self, 't_dl'):
            print("load t_dl...")
            self.t_dl = DataLoader(num_steps=self.num_steps, fuse_x_y=self.fuse_x_y, train=False, **self.get_batch_kwargs)

        ps = []
        ys = []
        for x,y in self.t_dl:  # num_steps
            eval_pos = x[0].shape[0] // 2  # seq_len//2
            p = finetuned_model(tuple(e.to(device) for e in x), single_eval_pos=eval_pos)
            # p.shape: torch.Size([eval_pos, batch_size, num_outputs]), y.shape: torch.Size([eval_pos, batch_size])
            ps.append(p)
            ys.append(y[eval_pos:])

        ps = torch.cat(ps,1)
        ys = torch.cat(ys,1)
        # ps.shape: torch.Size([eval_pos, batch_size*num_steps, num_outputs]), ys.shape: torch.Size([eval_pos, batch_size*num_steps])
        
        # set loss function
        if self.num_outputs > 1:          # for multiclass targets
            losses = Losses.ce(ps.reshape(-1, self.num_outputs), ys.to(device).flatten().long())
        else:                      # for binary targets: sigmoid() + bce()
            losses = Losses.bce(ps.flatten(), ys.to(device).flatten())
        # output: (eval_pos*batch_size*num_steps, n_out), y: (eval_pos*batch_size*num_steps)
        # losses: (eval_pos*batch_size*num_steps)

        return losses.mean().item()


DataLoader.num_outputs = 1                # default for binary, set it in config for multiclass