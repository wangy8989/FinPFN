import argparse
import datetime
import random
random.seed(42)  # always sample the same batches

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from tabpfn import TabPFNClassifier
"""parent code"""
from train import train, Losses
import encoders
from datasets import *
"""our code"""
import financial_dataloader
from data_utils import *  # test_set, get_dates_pair()


def get_default_config(class_type="multiclass"):
    """
    Configurations (default hyperparameters) for the transformer model
    From BayesianModels_And_Custom_Pyro_Modules.ipynb
    """
    config = {'lr': 2.006434218345026e-05   # 2.006434218345026e-05
         , 'epochs': 100            # <= 100
         , 'dropout': 0.0
         , 'emsize': 256
         , 'batch_size': 256         # 32, 64
         , 'nlayers': 5
         , 'num_outputs': 1          # vary for multiclass
         , 'num_features': 60         # 5,60
         , 'steps_per_epoch': 100
#          , 'warmup_epochs': 25       # epochs // 4
         , 'nhead': 4              # 256//64=4
         , 'seq_len': 300            # 200, 100
         , 'nhid_factor': 2}
    
    if class_type == "multiclass":
        config["lr"] = 3e-5
        config["num_outputs"] = 10     # 10 classes by default
        config["epochs"] = 60
        config["seq_len"] = 100
        config["validation_period"] = 1
        config["num_features"] = 30
        config["batch_size"] = 64
    elif class_type == "binary":
        config["lr"] = 5e-6
        config["epochs"] = 30
        config["seq_len"] = 100
        config["validation_period"] = 1
        config["num_features"] = 30
        config["batch_size"] = 64
    
    return config


def get_prior_config(config):
    """
    Configurations (default hyperparameters) for prior data
    """
    
    return {  'date_style': "consecutive"        # consecutive/random
            , 'sample_replacement': False
#             , 'select_features': None         # can set values as a list
#             , 'bins': None
            , 'multiclass': config['num_outputs'] if config['num_outputs'] > 1 else 2  # vary for multiclass
            , 'num_outputs': config['num_outputs']  # vary for multiclass
            , 'num_features': config['num_features']
            , 'fuse_x_y': False
            , 'device': "cpu"
            }


def train_model(config, prior_config, save=True, device='cuda:0'):
    """
    Train model using hyperparameters configurations
    """
    print("Training model...")
    
    # set loss function
    if config["num_outputs"] > 1:       # for multiclass targets
        criterion = Losses.ce
    else:                      # for binary targets
        criterion = Losses.bce  

    model = train(financial_dataloader.DataLoader  # financial data Dataloader
                  , criterion
                  , encoders.Linear
                  , emsize=config['emsize']
                  , nhead=config['nhead']
                  , y_encoder_generator=encoders.Linear
                  , pos_encoder_generator=None
                  , batch_size=config['batch_size']
                  , nlayers=config['nlayers']
                  , nhid=config['emsize'] * config['nhid_factor']
                  , epochs=config['epochs']
                  # small lr for first 25 epochs, later use our own lr
                  , warmup_epochs=config['warmup_epochs'] if 'warmup_epochs' in config else config['epochs']//4
                  , bptt=config['seq_len']
                  , gpu_device=device
                  , dropout=config['dropout']
                  , steps_per_epoch=config['steps_per_epoch']
                  , validation_period=config["validation_period"]  # default=10
                  , single_eval_pos_gen=config['seq_len']//2  # only split into a half train + a half test
                  # continue training the model
                  , load_weights_from_this_state_dict=config.pop('state_dict') if 'state_dict' in config else None
                  , extra_prior_kwargs_dict=prior_config
                  , lr=config['lr']
                  , verbose=True)
    
    config["extra_prior_kwargs_dict"] = prior_config
    print(criterion, "\n", config)
    
    if save:
        # save model, config
        datetimestr = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        model_path = f'model/finance_{datetimestr}.cpkt'
        print("Saving model...", model_path)
        torch.save((model[2].state_dict(), None, config), model_path)

    return model


def get_model(model_path, device='cpu'):
    """
    Load model from model path
    """
    print("Loading model...", model_path)
    model_state, _, config = torch.load(model_path)
    
    if config["num_outputs"] > 1:  # for multiclass targets
        criterion = Losses.ce
    else:                  # for binary targets
        criterion = Losses.bce  

    model = train(financial_dataloader.DataLoader  # data only loaded during training
                  , criterion  
                  , encoders.Linear
                  , emsize=config['emsize']
                  , nhead=config['nhead']
                  , y_encoder_generator=encoders.Linear
                  , pos_encoder_generator=None
                  , batch_size=config['batch_size']
                  , nlayers=config['nlayers']
                  , nhid=config['emsize'] * config['nhid_factor']
                  , epochs=0                        # if epochs=0, not training
                  , warmup_epochs=config['warmup_epochs'] if 'warmup_epochs' in config else config['epochs']//4
                  , bptt=config['seq_len']
                  , gpu_device=device
                  , dropout=config['dropout']
                  , steps_per_epoch=config['steps_per_epoch']
                  , validation_period=config["validation_period"]  # default=10
                  , single_eval_pos_gen=config['seq_len']//2 
                  , extra_prior_kwargs_dict=config["extra_prior_kwargs_dict"]
                  , lr=config['lr']
                  , verbose=True)
    
    model = model[2]               # transformermodel, initial model state
    model.load_state_dict(model_state)   # load checkpoint model state
    model.to(device)
    model.eval() 
    
    return model, config


"""""""""""""""""""""""""""""""""""
Evaluation methods
"""""""""""""""""""""""""""""""""""


def test_model(eval_pos, method_list, kwargs_dict):
    """
    Evaluate different models after eval_pos
    
    :param eval_pos: the number of stocks to evaluate on a date. can be any length, no need to have seq_len//2
    :param method_list: a list of evaluation methods
    :param kwargs_dict: config dict for evaluation, from prior config
    :return: a dataframe that contain asset id, date, target, predicted probability, method, for the entire evaluation period
    """
    
    data = test_set.copy()

    dates = data["date"].drop_duplicates().sort_values().tolist()  # all dates available
    dates_pair = get_dates_pair(dates, date_style=kwargs_dict["date_style"])  # date_style in testing should match training

    """
    Preprocess test data
    """
    # fill a number of features with zeros if not enough features (zero padding)
    add_feats = kwargs_dict["num_features"] - (data.shape[1] - 3)
    for feat in range(add_feats):  # if add_feats < 0, do not add columns
        data["addfeat%s"%feat] = 0
    featnames = data.drop(columns=["date", "id", "target"]).columns.values#.tolist()

    # select features
    if 'select_features' not in kwargs_dict:
        sample_feats = featnames[:kwargs_dict["num_features"]]  # first n features, fixed features
    else:
        sample_feats = kwargs_dict["select_features"]  # given features
    
    result_df = pd.DataFrame()
    
    for selected_dates in dates_pair:  # one date
        print(selected_dates)
        data_date = data[data["date"].isin(selected_dates)]  # one date pair's data

        # get stocks to sample from
        stocks = data_date.groupby("id").apply(len)
        stocks = stocks[stocks > 1].index  # only select stock with two dates data available
        print(len(stocks))

        # slice stocks' data
        data_date = data_date.set_index("id").loc[stocks]
        
        # convert target (returns) into quantiles by date
        # data_date["target"] = pd.qcut(x=data_date["target"], q=kwargs_dict["multiclass"], labels=range(kwargs_dict["multiclass"]))
        df = []                                                                   # stores daily ranks
        for _, sdf in data_date.groupby('date'):                                             # group predictions each date
            sdf['target']  = pd.qcut(x=sdf["target"], q=kwargs_dict["multiclass"], duplicates='drop', labels=False)  # create bins
            df          += [sdf]                                                   # append to list
        data_date = pd.concat(df, axis=0)
        _, bins = pd.qcut(x=data_date[data_date.date==dates_pair[0]]["target"], q=config["multiclass"], duplicates='drop', labels=False, retbins=True)
        data_date["target"] = pd.cut(x=data_date["target"], bins=bins, include_lowest=True, labels=False)
        
        # have to make sure all samples are included
        # stocks splitted into subsets of max length of eval_pos: ex. [99, 98, 98]
        split_stks = np.array_split(stocks, np.ceil(len(stocks)/eval_pos))

        for sample_stks in split_stks:  # one batch
            sample_stks = random.choices(sample_stks, k=eval_pos)  # with replacement, same with a seed

            # sort so that the first half is train, the later half is test during bptt split
            # stk orders doesn't matter
            test_X_Y = data_date.loc[sample_stks].reset_index().sort_values(["date","id"])  # slice index

            # sample features, get target
            test_X = list(test_X_Y[sample_feats].values)  # slice columns, tolist() too slow
            test_Y = list(test_X_Y["target"].values)
            
            # only one datapoint for each batch: shape=(seq_len,1,feats), (seq_len,1)
            test_X = torch.tensor(test_X, device="cpu", dtype=torch.float32).unsqueeze(1)
            test_Y = torch.tensor(test_Y, device="cpu", dtype=torch.float32).unsqueeze(1)
            
            # evaluation: FinPFN
            for method in method_list:
                # kwargs_dict["models"]={method1: xxx.ckpt}
                modelname = kwargs_dict["models"][method] if "models" in kwargs_dict and method in kwargs_dict["models"] else None
                result_X_Y = evaluate_method(eval_pos, test_X, test_Y, test_X_Y, method, 
                                             multiclass=kwargs_dict["multiclass"],
                                             modelname=modelname)
                result_df = pd.concat([result_df, result_X_Y])
    
    # so that repeated samples from replacement sampling will be deleted
    result_df.drop_duplicates(subset=["id", "date", "method"], inplace=True)
            
    return result_df


def evaluate_method(eval_pos, test_X, test_Y, test_X_Y, method, multiclass, modelname=None):
    """
    Evaluate different transformer models after eval_pos
    
    :param test_X: new data features torch.tensor(seq_len,batch_size,feats)
    :param test_Y: new data targets torch.tensor(seq_len,batch_size)
    :param test_X_Y: a dataframe contains two dates' preprocessed data, before eval_pos is train and after is test
    :param eval_pos: evaluation position = seq_len//2
    :param multiclass: if=2:binary, else:multiclass
    :param method: model method
    :param modelname: a model location
    :return: a dataframe that contain asset id, date, target, predicted probability, method, for one method and one batch
    """
    classes = range(multiclass)  # intergers

    if method == "FinPFN" and multiclass == 2:
        model, _ = get_model(f'model/financial_binary_models_transformer_features_checkpoint_epochs_30_lr_5e-06.cpkt')
    elif method == "FinPFN" and multiclass == 10:
        # financial_multi10_models_transformer_features_checkpoint_epochs_60_lr_3e-05.cpkt
        model, _ = get_model(f'model/financial_multi10_models_transformer_features_checkpoint_epochs_60_lr_3e-05.cpkt')
    else:  # multiclass of other classes, or different gap, or FinPFN-rand
        if not modelname:
            print(f"Please give a model location dictionary for {method}-{multiclass}classes in config.")
            return
        model, _ = get_model(modelname)
        
    pred, _ = eval_transformer(X=test_X, y=test_Y, model=model, eval_pos=eval_pos, 
                               multiclass=multiclass, verbose=False)
    
    # test_X_Y.index=stock to None -> slice later eval_pos values -> reset index to 0:eval_pos
    result_X_Y = test_X_Y.reset_index().loc[eval_pos:, ["id", "date", "target"]].reset_index(drop=True)
    
    if multiclass == 2:  # store only class 1 probability
        result_X_Y["pred_prob"] = pred
        
    else:            # store all classes probabilities, squeeze 2nd dimension: batch_size=1
        print(pred.shape, classes)
        if len(pred.shape) > 2:  
            pred = pred.squeeze(1) 
        result_X_Y[["pred_prob_%s"%(int(i)) for i in classes]] = pd.DataFrame(pred)
    
    result_X_Y["method"] = method
    result_X_Y = result_X_Y.drop_duplicates()  # .reset_index()
    
    return result_X_Y


def eval_transformer(X, y, model, eval_pos, multiclass=2, verbose=True):
    """
    Evaluate tranformer model after eval_pos, passed in 3D tensor, output 3D array
    
    :param X: new data features (seq_len,batch_size,feats)
    :param y: new data targets (seq_len,batch_size)
    :param eval_pos: evaluation position = seq_len//2
    :param multiclass: if=2:binary, else:multiclass
    :param verbose: whether to show model structure
    :return: predicted probability, a roc auc score numpy array
    """
    print(">> Evaluating transformer model performance...\n")
    if verbose:
        print(model)
    
    pred = model((X, y[:eval_pos].float()), single_eval_pos=eval_pos)  # pred shape=(seq_len,batch_size,num_classes)
    # must have 10 classes as output, even 9 classes in the data
    
    if multiclass == 2:
        # for binary classification: transform to range 0 to 1, output shape=(seq_len,batch_size)
        output = torch.sigmoid(pred).squeeze(-1)  # squeeze dimension 3: class=1
        outputs = output.detach().cpu().numpy()

        # get one auc_roc for each batch (batch_size)
        auc = np.array([roc_auc_score(y[eval_pos:, i].cpu(), outputs[:, i]) for i in range(X.shape[1])])
    
    else:
        # for multiclass classification: 
        m = torch.nn.Softmax(dim=-1)  # sum of probability of 10 classes = 1
        output = m(pred)           # transform to range 0 to 1, output shape=(seq_len,batch_size,num_classes)
        outputs = output.detach().cpu().numpy()

        # get one auc_roc for each batch (batch_size), y.shape=(seq_len) vs. output.shape=(seq_len,num_classes)
        try:  
            auc = np.array([roc_auc_score(y[eval_pos:, i].cpu(), outputs[:, i], multi_class="ovo") for i in range(X.shape[1])])
        except:  # maybe y[eval_pos:] does not contain all classes, produce error
            auc = [0]

    return outputs, auc

    
"""
Can add other baseline methods.
"""

## Random Forest
param_grid = {}
param_grid['random_forest'] = {
    'n_estimators': [500],  # fixed, large enough
    # depends on number of features
    'max_features': ['sqrt'],  # select a subset of features to split: sqrt similar to log2(), or 25. 
    # depends on number of samples
    'max_depth': np.arange(2, 9, 1),  # tree depth: 2^8=256 nodes
    'min_samples_split': [0.01, 0.02, 0.05, 0.1]  # fractions are better: 600*0.01=6, 600*0.1=60
}


def random_forest_metric(train_x, train_y, test_x, test_y, clf=None, multiclass=2):
    """
    Evaluate random forest after eval_pos
    :params train_x, train_y: numpy before fitting classifier
    :params clf: if not pass in trained RF classifier, train on latest data
    """
    
    classes = np.unique(train_y).astype(int).tolist()  # can < 10
    print(">>> Evaluating Random Forest model performance...\n", classes)
    
    if not clf:
        # can only fit two dimension
#         train_x, train_y, test_x, test_y = train_x.squeeze(1), train_y.squeeze(1), test_x.squeeze(1), test_y.squeeze(1)
        # tensor to numpy
#         train_x, train_y, test_x, test_y = train_x.numpy(), train_y.numpy().astype(int), \
#                                 test_x.numpy(), test_y.numpy().astype(int)

        clf = RandomForestClassifier()
        # use gridsearch to test all values
        clf = GridSearchCV(clf, param_grid['random_forest'], cv=5, n_jobs=-1, verbose=2)  # cv: can try time series split
        # fit model to data
        clf.fit(train_x, train_y)
        print(clf.best_params_)
    
    if multiclass > 2:  # multiclass
        pred = clf.predict_proba(test_x)  # can have dim < 10
        pred_ = np.zeros((pred.shape[0], multiclass))  # right dimension
        pred_[:, classes] = pred  # insert column classes
        pred_class = np.argmax(pred, axis=-1)  # max probability class
        accuracy = (pred_class == test_y).astype(float).mean()
        try:
            auc = roc_auc_score(test_y, pred_, multi_class="ovo") 
        except:
            auc = 0
        
    else:  # binary
        pred = clf.predict_proba(test_x)[:, 1]  # probability of 1
        accuracy = ((pred > 0.5) == test_y).astype(float).mean()
        auc = roc_auc_score(test_y, pred) 

    return pred, accuracy, auc, classes


## TabPFN
def eval_tabpfn(X, y, eval_pos, multiclass=2, device="cuda:0"):
    """
    Evaluate TabPFN model after eval_pos
    TabPFN: https://github.com/automl/TabPFN
    """
    print(">>> Evaluating TabPFN model performance...\n")

    X, y = X.squeeze(1), y.squeeze(1)  # can only fit two dimension

    # model from TabPFN
    classifier_tab = TabPFNClassifier(device=device,
                                      N_ensemble_configurations=3,
                                      no_preprocess_mode=True,
                                      # do not preprocess data as our data is already preprocessed
                                      multiclass_decoder=None,  # cannot shuffle classes, order matters
                                      feature_shift_decoder=False
                                      # feature orders doesn't matter, can set to True (default)
                                      )
    # fit classes from train data, output may be dim < 10 classes
    classifier_tab.fit(X[:eval_pos], y[:eval_pos],
                       overwrite_warning=True)  # only saves the training data self.X=X, self.y=y.

    # including preprocessing of data: probably don't need it?
    if multiclass == 2:  # binary
        prediction_ = classifier_tab.predict_proba(X[eval_pos:])[:, 1]  # return predicted probabilities for class 1
        auc = roc_auc_score(y[eval_pos:], prediction_)

    else:  # multiclass
        prediction_ = classifier_tab.predict_proba(X[eval_pos:])  # return predicted probabilities for each class
        try:
            auc = roc_auc_score(y[eval_pos:], prediction_, multi_class="ovo")
        except:  # maybe y[eval_pos:] does not contain all classes, produce error
            auc = 0

    return prediction_, auc, classifier_tab.classes_.astype(int).tolist()


def get_tabpfn_config():
    # https://github.com/automl/TabPFN/issues/26
     return {'lr': 0.0001,
             'dropout': 0.0,
             'emsize': 512,
             'batch_size': 8,
             'nlayers': 12,
             'num_features': 100,
             'nhead': 4,
             'nhid_factor': 2,
             'bptt': 1024,
             'eval_positions': [972],
             'seq_len_used': 50,
             'sampling': 'mixed',
             'epochs': 400,
             'num_steps': 1024,
             'verbose': False,
             'mix_activations': True,
             'nan_prob_unknown_reason_reason_prior': 1.0,
             'categorical_feature_p': 0.2,
             'nan_prob_no_reason': 0.0,
             'nan_prob_unknown_reason': 0.0,
             'nan_prob_a_reason': 0.0,
             'max_num_classes': 10,
             'num_classes': '<function <lambda>.<locals>.<lambda> at 0x7fc575dfb550>',
             'noise_type': 'Gaussian',
             'balanced': False,
             'normalize_to_ranking': False,
             'set_value_to_nan': 0.1,
             'normalize_by_used_features': True,
             'num_features_used': {'uniform_int_sampler_f(3,max_features)': '<function <lambda>.<locals>.<lambda> at 0x7fc575dfb5e0>'},
             'num_categorical_features_sampler_a': -1.0,
             'differentiable_hyperparameters': {'prior_bag_exp_weights_1': {'distribution': 'uniform',
               'min': 1000000.0,
               'max': 1000001.0},
              'num_layers': {'distribution': 'meta_trunc_norm_log_scaled',
               'max_mean': 6,
               'min_mean': 1,
               'round': True,
               'lower_bound': 2},
              'prior_mlp_hidden_dim': {'distribution': 'meta_trunc_norm_log_scaled',
               'max_mean': 130,
               'min_mean': 5,
               'round': True,
               'lower_bound': 4},
              'prior_mlp_dropout_prob': {'distribution': 'meta_beta',
               'scale': 0.9,
               'min': 0.1,
               'max': 5.0},
              'noise_std': {'distribution': 'meta_trunc_norm_log_scaled',
               'max_mean': 0.3,
               'min_mean': 0.0001,
               'round': False,
               'lower_bound': 0.0},
              'init_std': {'distribution': 'meta_trunc_norm_log_scaled',
               'max_mean': 10.0,
               'min_mean': 0.01,
               'round': False,
               'lower_bound': 0.0},
              'num_causes': {'distribution': 'meta_trunc_norm_log_scaled',
               'max_mean': 12,
               'min_mean': 1,
               'round': True,
               'lower_bound': 1},
              'is_causal': {'distribution': 'meta_choice', 'choice_values': [True, False]},
              'pre_sample_weights': {'distribution': 'meta_choice',
               'choice_values': [True, False]},
              'y_is_effect': {'distribution': 'meta_choice',
               'choice_values': [True, False]},
              'prior_mlp_activations': {'distribution': 'meta_choice_mixed',
               'choice_values': ["<class 'torch.nn.modules.activation.Tanh'>",
                "<class 'torch.nn.modules.linear.Identity'>",
                '<function get_diff_causal.<locals>.<lambda> at 0x7fc575dfb670>',
                "<class 'torch.nn.modules.activation.ELU'>"]},
              'block_wise_dropout': {'distribution': 'meta_choice',
               'choice_values': [True, False]},
              'sort_features': {'distribution': 'meta_choice',
               'choice_values': [True, False]},
              'in_clique': {'distribution': 'meta_choice', 'choice_values': [True, False]},
              'sampling': {'distribution': 'meta_choice',
               'choice_values': ['normal', 'mixed']},
              'pre_sample_causes': {'distribution': 'meta_choice',
               'choice_values': [True, False]},
              'outputscale': {'distribution': 'meta_trunc_norm_log_scaled',
               'max_mean': 10.0,
               'min_mean': 1e-05,
               'round': False,
               'lower_bound': 0},
              'lengthscale': {'distribution': 'meta_trunc_norm_log_scaled',
               'max_mean': 10.0,
               'min_mean': 1e-05,
               'round': False,
               'lower_bound': 0},
              'noise': {'distribution': 'meta_choice',
               'choice_values': [1e-05, 0.0001, 0.01]},
              'multiclass_type': {'distribution': 'meta_choice',
               'choice_values': ['value', 'rank']}},
             'prior_type': 'prior_bag',
             'differentiable': True,
             'flexible': True,
             'aggregate_k_gradients': 8,
             'recompute_attn': True,
             'bptt_extra_samples': None,
             'dynamic_batch_size': False,
             'multiclass_loss_type': 'nono',
             'output_multiclass_ordered_p': 0.0,
             'normalize_with_sqrt': False,
             'new_mlp_per_example': True,
             'prior_mlp_scale_weights_sqrt': True,
             'batch_size_per_gp_sample': None,
             'normalize_ignore_label_too': True,
             'differentiable_hps_as_style': False,
             'max_eval_pos': 1000,
             'random_feature_rotation': True,
             'rotate_normalized_labels': True,
             'canonical_y_encoder': False,
             'total_available_time_in_s': None,
             'train_mixed_precision': True,
             'efficient_eval_masking': True,
             'multiclass_type': 'rank',
             'done_part_in_training': 0.8425
             }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='2021-01-01', type=str)

    args = parser.parse_args()
    
    config = get_default_config()
    prior_config = get_prior_config(config)

    train_model(config, prior_config, save=True, device='cuda:0')



