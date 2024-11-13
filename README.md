## Official Code for the Paper "Meta-Learning for Cross-Sectional Return Prediction in Financial Markets"

In this paper, we propose the Financial Prior-Data Fitted Network (FinPFN), a meta-learning framework utilizing a Transformer architecture for cross-sectional stock return prediction and classification. 

__Getting Started__

This is a Python project, we used Python 3.9 in development and recommend to use a `virtualenv` or `conda`.
To use our code, clone the project with

```
git clone git@github.com:wangy8989/FinPFN.git
```

install all dependencies with

```
pip install -r requirements.txt
```

Please download the PFN https://github.com/automl/PFNs.git before using our code.
```
git clone https://github.com/automl/PFNs.git
cd PFNs
pip install -e .
```

__Training a model__

[financial_dataloader.py](financial_dataloader.py) and [data_utils.py](data_utils.py) provides the dataloader of financial data prior for the Transformer.

[financial_model_training.py](financial_model_training.py) provides methods to train and evaluate a PFN model with data prior.
```
config =
{'lr': 3e-05,
 'epochs': 60,
 'dropout': 0.0,
 'emsize': 256,
 'batch_size': 64,
 'nlayers': 5,
 'num_outputs': 10,
 'num_features': 30,
 'steps_per_epoch': 100,
 'nhead': 4,
 'seq_len': 100,
 'nhid_factor': 2,
 'validation_period': 1}
prior_config =
{'date_style': 'consecutive',
 'sample_replacement': False,
 'multiclass': 10,
 'num_outputs': 10,
 'num_features': 30,
 'fuse_x_y': False,
 'device': 'cpu'}
model = train_model(config, prior_config, save=True)
```

__Evaluating Models__

[Financial_Data_Model_Training.ipynb](Financial_Data_Model_Training.ipynb) provides a workflow to evaluate baselines and the transformer.

```
method_list = ["FinPFN"]
eval_pos = 50  #half of sequence length=100
output_df = test_model(eval_pos, method_list, prior_config)
```

__Cite__

FinPFN is introduced in
```

```

PFNs were introduced in
```
@inproceedings{
    muller2022transformers,
    title={Transformers Can Do Bayesian Inference},
    author={Samuel M{\"u}ller and Noah Hollmann and Sebastian Pineda Arango and Josif Grabocka and Frank Hutter},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=KSugKcbNf9}
}
```
