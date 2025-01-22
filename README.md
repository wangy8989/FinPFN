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

The code is forked from [TransformersCanDoBayesianInference](https://github.com/automl/TransformersCanDoBayesianInference).

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

When using, please cite [FinPFN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5022829)
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

TabPFNs were introduced in 
```
@article{hollmann2025tabpfn,
 title={Accurate predictions on small data with a tabular foundation model},
 author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and
         Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and
         Schirrmeister, Robin Tibor and Hutter, Frank},
 journal={Nature},
 year={2025},
 month={01},
 day={09},
 doi={10.1038/s41586-024-08328-6},
 publisher={Springer Nature},
 url={https://www.nature.com/articles/s41586-024-08328-6},
}
```
