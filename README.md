## Official Code for the Paper "Meta-Learning for Cross-Sectional Return Prediction in Financial Markets"

In this paper, we propose the Financial Prior-Data Fitted Network (FinPFN), a meta-learning framework utilizing a Transformer architecture for cross-sectional stock return prediction. 

__Getting Started__

This is a Python project, we used `Python 3.10` in development and recommend to use a `virtualenv` or `conda`.
To use our code, clone the project with

```
git clone git@github.com:wangy8989/FinPFN.git
```

install all dependencies with

```
pip install -r requirements.txt
```

The [TabPFN](https://github.com/PriorLabs/TabPFN) version we used is `2.0.8`.

The code is forked from [finetune_tabpfn_v2](https://github.com/LennartPurucker/finetune_tabpfn_v2).


__Training a model__

[data_utils.py](scripts/training_utils/data_utils.py) provides the dataloader of financial data prior for the model.

[main.py](scripts/main.py) provides methods to finetune a TabPFN model.


__Evaluating Models__

[finpfn.ipynb](finpfn.ipynb) provides a workflow to train and evaluate models.


__Downloading Data__

[Data link](https://1drv.ms/f/c/ada3d0b0856299f2/IgA5rSYiS684Rpoo5u1S8AcfAYGx-OuQw3YiyGb4ePJcHBA?e=GdgK1i) with Password: finpfn12345


__Cite__

When using, please cite [FinPFN](https://authors.elsevier.com/c/1mCkn4xF2VsT1v) <-- free link for 50 days
```
@article{wang2025finpfn,
title = {Meta-learning for return prediction in shifting market regimes},
journal = {Journal of Financial Markets},
pages = {101042},
year = {2025},
issn = {1386-4181},
doi = {https://doi.org/10.1016/j.finmar.2025.101042},
url = {https://www.sciencedirect.com/science/article/pii/S1386418125000825},
author = {Yicheng Wang and Sandro Claudio Lera},
keywords = {Financial machine learning, Return prediction, Regime shifts, Meta-learning},
abstract = {We propose a meta-learning framework for cross-sectional return prediction that adapts to regime-dependent dynamics. Instead of learning a fixed mapping from features to returns, we condition our model forecasts on recent feature-return relationships. This allows it to adjust to evolving market states without explicit regime labels or frequent re-estimation. We implement the framework with a Transformer-based Bayesian predictor, the Financial Prior-data Fitted Network (FinPFN), and evaluate it on daily Chinese A-shares and monthly U.S. equities. During regime changes, proxied by large volatility shifts, our method significantly outperforms benchmarks, offering a practical tool for dynamic return prediction.}
}
```

TabPFNs were from
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
