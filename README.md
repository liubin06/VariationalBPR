# Variational-BPR
This paper introduces latent user-specific interest prototypes and constructs a variational lower bound of the Bayesian Personalized Ranking (BPR) as an alternative optimization objective. Correspondingly, we propose a generalized pairwise loss, termed Variational BPR, to address the challenges of calculating and optimizing the likelihood function caused by implicit feedback. The user interest prototypes are obtained through an attention function, enabling more robust pairwise comparisons to mitigate the impact of noise. Additionally, Variational BPR offers a flexible approach for mining hard samples and contributes to achieving uniform feature distribution.

<p align='left'>
<img src='https://github.com/liubin06/Variational-BPR/blob/main/bound.png' width='600'/>
</p>

## Prerequisites
- Python 3.7 
- PyTorch 1.12.1

## Flags 

`--loss`: loss function, choose from ['VBPR' , 'BPR']

`--dataset`: dataset name, choose from ['100k', '1M', 'gowalla' or 'yelp2018'].

`--backbone`: Backbone model to encoder feature representations, choose from ['MF', 'LightGCN'].

*`(VBPR-specific Hyperparameters)`*

`--M`: Number of positive samples.

`--N`: Number of negative samples.

`--cpos`: Positive scalling factor.

`--cneg`: Negitive scalling factor




## Model Pretraining

For each dataset, the backbone model hyperparameters for BPR and VBPR are fixed the same. For instance, run the following command to train an embedding on different datasets.


```
python main.py  --loss 'VBPR'  --dataset_name '100k'  --encoder 'MF' --M 2 --N 4  --cpos 10 --cneg 0.5 --weight_decay 1e-05 --feature_dim 64
```
```
python main.py  --loss 'VBPR'  --dataset_name '100k'  --encoder 'LightGCN'  --M 2 --N 4  --cpos 10 --cneg 0.5 --feature_dim 64 --weight_decay 1e-05 --hop 1
```


## Public Model Parameter Settings
| Dataset  | Backbone | Feature dim | Learning Rate | l2 | Batch Size  | Hop  | 
|---------|:--------------:|:--------------:|:----:|:-----:|:---:|:---:|
| MovieLens 100K  |     MF        |       64       | 1e-3 |  1e-5  | 1024  |  -|
| MovieLens 1M  |     MF        |        64        | 1e-3 |  1e-6  | 1024  |  -|
| Gowalla |     MF        |        128       | 1e-3 |  1e-6  | 1024  |  -|
| Yelp2018  |     MF        |       128      | 1e-3 |  1e-6  | 1024  |  -|
| MovieLens 100K |    LightGCN        |        64        | 1e-3 | 1e-5  | 1024  | 1|
| MovieLens 1M |     LightGCN        |        64        | 1e-3 |  1e-6  | 1024  |  2|
| Gowalla |     LightGCN        |        128        | 1e-3 |  1e-7  | 1024  |  1 |
| Yelp2018  |     LightGCN        |        128        | 1e-3 |  1e-7  | 1024  |  2|

## VBPR-specific  Parameter Settings
| Dataset  | Backbone | $M$ | $N$ | $c_\text{pos}$ | $c_\text{neg}$  | 
|---------|:--------------:|:--------------:|:----:|:-----:|:---:|
| MovieLens 100K  |     MF        |      2       | 4 |  10  | 0.5 |  
| MovieLens 1M  |     MF        |       2       | 4 |  10 | 0.5 |   
| Gowalla  |     MF        |         2    | 15 | 10 | 0.1 | 
| Yelp2018  |     MF        |        2     | 15  |10  | 0.1 | 
| MovieLens 100K |    LightGCN        |      2       | 5 |10  | 0.1 | 
| MovieLens 1M  |     LightGCN        |       2      | 6 | 10 |0.1  | 
| Gowalla  |     LightGCN       |         2    | 20 | 10 | 0.1 | 
| Yelp2018   |     LightGCN        |        2     |20  |10  |0.1  | 

