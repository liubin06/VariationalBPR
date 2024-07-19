# Variational-BPR

## Flags 

`--loss`: loss function, choose VBPR or BPR.

*`(Common Model Hyperparameters)`*

`--dataset`: dataset name, choose 100k, 1M, gowalla or yelp2018.

`--backbone`: Backbone model to encoder feature representations.

`--batch_size`: Batch size in each mini-batch.

`--feature_dim`: Feature dim for latent vector.

`--learning_rate`: Learning rate for model training.

`--weight_decay` : l2 regulation constant.

*`(VBPR-specific Hyperparameters)`*

`--M`: Number of positive samples.

`--N`: Number of negative samples.

`--cpos`: Positive scalling factor.

`--cneg`: Negitive scalling factor




## Model Pretraining
For instance, run the following command to train an embedding on different datasets.
```
python main.py  --loss VBPR  --dataset_name '100k'  --encoder MF  --M 2 --N 4  --cpos 10 --cneg 0.5
```


## Public Parameter Settings
| Dataset  | Backbone | Feature dim | Learning Rate | l2 | Batch Size  | 
|---------|:--------------:|:--------------:|:----:|:-----:|:---:|
| MovieLens 100K  |     MF        |       64       | 1e-3 |  1e-6  | 1024  |  
| MovieLens 1M  |     MF        |        64        | 1e-3 |  1e-6  | 1024  |   
| Yelp2018  |     MF        |       -      | - |  -  | -  |  
| Gowalla |     MF        |        -        | - |  -  | -  |   

## VBPR-specific  Parameter Settings
| Dataset  | Backbone | M | N | $c_\text{pos}$ | $c_\text{neg}$  | 
|---------|:--------------:|:--------------:|:----:|:-----:|:---:|
| MovieLens 100K  |     MF        |      2       | 4 |  10  | 0.5 |  
| MovieLens 1M  |     MF        |       -       | - |  - | - |   

