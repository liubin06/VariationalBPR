# Variational-BPR

## 1. Introduction
Variational BPR is a novel and easily implementable learning objective that integrates key components for enhancing collaborative filtering: likelihood optimization, noise reduction, and popularity debiasing. Our approach involves decomposing the pairwise loss under the ELBO-KL framework and deriving its variational lower bound to establish a manageable learning objective for approximate inference.  Within this bound, we introduce an attention-based latent interest prototype contrastive mechanism, replacing instance-level contrastive learning,  to effectively reduce noise from problematic samples. The process of deriving interest prototypes implicitly incorporates a flexible hard sample mining strategy, capable of simultaneously identifying hard positive and hard negative samples. Furthermore, we demonstrate that this hard sample mining strategy promotes feature distribution uniformity, thereby alleviating popularity bias. 

<p align='center'>
<img src='https://github.com/liubin06/Variational-BPR/blob/main/bound.png' width='700'/>
</p>

## 2. Prerequisites
- Python 3.7 
- PyTorch 1.12.1
  
## 3. Code Architecture
| File | Description |
|------|-------------|
| `utils.py` | Data loading utilities and GPU-optimized dataset organization |
| `model.py` | Backbone architecture implementation |
| `evaluation.py` | Top-k performance evaluation on validation set |
| `main.py` | Central workflow controller (data processing, training, evaluation) |

## 4. Configuration Flags

`--loss`: loss function, choose from ['VBPR' , 'BPR']

`--dataset`: dataset name, choose from ['100k', '1M', 'gowalla' or 'yelp2018'].

`--backbone`: Backbone model to encoder feature representations, choose from ['MF', 'LightGCN'].

*`(VBPR-specific Hyperparameters)`*

`--M`: Number of positive samples.

`--N`: Number of negative samples.

`--cpos`: Positive scalling factor.

`--cneg`: Negitive scalling factor


## 5.Model Pretraining

For each dataset, the backbone model hyperparameters for BPR and VBPR are fixed the same. For instance, run the following command to train an embedding on different datasets.


```
python main.py  --loss 'VBPR'  --dataset_name '100k'  --encoder 'MF' --M 2 --N 4  --cpos 10 --cneg 0.5 --weight_decay 1e-05 --feature_dim 64
```
```
python main.py  --loss 'VBPR'  --dataset_name '100k'  --encoder 'LightGCN'  --M 2 --N 4  --cpos 10 --cneg 0.5 --feature_dim 64 --weight_decay 1e-05 --hop 1
```


### 5.1 Public Model Parameter Settings
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

### 5.2 VBPR-specific  Parameter Settings
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

## 6. Customization Guide
To train on a private dataset or use a customized encoder, follow these steps:

1. **Data Formatting**:
   - Organize the data into (u, i) tuples, as shown in the example data in the "data" directory.
   - Split data as `tran.txt` and `test.txt`


2. **Encoder Configuration**:
   - Rewrite `model.py` to implemente YOUR_OWN_Backbone architecture.
   - Ensure that for any user or item, the encoding results in a $d$ - dimensional feature representation.

2. **Parameter Configuration**:
   ```python
   # In main.py
   parser.add_argument('--dataset',  default='YOUR_DATA_SET', type=str, help='Dataset name')
   parser.add_argument('--backbone', default='YOUR_BACKBONE', type=str, help='Backbone model')
   ```


## 7. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

