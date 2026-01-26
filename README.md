![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

# Variational Bayesian Personalized Ranking


🎯 
VarBPR is a unified variational framework that brings **exposure controllability** and **theoretical interpretability** implicit-feedback pairwise learning into principled variational inference, enabling:

- **Unified Noise & Bias Handling** — Seamlessly integrate denoising, popularity debiasing, and preference alignment in a unified variational framework.  
- **Endogenous Exposure Control** — Shape exposure profiles directly via variational inference procedure instead of post-hoc adjusting.  
- **Linear-Time Scalability** — Maintain BPR's efficiency while adding expressive control mechanisms.  
- **Theoretical Guarantees** — With generalization insight from an opportunity-cost perspective for designing controllable recommender systems.

Whether you're pushing the boundaries of recommendation research or building real-world systems with fairness requirements, VarBPR offers both **performance** and **policy control** in one elegant package.

📖 **Paper**: Link to paper [Variational Bayesian Personalized Ranking ](https://arxiv.org/pdf/2503.11067v2)
<p align='center'>
<img src='https://github.com/liubin06/Variational-BPR/blob/main/bound.png' width='700'/>
</p>

🛠️ **Features**:
- Plug-and-play replacement for standard BPR loss
- Support for MF, LightGCN, XSimGCL backbones
- Dual implementation: efficient plug-in & exact ELBO versions
- Full reproducibility with open datasets
- Practical generalization guarantees and controllable exposure policies.

🚀 **Get started in minutes** — Transform your recommender from a black box into a controllable, interpretable system.

### Major Changes

- **Introduce a designable prior** `(π⁺, π⁻)` that encodes customizable exposure objectives—such as promoting long‑tail items, ensuring content quality, or enhancing diversity—into the variational inference framework.
- **Explicitly separate and implement** the two-stage training procedure: **variational inference** (solving for posteriors) and **variational learning** (updating model parameters).
- **Update license** from MIT to Apache License 2.0 for broader compatibility and explicit patent protection in open‑source and commercial use.
- **Improved evaluation** by incorporating long-tail exposure (APLT) and Top-5 evaluation.
- Refined the multi-threaded data loading pipeline to minimize idle time.




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

`--loss`: loss function, choose from ['VarBPRExact' , 'VarBPRPlugIn']
- `VarBPRExact` This is the VarBPR implementation without plug-in approximation (ELBO), for small M,N, choose VarBPRExact to achieve better performance.
- `VarBPRPlugIn` This is the VarBPR implementation with plug-in approximation, for large M,N, choose VarBPRPlugIn to ensure efficiency.

`--dataset`: dataset name, choose from ['100k', '1M', 'gowalla' or 'yelp2018'].

`--backbone`: Backbone model to encoder feature representations, choose from ['MF', 'LightGCN'].

*`(VarBPR-specific Hyperparameters)`*

`--M`: Number of positive samples.

`--N`: Number of negative samples.

`--cpos`: Positive scalling factor (for positve side exposure control)

`--cneg`: Negitive scalling factor (for negative side suppression control)


## 5.Model Pretraining

For each dataset, the backbone model hyperparameters for VarBPR are fixed the same. For instance, run the following command to train an embedding on different datasets.


```
python main.py  --loss 'VarBPRExact'  
```



### 5.1 Public Model Parameter Settings
| Dataset  | Backbone | Feature dim | Learning Rate |  l2  | Batch Size  | Hop  | 
|---------|:--------------:|:--------------:|:----:|:----:|:---:|:---:|
| MovieLens 100K  |     MF        |       64       | 1e-3 | 1e-5 | 1024  |  -|
| MovieLens 1M  |     MF        |        64        | 1e-3 | 1e-6 | 1024  |  -|
| Gowalla |     MF        |        128       | 1e-3 | 1e-6 | 1024  |  -|
| Yelp2018  |     MF        |       128      | 1e-3 | 1e-6 | 1024  |  -|
| MovieLens 100K |    LightGCN        |        64        | 1e-3 | 1e-6 | 1024  | 1|
| MovieLens 1M |     LightGCN        |        64        | 1e-3 | 1e-6 | 1024  |  2|
| Gowalla |     LightGCN        |        128        | 1e-3 | 5e-7 | 1024  |  1 |
| Yelp2018  |     LightGCN        |        128        | 1e-3 | 1e-7 | 1024  |  2|

### 5.2 VarBPR-specific  Parameter Settings
| Dataset  | Backbone | $M$ | $N$ | $c_\text{pos}$ | $c_\text{neg}$ | 
|---------|:--------------:|:--------------:|:---:|:--------------:|:--------------:|
| MovieLens 100K  |     MF        |      2       |  4  |       4        |       4        |  
| MovieLens 1M  |     MF        |       2       |  4  |       4        |       4        |   
| Gowalla  |     MF        |         2    | 16  |      8       |     8       | 
| Yelp2018  |     MF        |        2     | 16  |       8       |      8       | 
| MovieLens 100K |    LightGCN        |      2       |  4  |       4        |       4        | 
| MovieLens 1M  |     LightGCN        |       2      |  4  |       4        |       4        | 
| Gowalla  |     LightGCN       |         2    | 20  |       8       |      8       | 
| Yelp2018   |     LightGCN        |        2     | 20  |       8       |      8       | 

### 5.3 Prior Coding
| Component        | pos_rarity $\lambda_1^+$ | pos_quality $\lambda_2^+$ | pos_hardnees $\lambda_3^+$ | neg_popularity $\lambda_1^-$ | neg_badquality $\lambda_2^-$ | neg_hardnees $\lambda_3^-$ | 
|------------------|:------------------------:|:-------------------------:|:--------------------------:|:----------------------------:|:----------------------------:|:--------------------------:|
| MovieLens100k/1M |            0             |             1             |             0              |              0               |             0.5              |            0.5             |
| Yelp2018/Gowalla |            0             |             0             |             1            |              0               |             0             |            1             |
## 6. Customization Guide
To train on a private dataset or use a customized encoder, follow these steps:

1. **Data Formatting**:
   - Organize the data into (u, i) tuples, as shown in the example data in the [data](data) directory.
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
This project is licensed under the Apache License 2.0. - see the [LICENSE](LICENSE) file for details.

## 8. Acknowledgements
We are profoundly grateful to the anonymous reviewers of IEEE TPAMI for their exceptional efforts in elevating the conceptual foundation and theoretical depth of the Variational BPR framework. 
