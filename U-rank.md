# U-rank

An easy implementation of algorithms of  U-rank, including U-rank, U-rank+$_{lambda}$, U-rank+$_{sinkhorn}$ and Tree-based U-rank

## 1. Requirements

* python 3.6

* Tensorflow 1.9.0

* lightdbm 2.3.2

* scikit-learn 0.21.2 

* networkx 2.3

* scipy 1.3.1

* some other basic packages

  

## 2. Usage

Here we sample some queries from MSLR10K dataset and generate a example dataset in folder `./data/MSLR10K_small/`. There also are some example models in `./result/MSLR10K_train/` for demonstration(Note that they are not well trained or tuned).

### 2.1. Train

#### 2.1.1 Bias model

```bash
python main.py -train_stage=bias -use_debias=True 
```

#### 2.1.2 Click model

##### Click model without debias:

```bash
python main.py -train_stage=click -use_debias=False
```

##### Debias click model(need a bias model for initialization):

`use_debias` indicates whether model use debias method and  `bias_model_path` is the path where the bias model lies in. Here we give an example `'./result/MSLR10K_train/bias/bias_model_example'` for bias model

```bash
python main.py -train_stage=click -use_debias=True -bias_model_path='./result/MSLR10K_train/bias/bias_model_example/'
```

#### 2.1.3 Ranker model

All the ranker models need a a click model for initialization. `click_model_path` is the path where the click model lies in. We give a debias click model example in `./result/MSLR10K_train/click/debias_click_model_example` and a without debias click model in `./result/MSLR10K_train/click/click_model_example`

##### Lambdaloss ranker model without debias (need a click model for initialization)

```bash
python main.py -train_stage=ranker -use_debias=False -click_model_path='./result/MSLR10K_train/click/click_model_example'
```

##### Debias Lambdaloss ranker model (need a debias click model for initialization)

```bash
python main.py -train_stage=ranker -use_debias=True -click_model_path='./result/MSLR10K_train/click/debias_click_model_example'
```

##### Other ranker model

Similar to lambdaloss ranker, sinkhorn  and tree rankers also need `use_debias` and `click_model_path` .

For debias **sinkhorn** ranker:

```bash
python main_sinkhorn.py -train_stage=ranker -use_debias=True -click_model_path='./result/MSLR10K_train/click/debias_click_model_example'
```

For **tree** ranker without debias:

```bash
python main_tree.py -train_stage=ranker -use_debias=False -click_model_path='./result/MSLR10K_train/click/click_model_example'
```

### 2.2 Test

we use `-decode` to control the evaluation and model in `restore_model_path` will be evaluated.

##### For debias lambdaloss ranker:

```bash
python main.py -train_stage=ranker -use_debias=True -restore_model_path='./result/MSLR10K_train/ranker/lambdaloss/debias_lambda_ranker_model_example' -decode
```

##### For debias sinkhorn ranker:

```bash
python main_sinkhorn.py -train_stage=ranker -use_debias=True -restore_model_path='./result/MSLR10K_train/ranker/sinkhorn/debias_sinkhorn_ranker_model_example' -decode
```

##### For debias tree ranker:

```bash
python main_tree.py -train_stage=ranker -use_debias=False -restore_model_path='./result/MSLR10K_train/ranker/tree/tree_ranker_model_example' -decode
```





