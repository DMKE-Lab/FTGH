# FTGH
Few-Shot Temporal Knowledge Graph Completion by Embedding enhancement with gated mechanism and hierarchical attention

This repository contains the implementation of the FTGH architectures described in the paper.
# Installation
 Install Pytorch (>= 1.1.0)
 ```
pip install pytorch
 ```
 Python 3.9
  ```
pip install python 3.9
  ```
 Numpy
  ```
pip install numpy
  ```
 Pandas
  ```
pip install pandas
  ```
 tqdm
  ```
pip install tqdm
  ```
# How to use
run the code:
```
python train.py --parameters
```

# Parameters setting

1. The embedding dimension of the two datasets is uniformly set to 100.
2. A single entity's maximum number of local neighbors is set to 50 for two datasets.
3. The number of layers of Bi-LSTM is selected to be 2. 
4. For the two dataset, we set the dimensions of input and hidden state of Bi-LSTM to 200 and 100, respectively.
5. In the process of updating model parameters, we choose Adam optimizer.
6. The Adam optimizer with an initial learning rate of 0.001 is used to update the model parameters. 
7. The initial learning rate is 0.001, and the dropout rate is 0.5. The margin is set to 1.0. 
8. We apply mini-batch gradient descent during training to update the parameters. The batch size is 128.

# Dateprocess

To run our code, we need to divide the data set according to the data set partition file first, or divide it according to our own needs. If we want to get the best results, we need to use Complex to pre-train and then embed it into the model.

| Baselines   | Code                                                         |
| ----------- | ------------------------------------------------------------ |
| TransE      | [Link](https://github.com/jimmywangheng/knowledge_representation_pytorch) |
| TTransE     | [link](https://github.com/INK-USC/RE-Net)                    |
| DistMult    | [link](https://github.com/BorealisAI/DE-SimplE)              |
| TA-DistMult | [link](https://github.com/INK-USC/RE-Net)                    |
| TA-TransE   | [link](https://github.com/INK-USC/RE-Net)                    |
| Gamtching   | [link](https://github.com/xwhan/One-shot-Relational-Learning) |
| MateR       | [link](https://github.com/AnselCmy/MetaR)                    |
| FSRL        | [link](https://github.com/chuxuzhang/AAAI2020_FSRL)          |
| FAAN        | [link](https://github.com/JiaweiSheng/FAAN)                  |
| TFSC        | [link](https://github.com/DMKE-Lab/TFSC)                     |
| TransAM     | [link](https://github.com/gawainx/TransAM.)                  |
| NP-FKGC     | [link](https://github.com/RManLuo/NP-FKGC)                   |

TTransE, TA-Distmult, and TA-TransE have been implemented in baselines in [Re-Net]((https://github.com/INK-USC/RE-Net/tree/master/baselines)). The user can run the baselines by the following command.

```
CUDA_VISIBLE_DEVICES=0 python3 TTransE.py -f 1 -d ICEWS18 -L 1 -bs 1024 -n 1000`
```

We have implemented DistMult refer to [RotatE](: https://github.com/DeepGraphLearning/ KnowledgeGraphEmbedding.).

```
cd ./baselines
bash run.sh train MODEL_NAME DATA_NAME 0 0 512 1024 512 200.0 0.0005 10000 8 0
```

# Output

The metrics used for evaluation are Hits@{1,3,5} and MRR. The results show that our model works better than most models.



