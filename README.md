# Improving Semi-Supervised Text Classification with Dual Meta-Learning



## Dataset

we only sample a small part of data for submission, the complete data can be downloaded through the following link：

- AGNews : https://pytorch.org/text/stable/datasets.html#ag-news
- Yelp：https://pytorch.org/text/stable/datasets.html#yelpreviewfull
- Yahoo: https://huggingface.co/datasets/yahoo_answers_qa
- Amazon: http://jmcauley.ucsd.edu/data/amazon/

## Usage

Train the model by 100 labeled data of Yelp dataset:

```
python main.py --dataset Yelp --num_labeled 100 --num_unlabeled 20000 --batch-size 4 --max_len 256 
```

For the number of labeled data is less than 1000, we try to take the learning rate of 1e-5, 2e-5, 5e-5.
For the number of labeled data is greater than or equal to 1000, we try to take the learning rate of 7e-5, 1e-4.
Parallel training is not used since we have better resources.

Monitoring training progress :

```
tensorboard --logdir results
```

## Requirements
- python 3.6+
- torch 1.7+
- torchvision 0.8+
- tensorboard
- wandb
- numpy
- tqdm
- pandas
- sentencepiece
- sklearn
- Transformers
