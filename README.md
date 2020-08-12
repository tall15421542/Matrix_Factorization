# Information Retrieval Programming Homework 2 Report
## Paper reference
[BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf)
## Q1 : Describe your MF with BCE (e.g. parameters, loss function, negative sample method and MAP score on Kaggle public scoreboard)
* Initialization of User Latent Matrix and Item Latent Matrix ~ N(0, 0,01)
* Learning Rate: 0.01
* Regular Lambda: 1e-7
* Batch Size: 4096
* Loss function: ``Torch.nn.BCELoss(reduction = 'sum')``
![](https://i.imgur.com/zSRxHBR.png)
* Negative Sample Method
    * For each User, iterate all of his Items. For each Item, uniform-randomly pick ``NEG_SAMPLE_NUM`` negative sample(s)
    * For each epoch, I would re-sample the negative sample again, the process is same as above.

| Latent Dimension | Negative Sample Ratio | Epoch Numbers | Public Score |
|------------------|-----------------------|---------------|--------------|
| 16               | 1                     | 150           | 0.04067      |
| 16               | 2                     | 100           | 0.03965      |
| 16               | 3                     | 100           | 0.04742      |

| Latent Dimension | Negative Sample Ratio | Epoch Numbers | Public Score |
|------------------|-----------------------|---------------|--------------|
| 32               | 1                     | 150           | 0.04407      |
| 32               | 2                     | 100           | 0.04297      |
| 32               | 3                     | 100           | 0.03988      |

| Latent Dimension | Negative Sample Ratio | Epoch Numbers | Public Score |
|------------------|-----------------------|---------------|--------------|
| 64               | 1                     | 150           | 0.04688      |
| 64               | 2                     | 100           | 0.04711      |
| 64               | 3                     | 100           | 0.04337      |

| Latent Dimension | Negative Sample Ratio | Epoch Numbers | Public Score |
|------------------|-----------------------|---------------|--------------|
| 128              | 1                     | 150           | 0.04997      |
| 128              | 2                     | 100           | 0.04671      |
| 128              | 3                     | 100           | 0.04897      |
## Q2:DescribeyourMFwithBPR(e.g.parameters,lossfunction,negativesample method and MAP score on Kaggle public scoreboard)
* Initialization of User Latent Matrix and Item Latent Matrix ~ N(0, 0,01)
* Learning Rate: 0.01
* Regular Lambda: 1e-7
* Batch Size: 4096
* Loss function: 
```
def BPRLoss(posPrediction, negPrediction):
    assert len(posPrediction) == len(negPrediction)
    return -(posPrediction - negPrediction).sigmoid().log().sum()

```
![](https://i.imgur.com/JYKASAr.png)

* Negative Sample Method
    * For each User, iterate all of his Items. For each Item, uniform-randomly pick ``NEG_SAMPLE_NUM`` negative sample
    * For each epoch, I would re-sample the negative sample again, the process is same as above.

| Latent Dimension | Negative Sample Ratio | Epoch Numbers | Public Score |
|------------------|-----------------------|---------------|--------------|
| 16               | 1                     | 150           | 0.452            |
| 16               | 2                     | 100           | 0.04757      |
| 16               | 3                     | 100           | 0.04677      |

| Latent Dimension | Negative Sample Ratio | Epoch Numbers | Public Score |
|------------------|-----------------------|---------------|--------------|
| 32               | 1                     | 150           | 0.0496            |
| 32               | 2                     | 100           | 0.04808      |
| 32               | 3                     | 100           | 0.04994      |

| Latent Dimension | Negative Sample Ratio | Epoch Numbers | Public Score |
|------------------|-----------------------|---------------|--------------|
| 64               | 1                     | 150           | 0            |
| 64               | 2                     | 100           | 0.05058      |
| 64               | 3                     | 100           | 0.0496       |

| Latent Dimension | Negative Sample Ratio | Epoch Numbers | Public Score |
|------------------|-----------------------|---------------|--------------|
| 128              | 1                     | 150           | 0.5229       |
| 128              | 2                     | 100           | 0.05315      |
| 128              | 3                     | 100           | 0.05139      |


## Q3 : Compare your results of Q1 and Q2. Do you think the BPR loss benefits the performance? If do, write some reasons of why BPR works well; If not, write some reasons of why BPR fails.
From the above table, we can observe that given the same parameters, BPR outperperforms BCE in general.

Yes. Because for MAP evaluation, ranking of the item matters. For BCE, it only consider Positive or Negative error. For BPR, it considers the relation between two items. 

The loss function is important, it would decide where the parametes of the model go. In the case of BCE, the model would improve for the 0/1 decision. In the case of BPR, our model would be trained to describe the relation among two items, which is the key to rank a list of items. 

Therefore, given the rank related evaluation, MAP, BPR loss would lead the model toward the better direction.

## Q4 : Plot the MAP curve on testing data(Kaggle) for hidden factors 𝑑 = 16, 32, 64, 128 and describe your finding.
Given the same parameters, the performance is imporved given higher hidden factor in general, both BPR and BCE.

Here we use the model with Negative Sample Ratio = 2, Epoch = 100, BPR loss as example
![](https://i.imgur.com/hR7Qpmo.png)

Higher hidden factors give the model higher power to describe the real world data. Though the model might overfit the training data if we give them "too much" power with higher hidden factors, it seems that 128 hidden factors matrix factorization is not an over-complicated model for this problem. 

## (Bonus 10%) Q5 : Change the ratio between positive and negative pairs, compare the results and discuss your finding.
If we give higher ratio, our model can see more sample pairs in one epoch. In my algorithm, a training process would see ``num_of_Positive_pair * neg_ratio * epoch * batch_size`` negative samples in total. Therefore, instead of observing RATIO only, I want to discuss ``neg_ratio * epoch`` this two parameters together to see the relation between observed negative samples and testing performance

It could be great for our model to see more cooperative data to describe the real world. However, it could be bad, because not all of the negative samples are really "Negative", they might be negative only because user has not seen it yet. Therefore, overfitting to the "Negative samples" is not a good thing.

In a simple model, performance is imporved when increasing the observed cooperative data(negative samples) 
Here use the model with Latent Dimension = 16 and BCE loss as the example
| Latent Dimension | Negative Sample Ratio | Epoch Numbers | Ratio * Epoch | Public Score |
|------------------|-----------------------|---------------|---------------|--------------|
| 16               | 1                     | 150           | 150           | 0.04067      |
| 16               | 2                     | 100           | 200           | 0.03965      |
| 16               | 3                     | 100           | 300           | 0.04742      |

However, in a complicated model, performance might be downgraded due to overfitting to the negative samples. From the tables in the Q1 and Q2, we can see that in general the perfomance is downgraded from``neg_ratio * epoch = 200`` to ``neg_ratio * epoch = 300`` given the hidden factor >= 64

The performance is similiar when ``neg_ratio * epoch`` is the same. Take model with Hidden factor 128 and BPR loss as the example.

| Latent Dimension | Negative Sample Ratio | Epoch Numbers | Ratio * Epoch | Public Score |
|------------------|-----------------------|---------------|---------------|--------------|
| 128              | 1                     | 150           | 150           | 0.05244      |
| 128              | 3                     | 50            | 150           | 0.05229      |


