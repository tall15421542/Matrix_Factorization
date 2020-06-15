import torch
def BPRLoss(posPrediction, negPrediction):
    assert len(posPrediction) == len(negPrediction)
    return -(posPrediction - negPrediction).sigmoid().log().sum()
