import torch
import torch.utils.data
import torch.nn
import torch.utils.data
import random
import sys
class User:
    def __init__(self, userId, itemsSet):
        self.userId = userId
        self.itemsSet = itemsSet
        self.trainingSamples = []
        self.validationSamples = []
   
    def getNegativeSampleId(self, maxItemId):
        negativeSampleId = random.randrange(maxItemId)
        while negativeSampleId in self.trainingSamples:
            negativeSampleId = random.randrange(maxItemId)
        return negativeSampleId

    def getNegativeSample(self, maxItemId):
        return (self.userId, self.getNegativeSampleId(maxItemId), 1)

    def getNegativeSamples(self, ratio, maxItemId):
        numOfNegSamples = int(self.getNumOfTrainingItems() * ratio)
        samples = []
        for i in range(numOfNegSamples):
            samples.append(self.getNegativeSample(maxItemId))
        return samples

    def getPositiveSample(self):
        return (self.userId, random.choice(tuple(self.trainingSamples)))

    def getPositiveSamples(self):
        samples = []
        for item in self.trainingSamples:
            samples.append((self.userId, item, 1))
        return samples

    def getNumOfTrainingItems(ratio):
        return len(self.trainingSamples)

    def getNumOfItems(self):
        return len(self.itemsSet)

    def getPairWiseSample(self, ratio, maxItemId):
        negativeSamples = int(ratio)
        samples = []
        for item in self.trainingSamples:
            for i in range(negativeSamples):
                samples.append((self.userId, item, self.getNegativeSampleId(maxItemId)))
        return samples

    def splitValidationAndTesting(self, ratio):
        numOfValidations = int(self.getNumOfItems() * ratio)
        numOfTraining = self.getNumOfItems() - numOfValidations
        self.trainingSamples, self.validationSamples = torch.utils.data.random_split(tuple(self.itemsSet), [numOfTraining, numOfValidations])

    def evaluateMAP(self, model, maxItemId):
        validationSet = set(self.validationSamples)
        rankingList = self.getTopkIndices(model, maxItemId)
        score = 0.
        hit = 0
        for rank, itemId in enumerate(rankingList):
            if itemId.item() in validationSet:
                hit += 1
                score += (hit)/(rank + 1)
        score /= len(validationSet)
        return score
    
    def getTopkIndices(self, model, maxItemId):
        score_tensor = model(torch.tensor([self.userId]), torch.arange(0, maxItemId))
        score_tensor[self.trainingSamples] = -float('inf')
        return torch.topk(score_tensor, 50).indices

    def outputRankingList(self, model, maxItemId, writeFile):
        rankingList = self.getTopkIndices(model, maxItemId)
        writeFile.write('{0},'.format(self.userId))
        for indice in rankingList:
            writeFile.write('{0} '.format(indice))
        writeFile.write('\n')
