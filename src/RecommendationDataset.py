import torch
import torch.utils.data
import torch.nn
import torch.utils.data
import random
import sys
from User import User

class RecommendationDataset():
    def __init__(self, datasetURL):
        self.user = []
        self.samples = 0;
        with open(datasetURL) as file:
            try:
                # read first useless line
                file.readline()
                maxItemId = 0
                for line in file:
                    userId, itemsString = line.split(",")
                    userId = int(userId)
                    itemsSet = set(int(item) for item in itemsString.split())
                    self.user.append(User(userId, itemsSet))
                    maxItemIdInSet = max(itemsSet)
                    maxItemId = maxItemId if maxItemId > maxItemIdInSet else maxItemIdInSet
                    self.samples += len(itemsSet)
                self.numOfItems = maxItemId + 1
                print("Positive samples: ", self.samples)
            except:
                print(error)
                exit()
    def getNumOfSamples(self):
        return self.samples

    def sampleUser(self):
        return random.choice(self.user)

    def getNumOfItems(self):
        return self.numOfItems

    def getNumOfUsers(self):
        return len(self.user)

    def getUserWiseSample(self, ratio):
        samples = []
        for user in self.user:
            samples.extend(user.getPositiveSamples())
            samples.extend(user.getNegativeSamples(ratio, self.numOfItems))
        return samples

    def getPairWiseSample(self, ratio):
        samples = []
        for user in self.user:
            samples.extend(user.getPairWiseSample(ratio, self.numOfItems))
        return samples

    def splitValidationAndTesting(self, ratio):
        for user in self.user:
            user.splitValidationAndTesting(ratio)

    def evaluateMAP(self, model):
        score = 0.
        for user in self.user:
            score += user.evaluateMAP(model, self.getNumOfItems())
        return score / len(self.user)
    def outputRankingList(self, model, writeFile):
        for user in self.user:
            user.outputRankingList(model, self.getNumOfItems(), writeFile)
