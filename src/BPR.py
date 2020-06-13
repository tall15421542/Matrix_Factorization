import torch
import torch.utils.data
import torch.nn
import torch.utils.data
import random
import sys

#define some constant
LATENT_DIM = 32
NEG_RATIO = 1
VALIDATION_RATIO = 0.1
LEARNING_RATE = 1e-1
ITERATION_TIMES = 150000

class User:
    def __init__(self, userId, itemsSet):
        self.userId = userId
        self.itemsSet = itemsSet
    
    def getNegativeSample(self, maxItemId):
        negativeSampleId = random.randrange(maxItemId)
        while negativeSampleId in self.itemsSet:
            negativeSampleId = random.randrange(maxItemId)
        return (self.userId, negativeSampleId, 1)

    def getNegativeSamples(self, ratio, maxItemId):
        numOfNegSamples = int(self.getNumOfItems() * ratio)
        samples = []
        for i in range(numOfNegSamples):
            samples.append(self.getNegativeSample(maxItemId))
        return samples

    def getPositiveSample(self):
        return (self.userId, random.choice(tuple(self.itemsSet)))

    def getPositiveSamples(self):
        samples = []
        for item in self.itemsSet:
            samples.append((self.userId, item, 1))
        return samples

    def getNumOfItems(self):
        return len(self.itemsSet)


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
                print(self.samples)
            except:
                print(error)
                exit()
           

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


    #def getPairWiseSample():
    

class MatrixFactorization(torch.nn.Module):
    def __init__(self, userSize, numOfSize, latentDim):
        super(MatrixFactorization, self).__init__()
        self.user_factors = torch.nn.Embedding(userSize, latentDim)
        self.item_factors = torch.nn.Embedding(numOfSize, latentDim)
    def forward(self, userId, itemId):
        assert not torch.isnan(self.user_factors(userId)).any()
        assert not torch.isnan(self.item_factors(itemId)).any()
        return (self.user_factors(userId) * self.item_factors(itemId)).sum(1)

def BCELoss(prediction, label):
    if label is True:
        return -torch.log(torch.sigmoid(prediction))
    else:
        return -torch.log(1 - torch.sigmoid(prediction))

# Backpropagate
# Optimize - SGD
def main():
    if len(sys.argv) != 2:
        print("file path is required")
        exit()
    datasetURL = str(sys.argv[1])
    # Load data
    dataset = RecommendationDataset(datasetURL)
    
    # split to training dataset and validation set
    userWiseSample = dataset.getUserWiseSample(NEG_RATIO)
    validationSize = int(len(userWiseSample) * VALIDATION_RATIO)
    trainingSize = len(userWiseSample) - validationSize
    trainingData, validationData = torch.utils.data.random_split(userWiseSample, [trainingSize, validationSize]) 
    # Build Matrix factorization model
    numOfUser, numOfItem = dataset.getNumOfUsers(), dataset.getNumOfItems()
    matrixFactorizationModel = MatrixFactorization(numOfUser, numOfItem, LATENT_DIM)

    # Define criterion and optimizer
    criterion = torch.nn.BCELoss(reduction = "sum")
    optimizer = torch.optim.SGD(matrixFactorizationModel.parameters(), lr = LEARNING_RATE)
    
    # training - Forward / optimize
    for t in range(ITERATION_TIMES):
        if t >= len(trainingData):
            break
        userId, itemId, label = trainingData[t]
        prediction = matrixFactorizationModel(torch.tensor([userId]), torch.tensor([itemId]))
        assert not torch.isnan(prediction).any()
        loss = criterion(torch.sigmoid(prediction), torch.tensor(label, dtype = torch.float))
        assert not torch.isnan(loss)
        if t % 10000 == 0:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validation
    with torch.no_grad():
        validation_loss = torch.zeros((1))
        for data in validationData:
            userId, itemId, label = data
            prediction = matrixFactorizationModel(torch.tensor([userId]), torch.tensor([itemId]))
            assert not torch.isnan(prediction).any()
            loss = criterion(torch.sigmoid(prediction), torch.tensor(label, dtype=torch.float))
            assert not torch.isnan(loss).any()
            validation_loss += loss
            
        print(validation_loss/len(validationData))

if __name__ == "__main__":
    main()
