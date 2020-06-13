import torch
import torch.utils.data
import torch.nn
import torch.utils.data
import random
import sys

#define some constant
LATENT_DIM = 32
NEG_RATIO = 1
VALIDATION_RATIO = 0.2
LEARNING_RATE = 1e-1
NUM_OF_EPOCH = 15
ITER_FACTOR = 1
REGULAR_FACTOR = 2e-4

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
        validationSet = set(self.validationSamples)
        score_tensor = model(torch.tensor([self.userId]), torch.arange(0, maxItemId))
        return torch.topk(score_tensor, 50).indices

    def outputRankingList(self, model, maxItemId, writeFile):
        rankingList = self.getTopkIndices(model, maxItemId)
        writeFile.write('{0},'.format(self.userId))
        for indice in rankingList:
            writeFile.write('{0} '.format(indice))
        writeFile.write('\n')

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

class MatrixFactorization(torch.nn.Module):
    def __init__(self, userSize, numOfSize, latentDim):
        super(MatrixFactorization, self).__init__()
        self.user_factors = torch.nn.Embedding(userSize, latentDim)
        self.item_factors = torch.nn.Embedding(numOfSize, latentDim)
    def forward(self, userId, itemId):
        return (self.user_factors(userId) * self.item_factors(itemId)).sum(1)

def BCELoss(prediction, label):
    if label is True:
        return -torch.log(torch.sigmoid(prediction))
    else:
        return -torch.log(1 - torch.sigmoid(prediction))

def BPRLoss(posPrediction, negPrediction):
    return -torch.log(torch.sigmoid(posPrediction - negPrediction))

# Backpropagate
# Optimize - SGD
def main():
    if len(sys.argv) < 2:
        print("file path is required")
        exit()
    datasetURL = str(sys.argv[1])
    # Load data
    dataset = RecommendationDataset(datasetURL)
    dataset.splitValidationAndTesting(VALIDATION_RATIO)
    
    # Build Matrix factorization model
    numOfUser, numOfItem = dataset.getNumOfUsers(), dataset.getNumOfItems()
    matrixFactorizationModel = MatrixFactorization(numOfUser, numOfItem, LATENT_DIM)
    
    # Define criterion and optimizer
    criterion = BPRLoss
    optimizer = torch.optim.SGD(matrixFactorizationModel.parameters(), lr = LEARNING_RATE, weight_decay = REGULAR_FACTOR)
    

    for epoch in range(NUM_OF_EPOCH):
        trainingData = dataset.getPairWiseSample(NEG_RATIO)
        print(epoch ," Round")
        # split to training dataset and validation set

        # training - Forward / optimize
        loss_stat = torch.zeros((1))
        interval = int((1 - VALIDATION_RATIO) * dataset.getNumOfSamples() * ITER_FACTOR * 0.1) 
        print("interval: ", interval)
        for t in range(int((1 - VALIDATION_RATIO) * dataset.getNumOfSamples() * ITER_FACTOR)):
            userId, posItemId, negItemId = random.choice(trainingData)
            posPrediction = matrixFactorizationModel(torch.tensor([userId]), torch.tensor([posItemId]))
            negPrediction = matrixFactorizationModel(torch.tensor([userId]), torch.tensor([negItemId]))
            loss = criterion(posPrediction, negPrediction)
            loss_stat += loss
            # print(loss)
            if t % interval == 0:
                print(loss_stat/interval)
                loss_stat = torch.zeros((1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

         # validation
        print(dataset.evaluateMAP(matrixFactorizationModel))
    
    # Open output file
    if len(sys.argv) == 3:
        writeFile = open(str(sys.argv[2]), "w")
        writeFile.write("UserId,ItemId\n")
        dataset.outputRankingList(matrixFactorizationModel, writeFile)
        
if __name__ == "__main__":
    main()
