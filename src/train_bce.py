import torch
import torch.utils.data
import torch.nn
import torch.utils.data
import random
import sys
from User import User
from RecommendationDataset import RecommendationDataset
from MatrixFactorization import MatrixFactorization
from Loss import BPRLoss
#define some constant
LATENT_DIM = 64
NEG_RATIO = 3
VALIDATION_RATIO = 0.11
LEARNING_RATE = 1e-2
NUM_OF_EPOCH = 100
REGULAR_FACTOR = 1e-7
BATCH_SIZE = 4096


def main():
    if len(sys.argv) != 6:
        print("arg is missed")
        exit()
    VALIDATION = True
    datasetURL = str(sys.argv[1])
    datasetFileURL = datasetURL.split('/')
    datasetFileURL = datasetFileURL[1]
    if str(sys.argv[2]) == 'train':
        print("Training")
        VALIDATION = False
        validation_ratio = 0
    elif str(sys.argv[2] == 'validate'):
        validation_ratio = VALIDATION_RATIO
        print("Validation")
    else:
        print("Invalid arg. limited to train/validate")
        exit()
    LATENT_DIM = int(sys.argv[3])
    NEG_RATIO = int(sys.argv[4])
    NUM_OF_EPOCH = int(sys.argv[5])
    
    print('Latent: {0}'.format(LATENT_DIM))
    print('NEG: {0}'.format(NEG_RATIO))
    print('NUM_OF_EPOCH: {0}'.format(NUM_OF_EPOCH))
    print('REGULAR_FACTOR: {0}'.format(REGULAR_FACTOR))
    print('LEARNING_RATE: {0}'.format(LEARNING_RATE))
    print('BATCH_SIZE: {0}'.format(BATCH_SIZE))

    if VALIDATION:
        logURL = 'log/latent_{0}_epoch_{1}_neg_{2}_{3}.bce.log'.format(LATENT_DIM, NUM_OF_EPOCH, NEG_RATIO, datasetFileURL)
        logFile = open(logURL, "w")
        logFile.write('Train source: {0}\n'.format(datasetURL))
        logFile.write('Latent: {0}\n'.format(LATENT_DIM))
        logFile.write('NEG: {0}\n'.format(NEG_RATIO))
        logFile.write('NUM_OF_EPOCH: {0}\n'.format(NUM_OF_EPOCH))
        logFile.write('REGULAR_FACTOR: {0}\n'.format(REGULAR_FACTOR))
        logFile.write('LEARNING_RATE: {0}\n'.format(LEARNING_RATE))
        logFile.write('BATCH_SIZE: {0}\n'.format(BATCH_SIZE))
        logFile.write('epoch,map\n');
    # Load data
    dataset = RecommendationDataset(datasetURL)
    dataset.splitValidationAndTesting(validation_ratio)
    
    # Build Matrix factorization model
    numOfUser, numOfItem = dataset.getNumOfUsers(), dataset.getNumOfItems()
    matrixFactorizationModel = MatrixFactorization(numOfUser, numOfItem, LATENT_DIM)
    
    # Define criterion and optimizer
    criterion = torch.nn.BCELoss(reduction = 'sum')
    optimizer = torch.optim.SGD(matrixFactorizationModel.parameters(), lr = LEARNING_RATE, weight_decay = REGULAR_FACTOR)

    for epoch in range(NUM_OF_EPOCH):
        trainingData = dataset.getPairWiseSample(NEG_RATIO)
        print(epoch ," Round")
        # split to training dataset and validation set

        # training - Forward / optimize
        sizeOfTrainingData = len(trainingData) 
        start = 0
        interval = sizeOfTrainingData/10
        loss_stat = torch.zeros((1))
        cnt_stat = 0
        interval_stat = interval 
        while start < sizeOfTrainingData:
            end = (start + BATCH_SIZE) if (start + BATCH_SIZE) < sizeOfTrainingData else sizeOfTrainingData 
            dataList = trainingData[start:end]
            userIds = [data[0] for data in dataList] 
            posItemIds = [data[1] for data in dataList] 
            negItemIds = [data[2] for data in dataList] 
            posPrediction = matrixFactorizationModel(torch.tensor(userIds), torch.tensor(posItemIds))
            negPrediction = matrixFactorizationModel(torch.tensor(userIds), torch.tensor(negItemIds))
            loss = criterion(posPrediction.sigmoid(), torch.ones(posPrediction.shape, dtype = torch.float)) \
                 + criterion(negPrediction.sigmoid(), torch.zeros(negPrediction.shape, dtype = torch.float))
            loss_stat += loss
            cnt_stat += 1
            if (end > interval_stat) or end == sizeOfTrainingData:
                interval_stat += interval
                print(loss_stat/BATCH_SIZE/cnt_stat)
                loss_stat = torch.zeros((1))
                cnt_stat = 0

            # Backpropagate
            # Optimize - SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            start += BATCH_SIZE

        print(epoch, "Round end")
        # validation
        if VALIDATION:
            mapScore = (dataset.evaluateMAP(matrixFactorizationModel))
            logFile.write('{0} {1}\n'.format(epoch, mapScore))
            print(mapScore)
    
    datasetURL = datasetURL.split('/')
    datasetURL = datasetURL[1]
    # Open output file
    

    # write model
    if not VALIDATION:
        torch.save(matrixFactorizationModel.state_dict(), '{0}latent_{1}_epoch_{2}_neg_{3}_{4}.bce'.format('model/', LATENT_DIM, NUM_OF_EPOCH, NEG_RATIO, datasetURL))
        outputPath = '{0}latent_{1}_epoch_{2}_neg_{3}_{4}.bce.out'.format('output/', LATENT_DIM, NUM_OF_EPOCH, NEG_RATIO, datasetURL)
        writeFile = open(outputPath, "w")
        writeFile.write("UserId,ItemId\n")
        dataset.outputRankingList(matrixFactorizationModel, writeFile)

    print('Latent: {0}'.format(LATENT_DIM))
    print('NEG: {0}'.format(NEG_RATIO))
    print('NUM_OF_EPOCH: {0}'.format(NUM_OF_EPOCH))
    print('REGULAR_FACTOR: {0}'.format(REGULAR_FACTOR))
    print('LEARNING_RATE: {0}'.format(LEARNING_RATE))
    print('BATCH_SIZE: {0}'.format(BATCH_SIZE))
        
if __name__ == "__main__":
    main()
