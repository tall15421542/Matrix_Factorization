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
def main():
    if len(sys.argv) != 5:
        print("model and output and file path and latenr dim is required")
        exit()

    datasetURL = str(sys.argv[1])
    outputURL = str(sys.argv[2])
    modelURL = str(sys.argv[3])
    LATENT_DIM = int(sys.argv[4])
    # Load data
    dataset = RecommendationDataset(datasetURL)
    dataset.splitValidationAndTesting(0)
    
    # Build Matrix factorization model
    numOfUser, numOfItem = dataset.getNumOfUsers(), dataset.getNumOfItems()
    matrixFactorizationModel = MatrixFactorization(numOfUser, numOfItem, LATENT_DIM)
    matrixFactorizationModel.load_state_dict(torch.load(modelURL)) 
    
    # Open output file
    writeFile = open(outputURL, "w")
    writeFile.write("UserId,ItemId\n")
    dataset.outputRankingList(matrixFactorizationModel, writeFile)
        
if __name__ == "__main__":
    main()
