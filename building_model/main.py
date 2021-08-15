
# Libraries
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import argparse
import numpy as np
from os import makedirs
from os.path import join, exists
import json

# Visualization Libraries
import matplotlib.pyplot as plt

# DL Libraries
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Evaluation Libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from scipy.io import loadmat
from scipy.io import savemat
from scipy.spatial import distance

# Import from our files
from dataloader import speech_data
from model import CNN
from training import train_model


def main():

    # Check GPU availability
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} number_of_gpu: {}".format(device, n_gpu)

    # Parse script arguments
    parser = argparse.ArgumentParser()

    ## required argument
    parser.add_argument("--data_dir", default="",
                        type=str, required=True,
                        help="Provide path of folder where the trainig data is stored")
    parser.add_argument("--output_dir", default="",
                        type=str, required=True,
                        help="Provide path of folder where to save the trained models")
    
    ## Optional Argument
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_evaluation",
                        action='store_true',
                        help="Whether to run evaluation on the test set.")
    parser.add_argument("--batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=0.0001,
                        type=float,
                        help="The initial learning rate for Adam optimizer.")
    parser.add_argument("--training_epochs",
                        default=50,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--test_epoch",
                        default=50,
                        type=int,
                        help="The epoch number that you want to test.")
    parser.add_argument("--alpha_parameter", 
                        default=0.1,
                        type=float,
                        help="Provide the value of alpha paramater"))

    args= parser.parse_args()
    
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `--do_train` or `--do_evaluation` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_dir= join(args.output_dir, 'trained_models_checkpoints')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if args.do_train:

        ''' Create dataloader for the training '''
        training_path = speech_data(folder_path=join(args.data_dir, 'train_data'))
        train_dataloader = DataLoader(dataset=training_path, batch_size=args.batch_size, shuffle=True, num_workers=2)

        ''' Initialization '''

        # Initialize network
        model = CNN().to(device)

        # Loss function
        loss_fn = nn.BCELoss().to(device)

        # Initialize optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate)

        # Initialize Alpha (relative importance of the standard and weak data) paramater
        alpha = args.alpha_parameter

        ''' Training '''
        logger.info("Training is Started")

        model = train_model(train_dataloader, device, model, 
                            optimizer, loss_fn, alpha, args.training_epochs, model_dir)

    if args.do_evaluation:  
        ''' Create dataloader for the testing '''
        testing_path = speech_data(folder_path=join(args.data_dir, 'test_data'))
        test_dataloader = DataLoader(dataset=testing_path, batch_size=args.batch_size, shuffle=True, num_workers=2)

        ''' Testing '''
        logger.info("Testing is Started")

        results, dict_results= test_model(test_dataloader, device, model_dir, args.test_epoch)

        # metrics_report is in the form of dictionary includes all the classification measures
        logger.info("Test Results:")
        print(results)

        # Save the results 
        with open(join(args.output_dir, "result.json"), "w") as outfile:
            json.dump(dict_results, outfile)
        

if __name__=="__main__":
    main()
