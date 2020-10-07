
import numpy as np
from scipy.io import savemat
import pandas as pd

# ML or DL Libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import Tensor
from sklearn.metrics import classification_report


def test_model(test_dataloader, device, model_dir, epoch_no):

    ''' Load the trained model of given epoch_no '''
    model = torch.load(model_dir+"/CNN_ep_"+str(epoch_no)+".pth")
    model.to(device)

    ''' Testing '''

    predicted_labels = list()
    gold_labels = list()

    for en, batch in enumerate(testing_data):
        
        ip_data, gold_label, _ = batch

        ip_data = Variable(ip_data.type('torch.FloatTensor')).to(device)
        ip_data = torch.reshape(ip_data, [1, ip_data.size(0), ip_data.size(1), ip_data.size(2)])

        if (gold_label.item() <= 1):
            label = Variable((Tensor(1).fill_(0)), requires_grad=False).to(device)
        else:                                                                                      
            label = Variable((Tensor(1).fill_(1)), requires_grad=False).to(device)

        # Chunk level prediction
        chunk_level_prediction = list()

        no_of_chunks = (ip_data.shape[2]//100)
        
        for i in range(no_of_chunks): 
            a = ip_data[:,:,i*100:(i*100)+100,:]
            out = model(a)

            if (out.item() < 0.5):
                chunk_level_prediction.append(0)
            else:
                chunk_level_prediction.append(1)

        if (ip_data.shape[2]%100) > 45:
            a = ip_data[:,:,ip_data.shape[2]-100:ip_data.shape[2],:]
            out = model(a)

            if (out.item() < 0.5):
                chunk_level_prediction.append(0)
            else:
                chunk_level_prediction.append(1)

        # Predict the label using max voting algorithm
        predicted_labels.append(max(set(chunk_level_prediction), key = chunk_level_prediction.count))
        gold_labels.append(label.item())

    ''' Calculate and save the result '''
    report= classification_report(gold_labels, predicted_labels)
    dict_report= classification_report(gold_labels, predicted_labels, output_dict=True)

    return report, dict_report
