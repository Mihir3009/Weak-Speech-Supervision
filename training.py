
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import Tensor
from tqdm import trange


def train_model(train_dataloader, device, model, optimizer, loss_fn, alpha, epoch, model_dir):

    for ep in trange(epoch):
        model.train()
        epoch_loss = 0

        for iteration, batch in enumerate(train_dataloader):
            
            mcc, label, boolean_indication_weak_data = batch
            # Initial variables
            batch_loss = 0
            
            # Reshape the input data
            mcc = Variable(mcc.type('torch.FloatTensor')).to(device)
            mcc = torch.reshape(mcc, [1, mcc.size(0), mcc.size(1), mcc.size(2)])
            label = Variable((Tensor(1, 1).fill_(label.item())), requires_grad=False).to(device)
            
            '''
            We devided the mcc features into chunk of size 100 and use those chunks for the training.
            After that, remaining feature part is included if those feature samples exceeded 45, deiscarded otherwise.
            E.g., if feature dimension of current batch is 766x40, then it will devide the features into
            chunks of size (100x40); hence, we will get 7 chunks for this batch, and we will keep remaining
            (66x40) as it exceed the sample size, i.e., 45.
            '''

            if (mcc.shape[2] < 100):
                continue
            no_of_chunks = (mcc.shape[2]//100)

            # Training for chunks
            for i in range(no_of_chunks):
                optimizer.zero_grad()
                a = mcc[:,:,i*100:(i*100)+100,:]
                out = model(a)
                
                # If weak data, then denoise the label
                if (boolean_indication_weak_data == 1):
                    loss = loss_fn(out, label)
                else:
                    loss = loss_fn(out, label) * alpha
            
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            # This will run in the case of inclusion of remaining features
            if (mcc.shape[2]%100) > 45:
                optimizer.zero_grad()
                a = mcc[:,:,mcc.shape[2]-100:mcc.shape[2],:]
                out = model(a)

                # If weak data, then denoise the label
                if (boolean_indication_weak_data == 1):
                    loss = loss_fn(out, label)
                else:
                    loss = loss_fn(out, label) * alpha

                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            epoch_loss += batch_loss

        print("Epoch: {} | Loss: {}".format(ep, epoch_loss))
    
    return model
