# import libraries
import numpy as np
#import matplotlib.pyplot as plt #throws an error ("Failed to load Tcl_SetVar") under Win 10 ands is not needed
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm

def run(dwi_image_long,batch_size,iter_max):

    # define ivim function
    def ivim(b, Dp, Dt, Fp):
        return Fp*np.exp(-b*Dp) + (1-Fp)*np.exp(-b*Dt)

    # define b values
    b_values = np.array([0,10,20,40,80,110,140,170,200,300,400,500,600,700,800,900])
    #b_values = np.array([0,10,20,60,150,300,500,1000])
    
    # training data
    X_train = dwi_image_long

    class Net(nn.Module):
        def __init__(self, b_values_no0):
            super(Net, self).__init__()

            self.b_values_no0 = b_values_no0
            self.fc_layers = nn.ModuleList()
            for i in range(3): # 3 fully connected hidden layers
                self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
            self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), 3))

        def forward(self, X):
            params = torch.abs(self.encoder(X)) # Dp, Dt, Fp
            Dp = params[:, 0].unsqueeze(1)
            Dt = params[:, 1].unsqueeze(1)
            Fp = params[:, 2].unsqueeze(1)

            X = Fp*torch.exp(-self.b_values_no0*Dp) + (1-Fp)*torch.exp(-self.b_values_no0*Dt)

            return X, Dp, Dt, Fp

    # Network
    b_values_no0 = torch.FloatTensor(b_values[1:])
    net = Net(b_values_no0)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.001)

    #batch_size = 128
    num_batches = len(X_train) // batch_size
    X_train = X_train[:,1:] # exlude the b=0 value as signals are normalized
    trainloader = utils.DataLoader(torch.from_numpy(X_train.astype(np.float32)),
                                    batch_size = batch_size, 
                                    shuffle = True,
                                    num_workers = 2,
                                    drop_last = True)

    # Best loss
    best = 1e16
    num_bad_epochs = 0
    patience = 10

    # Train
    #for epoch in range(1000): #GS
    loss_all=[None] * iter_max #GS
    for epoch in range(iter_max): #GS
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0.

        for i, X_batch in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            X_pred, Dp_pred, Dt_pred, Fp_pred = net(X_batch)
            loss = criterion(X_pred, X_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Loss: {}".format(running_loss))
        # early stopping
        if running_loss < best:
            print("############### Saving good model ###############################")
            final_model = net.state_dict()
            best = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience:
                print("Done, best loss: {}".format(best))
        #        break #GS: no early stopping

        loss_all[epoch] = running_loss #GS

    print("Done")
    # Restore best model
    net.load_state_dict(final_model)

    # normalize signal
    S0 = np.expand_dims(dwi_image_long[:,0], axis=-1)
    dwi_image_long = dwi_image_long[:,1:]/S0

    net.eval()
    with torch.no_grad():
        _, Dp, Dt, Fp = net(torch.from_numpy(dwi_image_long.astype(np.float32)))

    Dp = Dp.numpy()
    Dt = Dt.numpy()
    Fp = Fp.numpy()

    # make sure Dp is the larger value between Dp and Dt
    if np.mean(Dp) < np.mean(Dt):
        Dp, Dt = Dt, Dp
        Fp = 1 - Fp

    Dp = np.array(Dp) #GS: added
    Dt = np.array(Dt) #GS: added
    Fp = np.array(Fp) #GS: added
    loss_all = np.array(loss_all) #GS: added

    return Dp, Dt, Fp, loss_all


