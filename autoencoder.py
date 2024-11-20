import torch
import numpy as np
import os


class RNN(torch.nn.Module):
    '''
    This is the function that defines the structure of the LSTM neural network.
    The return value is the result after LSTM encoder and decoder.
    '''
    def __init__(self, s, d):
        super().__init__()
        self.s = s
        self.d = d
        self.rnn_1 = torch.nn.LSTM(input_size=self.s, hidden_size=64*2, num_layers=3, bidirectional=True, batch_first=True)
        self.out_1 = torch.nn.Linear(in_features=64*2, out_features=self.d)

        self.rnn_2 = torch.nn.LSTM(input_size=self.d, hidden_size=64*2, num_layers=3, bidirectional=True, batch_first=True, dropout=0.1)
        self.out_2 = torch.nn.Linear(in_features=64*2, out_features=self.s)

    def forward(self, x):
        #encoder
        output, (h_n, c_n) = self.rnn_1(x)
        output_in_last_timestep = h_n[-1, :, :]
        encoder = self.out_1(output_in_last_timestep)

        #decoder
        output1, (h_n1, c_n1) = self.rnn_2(encoder.view(-1, 1, self.d))
        output_in_last_timestep1 = h_n1[-1, :, :]
        decoder = self.out_2(output_in_last_timestep1)
        return encoder, decoder


def train_autoencoder(trajectory_loss_re):
    '''
    Function to train LSTM autoencoder, return value 's' is the original data dimension, 'd' is the dimension of the data after dimensionality reduction, 'net' is the trained classifier.
    '''
    trajectory_loss_re = torch.from_numpy(np.asmatrix(trajectory_loss_re)).float()
    X_train = trajectory_loss_re[:6000]
    X_test = trajectory_loss_re[6000:8000]
    s = X_train.shape[1]
    d = int(s * 2 / 3)
    net = RNN(s, d)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    loss_F = torch.nn.MSELoss()
    best_loss = 1000
    if not os.path.exists("./LSTM classifiers/Autoencoder.pth"):
        print("Training LSTM Based Autoencoder:")
        for epoch in range(10):
            #Training
            net.train()
            _, pred = net(X_train.view(-1, 1, s))
            loss = loss_F(pred, X_train.view(-1, s))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(' Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy())
            #Testing
            net.eval()
            _, pred1 = net(X_test.view(-1, 1, s))
            loss_e = loss_F(pred1, X_test.view(-1, s))
            print('  Eval loss: %.4f' % loss_e.cpu().data.numpy())
            if loss_e < best_loss:
                best_loss = loss_e
                torch.save(net.state_dict(), './LSTM classifiers/Autoencoder.pth')
                print("  Saving!!")
    else:
        print("  An already existing weight for autoencoder will be loaded.")

    return s, d, net
