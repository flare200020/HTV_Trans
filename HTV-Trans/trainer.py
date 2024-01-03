import torch
import numpy as np
import torch.optim as optim
from logger import Logger
import time
from tools import EarlyStopping
from metrics import All_Metrics

class Trainer(object):
    def __init__(self, vagt, trainloader, valloader, log_path='log_trainer', log_file='loss', epochs=20,
                 batch_size=1024, learning_rate=0.001, coeff=0.2, patience=5, checkpoints='kpi_model.path', checkpoints_interval=1,
                 device=torch.device('cuda:0')):
        """
        VRNN is well trained at this moment when training VAGT
        :param vrnn:
        :param vagt:
        :param train:
        :param trainloader:
        :param log_path:
        :param log_file:
        :param epochs:
        :param batch_size:
        :param learning_rate:
        :param checkpoints:
        :param checkpoints_interval:
        :param device:
        """
        self.trainloader = trainloader
        self.valloader = valloader
        # self.train = train
        self.log_path = log_path
        self.log_file = log_file
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.vagt = vagt
        self.vagt.to(device)
        self.learning_rate = learning_rate
        self.coeff = coeff
        self.checkpoints = checkpoints
        self.checkpoints_interval = checkpoints_interval
        print('Model parameters: {}'.format(self.vagt.parameters()))
        self.optimizer = optim.Adam(self.vagt.parameters(), self.learning_rate) #changed
        self.epoch_losses = []
        self.loss = {}
        self.logger = Logger(self.log_path, self.log_file)
        self.patience = patience



    def save_checkpoint(self, epoch, checkpoints):
        torch.save({'epoch': epoch + 1,
                    'beta': self.vagt.beta,
                    'state_dict': self.vagt.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'losses': self.epoch_losses},
                    checkpoints + '_epochs{}.pth'.format(epoch+1))

    def load_checkpoint(self, start_ep, checkpoints):
        try:
            print("Loading Chechpoint from ' {} '".format(checkpoints+'_epochs{}.pth'.format(start_ep)))
            checkpoint = torch.load(checkpoints+'_epochs{}.pth'.format(start_ep))
            self.start_epoch = checkpoint['epoch']
            self.vagt.beta = checkpoint['beta']
            self.vagt.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
            self.start_epoch = 0
        except:
            print("No Checkpoint Exists At '{}', Starting Fresh Training".format(checkpoints))
            self.start_epoch = 0

    def model_val(self):
        self.vagt.eval()
        mae_list = []
        mse_list = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.valloader):
                dec_inp = torch.zeros_like(batch_y[:, -self.vagt.horizon:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.vagt.label_len, :], dec_inp], dim=1).float().to(self.device)
                data, label = batch_x, batch_y[:,-self.vagt.horizon:,:]
                data = data.unsqueeze(3)
                label = label.unsqueeze(3)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_size = data.size(0)
                data = data.to(self.device)
                self.optimizer.zero_grad()
                x_mu, x_var, llh, kl_loss, output = self.vagt(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if(self.vagt.decoder_type != 'fcn'):
                    label = label.squeeze(-1)

                mae, rmse, mape, mse, _ = All_Metrics(output, label.to(self.device), None, 0.)
                mae_list.append(mae.cpu())
                mse_list.append(mse.cpu())
            mae_list = torch.tensor(np.array(mae_list)).to(self.device)
            mse_list = torch.tensor(np.array(mse_list)).to(self.device)
            mae = torch.mean(mae_list)
            mse = torch.mean(mse_list)

        return mae



    def train_model(self):
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        for epoch in range(self.start_epoch, self.epochs):
            self.vagt.train()
            since = time.time()
            losses = []
            llhs = []
            kld_zs = []
            task_loss = []
            print("Running Epoch : {}".format(epoch + 1))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.trainloader):
                dec_inp = torch.zeros_like(batch_y[:, -self.vagt.horizon:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.vagt.label_len, :], dec_inp], dim=1).float().to(self.device)

                data, label = batch_x.float(), batch_y[:, -self.vagt.horizon:, :].float()
                data = data.unsqueeze(3)
                label = label.unsqueeze(3)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_size = data.size(0)
                data = data.to(self.device)
                self.optimizer.zero_grad()


                x_mu, x_var, llh, kl_loss, h_out = self.vagt(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if (self.vagt.decoder_type != 'fcn'):
                    label = label.squeeze(-1)

                mae_loss = self.vagt.mae_loss(h_out.to(self.device), label.to(self.device))
                loss = self.coeff * (-llh / batch_size + self.vagt.beta * kl_loss / batch_size) + mae_loss
                loss.backward()
                self.optimizer.step()


                losses.append(loss.item())
                llhs.append(llh.item()/batch_size)
                kld_zs.append(kl_loss.item()/batch_size)
                task_loss.append(mae_loss.item())


            meanloss = np.mean(losses)
            meanllh = np.mean(llhs)
            meanz = np.mean(kld_zs)
            meantask = np.mean(task_loss)
            self.epoch_losses.append(meanloss)

            val_loss = self.model_val()
            early_stopping(val_loss, self.vagt, self.optimizer, task_loss, epoch, self.checkpoints)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            print("Epoch {} : Average Loss: {} Loglikelihood: {} KL of z: {}, Beta: {}, taskloss: {}".format(
                epoch + 1, meanloss, -meanllh, meanz, self.vagt.beta, meantask))
            self.loss['Epoch'] = epoch + 1
            self.loss['Avg_loss'] = meanloss
            self.loss['Llh'] = meanllh
            self.loss['KL_z'] = meanz
            self.loss['task'] = meantask
            self.logger.log_trainer(epoch + 1, self.loss)



            if (epoch + 1) % 1 == 0:
                self.vagt.beta = np.minimum((self.vagt.beta + 0.1) * np.exp(self.vagt.anneal_rate * (epoch + 1)),
                                             self.vagt.max_beta)
            time_elapsed = time.time() - since
            print('Complete in {:.2f}m {:.2f}s'.format(time_elapsed // 60, time_elapsed % 60))


        print("Training is complete!")
