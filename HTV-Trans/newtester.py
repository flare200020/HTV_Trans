import torch
import numpy as np
import torch.optim as optim
from metrics import All_Metrics
from logger import get_logger


class Tester(object):
    def __init__(self, model,  testloader, log_path='log_tester', log_file='loss', device=torch.device('cpu'),
                 learning_rate=0.0002, nsamples=None, sample_path=None, checkpoints=None):
        self.model = model
        self.model.to(device)
        self.device = device
        # self.test = test
        self.testloader = testloader
        self.log_path = log_path
        self.log_file = log_file
        self.learning_rate = learning_rate
        self.nsamples = nsamples
        self.sample_path = sample_path
        self.checkpoints = checkpoints
        self.start_epoch = 0
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.epoch_losses = []
        # self.logger = Logger(self.log_path, self.log_file)
        self.loss = {}
        self.logger = get_logger(self.log_path, name=None, debug=False)
        self.iter = 0

    def load_checkpoint(self, start_ep):
        try:
            print("Loading Chechpoint from ' {} '".format(self.checkpoints + '_epochs{}.pth'.format(start_ep)))
            checkpoint = torch.load(self.checkpoints + '_epochs{}.pth'.format(start_ep))
            self.start_epoch = checkpoint['epoch']
            self.model.beta = checkpoint['beta']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
            return 1

        except:
            print("No Checkpoint Exists At '{}', Starting Fresh Training".format(
                self.checkpoints + '_epochs{}.pth'.format(start_ep)))
            self.start_epoch = 0
            return 0

    def model_test_v2(self):
        self.model.eval()
        mae_list = []
        mse_list = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.testloader):
                dec_inp = torch.zeros_like(batch_y[:, -self.model.horizon:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.model.label_len, :], dec_inp], dim=1).float().to(self.device)

                data, label = batch_x, batch_y[:,-self.model.horizon:,:]
                data = data.unsqueeze(3)
                label = label.unsqueeze(3)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                self.optimizer.zero_grad()


                x_mu, x_var, _, kl_loss, output = self.forward_test(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if(self.model.decoder_type != 'fcn'):
                    label = label.squeeze(-1)

                mae, rmse, mape, mse, _ = All_Metrics(output, label.to(self.device), None, 0.)
                mae_list.append(mae.cpu())
                mse_list.append(mse.cpu())

            mae_list = torch.tensor(np.array(mae_list)).to(self.device)
            mse_list = torch.tensor(np.array(mse_list)).to(self.device)

            print('testing!!!!!!!!')
            mae = torch.mean(mae_list)
            mse = torch.mean(mse_list)
            self.logger.info("Average Result, MAE: {:.4f}, MSE:{:.4f}".format(mae, mse))

            print("Testing is complete!")


    def forward_test(self, batch_x, batch_x_mark, dec_inp,batch_y_mark):
        with torch.no_grad():

           x_mu, x_var, llh, kl_loss, h_out = self.model(batch_x, batch_x_mark, dec_inp,batch_y_mark)
        return x_mu, x_var, llh, kl_loss, h_out


    def loglikelihood_last_timestamp(self, x, recon_x_mu, recon_x_logsigma):
        llh = -0.5 * torch.sum(torch.pow(((x.float() - recon_x_mu.float()) / torch.exp(recon_x_logsigma.float())),
                                         2) + 2 * recon_x_logsigma.float() + np.log(np.pi * 2))
        return llh
