from models import EnergyGPT
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        if hasattr(args, 'static_weight_path') and args.static_weight_path:
            self.static_weights = pd.read_csv(os.path.join(args.root_path, args.static_weight_path)).values
        else:
            self.static_weights = None

    def _build_model(self):
        model_dict = {
            'EnergyGPT': EnergyGPT
        }

        self.models = nn.ModuleList([
            model_dict[self.args.model].Model(self.args).float() for _ in range(4)
        ])

        if self.args.use_multi_gpu and self.args.use_gpu:
            self.models = nn.ModuleList([nn.DataParallel(model, device_ids=self.args.device_ids) for model in self.models])

        return self.models
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, model):
        model_optim = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        return model_optim
                                                                  
    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        return criterion


    def vali(self, vali_data, vali_loader, criterion, is_test=True):
        total_loss = 0
        weights = [0.25, 0.25, 0.25, 0.25] 

        for model in self.models:
            model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                weighted_losses = []
                for idx, model in enumerate(self.models):
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, model_idx=idx, static_weights=self.static_weights)
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, model_idx=idx, static_weights=self.static_weights)

                    outputs = outputs[:, -self.args.pred_len:, idx:idx+1]
                    target = batch_y[:, -self.args.pred_len:, idx:idx+1].to(self.device)
                    loss = criterion(outputs, target)
                    weighted_losses.append(loss * weights[idx])

                total_weighted_loss = sum(weighted_losses)
                total_loss += total_weighted_loss.item()

        total_loss /= len(vali_loader)
        for model in self.models:
            model.train()
        return total_loss


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        optimizers = [self._select_optimizer(model) for model in self.models]
        criterion = self._select_criterion()
        scaler = GradScaler()  
                
        self.warmup_epochs = self.args.warmup_epochs
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        weights = [1, 1, 1, 1]       
        import math

        def adjust_learning_rate_new(optimizer, epoch, args):
            """Decay the learning rate with half-cycle cosine after warmup"""
            min_lr = 0
            if epoch < self.warmup_epochs:
                lr = self.args.learning_rate * epoch / self.warmup_epochs 
            else:
                lr = min_lr+ (self.args.learning_rate - min_lr) * 0.5 * \
                    (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.args.train_epochs - self.warmup_epochs)))
                
            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr
            print(f'Updating learning rate to {lr:.7f}')
            return lr



        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            iter_count = 0
            train_loss = []

            for optimizer in optimizers:
                adjust_learning_rate_new(optimizer, epoch+1, self.args)

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):

                for optimizer in optimizers:
                    optimizer.zero_grad()

                total_loss = 0
                for idx, (model, optimizer) in enumerate(zip(self.models, optimizers)):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    with autocast():
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, model_idx=idx, static_weights=self.static_weights)
                        outputs = outputs[:, -self.args.pred_len:, idx:idx+1]

                    target = batch_y[:, -self.args.pred_len:, idx:idx+1].to(self.device)    

                    loss = criterion(outputs, target)
                    weighted_loss = loss * weights[idx]
                    total_loss += weighted_loss 

                scaler.scale(total_loss).backward()

                for optimizer in optimizers:
                    scaler.step(optimizer)
                    optimizer.zero_grad()  

                scaler.update()  
                iter_count += 1  


                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {total_loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, is_test=False)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {len(train_loader)} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                model_preds = []
                model_trues = []
                for idx, model in enumerate(self.models):
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, model_idx=idx, static_weights=self.static_weights)
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, model_idx=idx, static_weights=self.static_weights)

                    outputs = outputs[:, -self.args.pred_len:, idx:idx+1]
                    target = batch_y[:, -self.args.pred_len:, idx:idx+1].to(self.device)

                    model_preds.append(outputs.detach().cpu().numpy())
                    model_trues.append(target.detach().cpu().numpy())

                preds.append(np.concatenate(model_preds, axis=-1))
                trues.append(np.concatenate(model_trues, axis=-1))
                inputx.append(batch_x.detach().cpu().numpy())

        np.save(os.path.join(folder_path, 'predictions.npy'), np.array(preds))
        np.save(os.path.join(folder_path, 'true_values.npy'), np.array(trues))

        inputx = np.concatenate(inputx, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)

        np.save(os.path.join(folder_path, 'inputx.npy'), inputx)
        np.save(os.path.join(folder_path, 'trues.npy'), trues)
        np.save(os.path.join(folder_path, 'preds.npy'), preds)

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for i in range(preds.shape[-1]):  
            mae, mse, rmse, mape, mspe, r2, rse, corr= metric(preds[:, :, i], trues[:, :, i])
            print("-----------preds[:, :, i]-------------",preds[:, :, i].shape)
            print(f'Feature {i}: mse:{mse}, mae:{mae}, rmse:{rmse} , mape:{mape},  mspe:{mspe},r2:{r2}, rse:{rse}, corr:{corr}')


        mae, mse, rmse, mape, mspe, r2, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, mape:{}, rmse:{}, mspe:{},r2:{}, rse:{}, corr:{}'.format(mse, mae, mape, rmse, mspe,r2, rse, corr))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, mape:{}, rmse:{}, mspe:{},r2:{}, rse:{}, corr:{}'.format(mse, mae, mape, rmse, mspe,r2, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mse, mae, mape, rmse, mspe,r2, rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)
        return
    
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)  
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                model_preds = []
                for idx, model in enumerate(self.models):
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, static_weights=self.static_weights)
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, static_weights=self.static_weights)

                    outputs = outputs[:, -self.args.pred_len:, idx:idx+1]
                    model_preds.append(outputs.detach().cpu().numpy())

                preds.append(np.concatenate(model_preds, axis=-1))

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return