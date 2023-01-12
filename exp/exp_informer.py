from torch.utils.data.dataset import Dataset
from data.data_loader import *
from exp.exp_basic import Exp_Basic
from models.model import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator #导入此类，设置坐标轴间隔
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric,RMSE,NMAE

from pytorchsummary.torchsummary import summary
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag , train_data_flag):
        args = self.args

        data_dict = {
            'Dataset_Energy_0_24_predict_6_18':Dataset_Energy_0_24_predict_6_18, #只預測6-18點(用0-23點資料)
            'Dataset_Energy_0_24_predict_0_24':Dataset_Energy_0_24_predict_0_24, #0-23點全預測(用0-23點資料)
            'Dataset_Energy_6_18_predict_6_18':Dataset_Energy_6_18_predict_6_18, #6-18點全預測(用6-18點資料)
            'Dataset_Energy_0_24_yunlin_predict_6_18':Dataset_Energy_0_24_yunlin_predict_6_18, #6-18點全預測(用6-18點資料)
            'Dataset_Energy_0_24_chiayi_predict_6_18':Dataset_Energy_0_24_chiayi_predict_6_18, #6-18點全預測(用6-18點資料)
            'Dataset_Energy_Morning_North_0_24_predict_6_18':Dataset_Energy_Morning_North_0_24_predict_6_18, #指預測6-18點(用0-23點資料)
            'Dataset_Energy_Morning_North_6_18_predict_6_18':Dataset_Energy_Morning_North_6_18_predict_6_18, #指預測6-18點(用0-23點資料)
        }

        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = train_data_flag; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            #inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            #cols=args.cols
        )

        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting,args):
        torch.cuda.set_device(args.gpu)
        train_data, train_loader = self._get_data(flag = 'train',train_data_flag=True)
        vali_data, vali_loader = self._get_data(flag = 'val',train_data_flag=False)
        test_data, test_loader = self._get_data(flag = 'test',train_data_flag=False)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()



        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()




        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            #LSTM需要的資料
            preds = [] #儲存預測值
            trues = [] #儲存真實值
            target_diff = [] #預測-實際 誤差值
            target_preds = [] #預測-實際 預測值

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

            
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                target_diff.append(pred.detach().cpu().numpy()-true.detach().cpu().numpy())


                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward(retain_graph=True)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

            #LSTM需要的資料
            preds = np.array(preds)
            trues = np.array(trues)
            target_diff = np.array(target_diff)
            print('train shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            target_diff = target_diff.reshape(-1, trues.shape[-2], trues.shape[-1])
    


        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    #使用train_data測試informer模型和訓練LSTM預測誤差
    def test_informer_and_LSTM_predict_error(self, setting ,args,flag):
        test_data, test_loader = self._get_data(flag = flag,train_data_flag=False)
        self.model.eval()
        
        preds = [] #informer預測值
        trues = [] #informer實際值
        target_diff = [] #預測-實際 誤差值
        target_preds = [] #預測-實際 預測值
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
        
            target_diff.append(pred.detach().cpu().numpy()-true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        target_diff = np.array(target_diff)
        print('informer test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        target_diff = target_diff.reshape(-1, trues.shape[-2], trues.shape[-1])
        #將夜間時段補0(是用於預測0-23點全部資料)
        
        if args.data == 'Dataset_Energy_0_24_predict_0_24':
            count_night = 0
            for i in range(len(preds)):
                #預測出來0以下直接改成0
                if preds[int(i)] < 0:
                    preds[int(i)] = 0.
                #將夜間時段改成0
                if i < 6+count_night or i >= 19+count_night:
                    preds[int(i)] = 0.
                    trues[int(i)] = 0.
                    if i == 19+count_night:
                        count_night += 24
                #print(i+2690,'-',preds[int(i)],'-',trues[int(i)])
        
        
        elif args.data == 'Dataset_Energy_0_24_predict_6_18'  or args.data == 'Dataset_Energy_6_18_predict_6_18' or args.data == 'Dataset_Energy_Morning_North_0_24_predict_6_18' or args.data == 'Dataset_Energy_Morning_North_6_18_predict_6_18' or args.data == 'Dataset_Energy_0_24_yunlin_predict_6_18' or args.data == 'Dataset_Energy_0_24_chiayi_predict_6_18':
            #將夜間時段補0(是用於預測6-18點全部資料)
            tmp_preds = list()
            tmp_trues = list()
            tmp_diff = list()
            for i in range(len(preds)):
                #預測出來0以下直接改成0
                if preds[int(i)] < 0:
                    preds[int(i)] = 0.
            for i in range(0,len(preds),13):
                #將夜間時段補0
                tmp_diff.extend([[0],[0],[0],[0],[0],[0]])
                tmp_diff.extend(target_diff[i:i+13])
                tmp_diff.extend([[0],[0],[0],[0],[0]])
                tmp_preds.extend([[0],[0],[0],[0],[0],[0]])
                tmp_preds.extend(preds[i:i+13])
                tmp_preds.extend([[0],[0],[0],[0],[0]])
                tmp_trues.extend([[0],[0],[0],[0],[0],[0]])
                tmp_trues.extend(trues[i:i+13])
                tmp_trues.extend([[0],[0],[0],[0],[0]])
                #print(i,'-',preds[int(i)],'-',trues[int(i)])

            
            preds = np.array(tmp_preds).astype('float64')
            trues = np.array(tmp_trues).astype('float64')
            target_diff = np.array(tmp_diff).astype('float64')
        


        

        

        print('test shape:', preds.shape, trues.shape)

        #LSTM設定資料集
        target_diff = torch.tensor(target_diff.reshape(len(target_diff),1)).float()
        data = Target_Loss_Data(target_diff.cuda())

        if flag == 'train':
        
            target_train_loader = DataLoader(
            data,
            batch_size=64,
            shuffle=False,
            drop_last=True)

            num_epochs = 50
            learning_rate = 0.01

            input_size = 1
            hidden_size = 1
            num_layers = 1

            num_classes = 1


            
            lstm = LSTM(num_classes,input_size, hidden_size, num_layers).cuda()

            criterion_lstm = torch.nn.MSELoss()   # mean-squared error for regression
            optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
            # Train the model
            lstm_train_loss = list()
            lstm.train()
            for epoch in range(num_epochs):

                for (trainX,trainY) in target_train_loader:
                    outputs = lstm(trainX)
                    optimizer_lstm.zero_grad()
                    # obtain the loss function
                    loss_lstm = criterion_lstm(outputs, trainY)
                    loss_lstm.backward()
                    optimizer_lstm.step()
                print("Epoch: %d, loss: %1.5f" % (epoch, loss_lstm.item()))
                lstm_train_loss.append(loss_lstm.item())
                
            torch.save(lstm, './lstm_best_model/lstm.pth')

            plt.plot(lstm_train_loss)
            plt.savefig('./lstm_best_model/lstm_train_lost.png')
        if flag == 'test':
            lstm = torch.load('./lstm_best_model/lstm.pth')
            lstm.eval()
            
            target_test_loader = DataLoader(
                data,
                batch_size=1,
                shuffle=False,
                drop_last=True)


            for (trainX,trainY) in target_test_loader:
                #print(trainX.shape)
                outputs = lstm(trainX)
                target_preds.append(outputs.detach().cpu().numpy())
                

            target_preds = np.hstack([np.zeros(24),np.array(target_preds).squeeze()])
            pred_sub_predict_error = np.add(preds.squeeze(),target_preds.squeeze())
            start = 6
            end = 19
            timestmap_24 = 0
            for i in range(len(pred_sub_predict_error)):
                if pred_sub_predict_error[i] < 0:
                    pred_sub_predict_error[i] = 0
                if ( i < timestmap_24 + start or i >= timestmap_24 + end):
                    pred_sub_predict_error[i] = 0
                    if i == timestmap_24 + end:
                        timestmap_24 += 24
                    
            #畫圖
            plt.gca().set_prop_cycle(color=['red','blue','green'])
            x_major_locator=MultipleLocator(48)
            plt.gca().xaxis.set_major_locator(x_major_locator)
            plt.xlim(0,240)
            plt.xlabel("time",fontsize=18)
            plt.ylabel("power",fontsize=18)
            plt.plot(preds.squeeze(),label='Prediction')
            plt.plot(trues.squeeze(),label='Ground-Truth')
            plt.plot(pred_sub_predict_error,label='Prediction + Predict_Error')
            #plt.plot(target_diff.squeeze(),label='Diff-Truth')
            #plt.plot(target_preds.squeeze(),label='Diff-Predict')
            plt.legend()
            img_path = './img/' + args.data_path.split('.')[0] + '/'
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            plt.savefig(img_path + args.data_path.split('.')[0] +'.png')
  

        

        

            # result save
            folder_path = './results/' + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)



            preds[preds == 0] = 'nan'
            trues[trues == 0] = 'nan'
            pred_sub_predict_error[pred_sub_predict_error == 0 ] = 'nan'

            print(trues[0:24])
            print(pred_sub_predict_error[0:24])
            rmse = RMSE(preds, trues)
            nmae = NMAE(preds, trues)
            rmse_pred_sub_predict_error = RMSE(pred_sub_predict_error, trues.squeeze())
            nmae_pred_sub_predict_error = NMAE(pred_sub_predict_error, trues.squeeze())
            print('nmae:{}, rmse:{}'.format(nmae, rmse))
            print('nmae_plus_predict_error:{}, rmse_plus_predict_error:{}'.format(nmae_pred_sub_predict_error, rmse_pred_sub_predict_error))


        

            np.save(folder_path+'metrics.npy', np.array([ nmae, rmse]))
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)



        
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred',train_data_flag=False)
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()


        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y