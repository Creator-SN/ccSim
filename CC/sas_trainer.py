import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from CC.ICCStandard import ITrainer
from CC.model import AutoModel
from CC.loader import AutoDataloader
from CC.analysis import Analysis
from tqdm import tqdm
from torch.autograd import Variable

class Trainer(ITrainer):

    def __init__(self, tokenizer, model_name, dataset_name, model_type="bert", padding_length=50, batch_size=16, batch_size_eval=64, fit_sample=False):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.padding_length = padding_length
        self.dataloader_init(tokenizer, dataset_name, model_type, padding_length, batch_size, batch_size_eval, fit_sample=fit_sample)
        self.model_init(tokenizer, model_name)
    
    def model_init(self, tokenizer, model_name):
        print('AutoModel Choose Model: {}\n'.format(model_name))
        a = AutoModel(tokenizer, model_name)
        self.model = a()

    def dataloader_init(self, tokenizer, data_name, model_type="bert", padding_length=50, batch_size=16, batch_size_eval=64, fit_sample=False):
        d = AutoDataloader(tokenizer, data_name, model_type, padding_length)
        if "weakly" not in model_type:
            self.train_loader, self.eval_loader = d(batch_size, batch_size_eval)
        else:
            self.train_loader, self.eval_loader, self.fit_loader = d(batch_size, batch_size_eval, fit_sample=fit_sample)
    
    def __call__(self, resume_path=False, num_epochs=30, lr=5e-5, gpu=[0, 1, 2, 3], score_fitting=False):
        self.train(resume_path, num_epochs, lr, gpu, score_fitting)

    def train(self, resume_path=False, num_epochs=30, lr=5e-5, gpu=[0, 1, 2, 3], score_fitting=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()

        if not resume_path == False:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.module.load_state_dict(model_dict)
        self.model.to(device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.)
        
        Epoch_loss = []
        Epoch_error = []
        Epoch_R = []
        Epoch_RMSE = []

        Epoch_loss_eval = []
        Epoch_error_eval = []
        Epoch_R_eval = []
        Epoch_RMSE_eval = []
        for epoch in range(num_epochs):
            train_count = 0
            train_loss = []
            X = []
            Y = []
            Epoch_X_eval = []
            Epoch_Y_eval = []
            train_error = []
            if not score_fitting:
                train_iter = tqdm(self.train_loader)
            else:
                train_iter = tqdm(self.fit_loader)
            self.model.train()
            for sentences, mask, token_type_ids, labels in train_iter:
                sentences = self.cuda(sentences)
                mask = self.cuda(mask)
                token_type_ids = self.cuda(token_type_ids)
                labels = self.cuda(labels)
                self.model.zero_grad()
                
                loss, pred = self.model(sentences=sentences, attention_mask=mask, token_type_ids=token_type_ids, labels=labels, padding_length=self.padding_length)
                loss = loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.append(loss.data.item())

                X += labels.tolist()
                Y += pred.tolist()
                r, r_mse, p_, s_ = Analysis.Evaluation(X, Y)
                train_error.append((torch.abs(pred - labels)).mean().data.item())
                train_count += 1

                train_iter.set_description('Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(train_loss=np.mean(train_loss), R=r, RMSE=r_mse, Pearson=p_, Spearman=s_, train_error=np.mean(train_error))
            
            Epoch_loss.append(np.mean(train_loss))
            Epoch_error.append(np.mean(train_error))
            Epoch_R.append(r)
            Epoch_RMSE.append(r_mse)
            
            if score_fitting == False:
                _dir = './log/{}/{}'.format(self.dataset_name, self.model_type)
            else:
                _dir = './log/{}/{}'.format(self.dataset_name, self.model_type + '_fit')
            Analysis.save_same_row_list(_dir, 'train_log', loss=Epoch_loss, error=Epoch_error, R=Epoch_R, RMSE=Epoch_RMSE)
            if resume_path == False:
                self.save_model(epoch, 0, score_fitting)
            else:
                self.save_model(epoch, int(resume_path.split('/')[-1].split('_')[1].split('.')[0]), score_fitting)
            
            A, B, C, D, E, F = self.eval(epoch, num_epochs)
            Epoch_X_eval += A
            Epoch_Y_eval += B
            Epoch_loss_eval.append(C)
            Epoch_error_eval.append(D)
            Epoch_R_eval.append(E)
            Epoch_RMSE_eval.append(F)
            Analysis.save_xy(Epoch_X_eval, Epoch_Y_eval, _dir)
            Analysis.save_same_row_list(_dir, 'eval_log', loss=Epoch_loss_eval, error=Epoch_error_eval, R=Epoch_R_eval, RMSE=Epoch_RMSE_eval)

    def save_model(self, epoch, save_offset=0, score_fitting=False):
        if score_fitting == False:
            _dir = './model/{}/{}'.format(self.dataset_name, self.model_type)
        else:
            _dir = './model/{}/{}'.format(self.dataset_name, self.model_type + '_fit')
        if not os.path.isdir(_dir):
            os.makedirs(_dir)
        torch.save(self.model, '{}/epoch_{}.pth'.format(_dir, epoch + 1 + save_offset))

    def eval(self, epoch, num_epochs):
        with torch.no_grad():
            eval_loss = []
            X = []
            Y = []
            eval_count = 0
            eval_error = []
            self.model.eval()
            eval_iter = tqdm(self.eval_loader)
            for sentences, mask, token_type_ids, labels in eval_iter:
                self.model.eval()
                sentences = self.cuda(sentences)
                mask = self.cuda(mask)
                token_type_ids = self.cuda(token_type_ids)
                labels = self.cuda(labels)

                loss, pred = self.model(sentences, attention_mask=mask, token_type_ids=token_type_ids, labels=labels, padding_length=self.padding_length)
                loss = loss.mean()

                eval_loss.append(loss.sum().data.item())

                X += labels.tolist()
                Y += pred.tolist()
                r, r_mse, p_, s_ = Analysis.Evaluation(X, Y)
                eval_error.append((torch.abs(pred - labels)).mean().data.item())
                eval_count += 1
                
                eval_iter.set_description('Eval: {}/{}'.format(epoch + 1, num_epochs))
                eval_iter.set_postfix(eval_loss=np.mean(eval_loss), R=r, RMSE=r_mse, Pearson=p_, Spearman=s_, eval_avg=np.mean(eval_error))
            
            return X, Y, np.mean(eval_loss), np.mean(eval_error), r, r_mse
        
    def cuda(self, inputX):
        if type(inputX) == tuple:
            if torch.cuda.is_available():
                result = []
                for item in inputX:
                    result.append(item.cuda())
                return result
            return inputX
        else:
            if torch.cuda.is_available():
                return inputX.cuda()
            return inputX