import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
import numpy as np
from CC.ICCStandard import ITrainer
from CC.model import AutoModel
from CC.loader import AutoDataloader
from CC.analysis import Analysis
from tqdm import tqdm


class Trainer(ITrainer):

    def __init__(self, tokenizer, loader_name, data_path, model_name="bert",  model_type="interactive", from_pretrained=None, data_present_path="./dataset/present.json", padding_length=50, batch_size=16, batch_size_eval=64, eval_mode='dev', task_name='SAS'):
        self.loader_name = loader_name
        self.model_name = model_name
        self.model_type = model_type
        self.from_pretrained = from_pretrained
        self.data_path = data_path
        self.task_name = task_name
        self.padding_length = padding_length
        self.dataloader_init(tokenizer, loader_name, data_path, model_type,
                             data_present_path, padding_length, batch_size, batch_size_eval, eval_mode)
        self.model_init(tokenizer, model_name)
        self.analysis = Analysis()

    def model_init(self, tokenizer, model_name):
        print('AutoModel Choose Model: {}\n'.format(model_name))
        a = AutoModel(tokenizer, model_name, self.from_pretrained)
        self.model = a()

    def dataloader_init(self, tokenizer, loader_name, data_path, model_type, data_present_path, padding_length, batch_size, batch_size_eval, eval_mode):
        d = AutoDataloader(tokenizer, loader_name, data_path,
                           model_type, data_present_path, padding_length)
        self.train_loader, self.eval_loader = d(
            batch_size, batch_size_eval, eval_mode)

    def __call__(self, resume_path=None, resume_step=None, num_epochs=30, lr=5e-5, fct_loss='MSELoss', gpu=[0, 1, 2, 3], eval_call_epoch=None):
        return self.train(resume_path=resume_path, resume_step=resume_step,
                          num_epochs=num_epochs, lr=lr, fct_loss=fct_loss, gpu=gpu, eval_call_epoch=eval_call_epoch)

    def train(self, resume_path=None, resume_step=None, num_epochs=30, lr=5e-5, fct_loss='MSELoss', gpu=[0, 1, 2, 3], eval_call_epoch=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()

        if resume_path is not None:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.module.load_state_dict(model_dict)
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)

        current_uid = str(uuid.uuid1()).split('-')[0]

        train_step = resume_step if resume_step is not None else 0
        for epoch in range(num_epochs):
            train_count = 0
            train_loss = 0
            train_R = []
            train_RMSE = []
            train_error = []
            train_pearson = []
            train_spearman = []
            X = []
            Y = []

            train_iter = tqdm(self.train_loader)
            self.model.train()

            for it in train_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                it['padding_length'] = int(self.padding_length / 2)
                it['fct_loss'] = fct_loss
                if fct_loss in ['BCELoss', 'MSELoss']:
                    it['labels'] = it['labels'].float()

                model_output = self.model(**it)
                loss, pred = model_output['loss'], model_output['pred']
                if 'scores' in model_output.keys():
                    scores = model_output['scores']
                else:
                    scores = None
                loss = loss.mean()

                loss.backward()
                scheduler.step()
                optimizer.step()
                self.model.zero_grad()

                train_loss += loss.data.item()
                train_count += 1
                train_step += 1

                gold = it['labels']
                X += pred.tolist()
                Y += gold.tolist()
                r, r_mse, pearsonr, spearmanr = self.analysis.evaluationSAS(
                    X, Y)
                train_error.append((torch.abs(pred - gold)).mean().data.item())
                train_R.append(r)
                train_RMSE.append(r_mse)
                train_pearson.append(pearsonr)
                train_spearman.append(spearmanr)

                train_iter.set_description(
                    'Train: {}/{}'.format(epoch + 1, num_epochs))
                if scores is not None:
                    train_iter.set_postfix(
                        train_loss=train_loss / train_count, R=r, RMSE=r_mse, Pearson=pearsonr, Spearman=spearmanr, train_error=np.mean(train_error), scoresA=np.mean(scores[0].tolist()), scoresB=np.mean(scores[1].tolist()), scoresC=np.mean(scores[2].tolist()), scoresD=np.mean(scores[3].tolist()))
                else:
                    train_iter.set_postfix(
                        train_loss=train_loss / train_count, R=r, RMSE=r_mse, Pearson=pearsonr, Spearman=spearmanr, train_error=np.mean(train_error))

            if scores is not None:
                self.analysis.append_train_record({
                    'epoch': epoch + 1,
                    'train_loss': train_loss / train_count,
                    'R': np.mean(train_R),
                    'RMSE': np.mean(train_RMSE),
                    'pearson': np.mean(train_pearson),
                    'spearman': np.mean(train_spearman),
                    'train_error': np.mean(train_error),
                    'scoresA': np.mean(scores[0].tolist()),
                    'scoresB': np.mean(scores[1].tolist()),
                    'scoresC': np.mean(scores[2].tolist()),
                    'scoresD': np.mean(scores[3].tolist())
                })
            else:
                self.analysis.append_train_record({
                    'epoch': epoch + 1,
                    'train_loss': train_loss / train_count,
                    'R': np.mean(train_R),
                    'RMSE': np.mean(train_RMSE),
                    'pearson': np.mean(train_pearson),
                    'spearman': np.mean(train_spearman),
                    'train_error': np.mean(train_error)
                })

            model_uid = self.save_model(train_step)
            if eval_call_epoch is None or eval_call_epoch(epoch):
                self.eval(epoch)

            self.analysis.save_all_records(
                uid=current_uid if self.task_name is None else self.task_name)
            yield (epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, model_uid)

    def save_model(self, current_step=0):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        if not os.path.exists('./save_model/{}/{}'.format(dir, self.model_name)):
            os.makedirs('./save_model/{}/{}'.format(dir, self.model_name))
        torch.save(
            self.model, './save_model/{}/{}/{}_{}.pth'.format(dir, self.model_name, self.model_name, current_step))
        self.analysis.append_model_record(current_step)
        return current_step

    def eval(self, epoch):
        with torch.no_grad():
            eval_count = 0
            eval_loss = 0
            eval_R = []
            eval_RMSE = []
            eval_error = []
            eval_pearson = []
            eval_spearman = []
            X = []
            Y = []

            eval_iter = tqdm(self.eval_loader)
            self.model.eval()

            for it in eval_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                it['padding_length'] = int(self.padding_length / 2)

                model_output = self.model(**it)
                loss, pred = model_output['loss'], model_output['pred']
                if 'scores' in model_output.keys():
                    scores = model_output['scores']
                else:
                    scores = None
                loss = loss.mean()

                eval_loss += loss.data.item()
                eval_count += 1

                gold = it['labels']
                X += pred.tolist()
                Y += gold.tolist()
                r, r_mse, pearsonr, spearmanr = self.analysis.evaluationSAS(
                    X, Y)
                eval_error.append((torch.abs(pred - gold)).mean().data.item())
                eval_R.append(r)
                eval_RMSE.append(r_mse)
                eval_pearson.append(pearsonr)
                eval_spearman.append(spearmanr)

                eval_iter.set_description(
                    f'Eval: {epoch + 1}')
                if scores is not None:
                    eval_iter.set_postfix(
                        eval_loss=eval_loss / eval_count, R=r, RMSE=r_mse, Pearson=pearsonr, Spearman=spearmanr, eval_error=np.mean(eval_error), scoresA=np.mean(scores[0].tolist()), scoresB=np.mean(scores[1].tolist()), scoresC=np.mean(scores[2].tolist()), scoresD=np.mean(scores[3].tolist()))
                else:
                    eval_iter.set_postfix(
                        eval_loss=eval_loss / eval_count, R=r, RMSE=r_mse, Pearson=pearsonr, Spearman=spearmanr, eval_error=np.mean(eval_error))

            if scores is not None:
                self.analysis.append_eval_record({
                    'epoch': epoch + 1,
                    'eval_loss': eval_loss / eval_count,
                    'R': np.mean(eval_R),
                    'RMSE': np.mean(eval_RMSE),
                    'pearson': np.mean(eval_pearson),
                    'spearman': np.mean(eval_spearman),
                    'eval_error': np.mean(eval_error),
                    'scoresA': np.mean(scores[0].tolist()),
                    'scoresB': np.mean(scores[1].tolist()),
                    'scoresC': np.mean(scores[2].tolist()),
                    'scoresD': np.mean(scores[3].tolist())
                })
            else:
                self.analysis.append_eval_record({
                    'epoch': epoch + 1,
                    'eval_loss': eval_loss / eval_count,
                    'R': np.mean(eval_R),
                    'RMSE': np.mean(eval_RMSE),
                    'pearson': np.mean(eval_pearson),
                    'spearman': np.mean(eval_spearman),
                    'eval_error': np.mean(eval_error)
                })

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
