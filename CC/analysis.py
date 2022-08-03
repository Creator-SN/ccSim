import os
import datetime
import numpy as np
from scipy.stats import pearsonr, spearmanr
from CC.ICCStandard import IAnalysis


class Analysis(IAnalysis):

    def __init__(self):
        self.train_record = {}
        self.eval_record = {}
        self.model_record = {}

    '''
    append data record of train
    train_record_item: dict
    '''

    def append_train_record(self, train_record_item):
        for key in train_record_item:
            if key not in self.train_record:
                self.train_record[key] = []
            self.train_record[key].append(train_record_item[key])

    '''
    append data record of eval
    eval_record_item: dict
    '''

    def append_eval_record(self, eval_record_item):
        for key in eval_record_item:
            if key not in self.eval_record:
                self.eval_record[key] = []
            self.eval_record[key].append(eval_record_item[key])

    '''
    append data record of model
    uid: model uid
    '''

    def append_model_record(self, uid):
        key = "model_uid"
        if key not in self.model_record:
            self.model_record[key] = []
        self.model_record[key].append(uid)

    def save_all_records(self, uid):
        self.save_record('train_record', uid)
        self.save_record('eval_record', uid)
        self.save_record('model_record', uid)

    def save_record(self, record_name, uid):
        record_dict = getattr(self, record_name)
        path = f'./data_record/{uid}'
        if not os.path.exists(path):
            os.makedirs(path)
        head = []
        for key in record_dict:
            head.append(key)
        result = ''
        for idx in range(len(record_dict[head[0]])):
            for key in head:
                result += str(record_dict[key][idx]) + '\t'
            result += '\n'

        result = "\t".join(head) + '\n' + result

        with open(f'{path}/{record_name}.csv', encoding='utf-8', mode='w+') as f:
            f.write(result)

        return uid

    '''
    X: the gold labels
    Y: the predicted labels
    '''
    @staticmethod
    def evaluationSAS(X, Y):
        if len(X) == 0:
            return 0, 0
        if len(X) != len(Y):
            raise Exception('mismatch length of X and Y')
        x_mean = np.mean(X)
        y_mean = np.mean(Y)
        r_a = 0
        r_b = 0
        r_c = 0
        r_mse = 0
        for i in range(len(X)):
            _x = (X[i] - x_mean)
            _y = (Y[i] - y_mean)
            r_a += _x * _y
            r_b += _x ** 2
            r_c += _y ** 2
            r_mse += (X[i] - Y[i]) ** 2
        r = r_a / (r_b ** 0.5 * r_c ** 0.5)
        r_mse = (r_mse / len(X)) ** 0.5

        return r, r_mse, pearsonr(X, Y)[0], spearmanr(X, Y)[0]

    @staticmethod
    def heatmap(data):
        return ValueError('')

    @staticmethod
    def save_xy(X, Y, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)
        result = ''
        for i in range(len(X)):
            result += '{}\t{}\n'.format(X[i], Y[i])
        with open('{}/predict_gold.csv'.format(dir), encoding='utf-8', mode='w+') as f:
            f.write(result)

    @staticmethod
    def save_same_row_list(dir, file_name, **args):
        if not os.path.isdir(dir):
            os.makedirs(dir)
        result = ''
        dicts = []
        for key in args.keys():
            dicts.append(key)
            result = key if result == '' else result + '\t{}'.format(key)
        length = len(args[dicts[0]])
        result += '\n'
        for i in range(length):
            t = True
            for key in args.keys():
                result += str(args[key][i]
                              ) if t else '\t{}'.format(args[key][i])
                t = False
            result += '\n'
        with open('{}/{}.csv'.format(dir, file_name), encoding='utf-8', mode='w+') as f:
            f.write(result)
