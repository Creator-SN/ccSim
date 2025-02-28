from CC.ICCStandard import IModel
from CC.models.bert import Bert
from CC.models.bertlm import BertLM
from CC.models.r2bert import R2Bert
from CC.models.albert import Albert
from CC.models.roberta import Roberta
from CC.models.xlnet import XLNet
from CC.models.wssbert import WSSBert
from CC.models.esim import ESIM
from CC.models.bimpm import BIMPM
from CC.models.sbert import SBert
from CC.models.abcnn import ABCNN
from CC.models.textcnn import TextCNN
from CC.models.siagru import SiaGRU
from CC.models.x import XSSBert
from CC.models.msim import MSIM
from CC.models.simcse import SIMCSE
from CC.models.ACBert import ACBert
from CC.models.ACErnie import ACErnie
from CC.models.ernie import Ernie
import os
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel as am


class AutoModel(IModel):

    def __init__(self, tokenizer, model_name, from_pretrained=None, matched_word_vocab_file=None, emb_pretrained_path=None):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.from_pretrained = from_pretrained
        self.load_model(model_name, matched_word_vocab_file, emb_pretrained_path)

    def load_model(self, model_name, matched_word_vocab_file=None, emb_pretrained_path=None):
        bert_config_path = './model/chinese_wwm_ext/bert_config.json'
        bert_pre_trained_path = self.from_pretrained if self.from_pretrained is not None else './model/chinese_wwm_ext/pytorch_model.bin'
        albert_config_path = './model/albert_chinese_base/config.json'
        albert_pre_trained_path = self.from_pretrained if self.from_pretrained is not None else './model/albert_chinese_base/pytorch_model.bin'
        roberta_config_path = './model/roberta_chinese_base/config.json'
        roberta_pre_trained_path = self.from_pretrained if self.from_pretrained is not None else './model/roberta_chinese_base/pytorch_model.bin'
        xlnet_config_path = './model/chinese-xlnet-base/config.json'
        xlnet_pre_trained_path = self.from_pretrained if self.from_pretrained is not None else './model/chinese-xlnet-base/pytorch_model.bin'
        ernie3_config_path = './model/ernie-3.0-base-zh/config.json'
        ernie3_pre_trained_path = self.from_pretrained if self.from_pretrained is not None else './model/ernie-3.0-base-zh/pytorch_model.bin'
        ernie2_config_path = './model/ernie-2.0-base-zh/config.json'
        ernie2_pre_trained_path = self.from_pretrained if self.from_pretrained is not None else './model/ernie-2.0-base-chinese/pytorch_model.bin'
        ernie_config_path = './model/ernie-1.0-base-zh/config.json'
        ernie_pre_trained_path = self.from_pretrained if self.from_pretrained is not None else './model/ernie-1.0-base-zh'
        if model_name == 'bert':
            self.model = Bert(tokenizer=self.tokenizer, config_path=bert_config_path,
                              pre_trained_path=bert_pre_trained_path)
        elif model_name == 'bertlm':
            self.model = BertLM(tokenizer=self.tokenizer, config_path=bert_config_path,
                                pre_trained_path=bert_pre_trained_path)
        elif model_name == 'r2bert':
            self.model = R2Bert(tokenizer=self.tokenizer, config_path=bert_config_path,
                                pre_trained_path=bert_pre_trained_path)
        elif model_name == 'albert':
            self.model = Albert(tokenizer=self.tokenizer, config_path=albert_config_path,
                                pre_trained_path=albert_pre_trained_path)
        elif model_name == 'roberta':
            self.model = Roberta(tokenizer=self.tokenizer, config_path=roberta_config_path,
                                 pre_trained_path=roberta_pre_trained_path)
        elif model_name == 'xlnet':
            self.model = XLNet(tokenizer=self.tokenizer, config_path=xlnet_config_path,
                               pre_trained_path=xlnet_pre_trained_path)
        elif model_name == 'wssbert':
            self.model = WSSBert(
                tokenizer=self.tokenizer, config_path=bert_config_path, pre_trained_path=bert_pre_trained_path)
        elif model_name == 'esim':
            self.model = ESIM()
        elif model_name == 'bimpm':
            self.model = BIMPM()
        elif model_name == 'sbert':
            self.model = SBert(tokenizer=self.tokenizer, config_path=bert_config_path,
                               pre_trained_path=bert_pre_trained_path)
        elif model_name == 'abcnn':
            self.model = ABCNN()
        elif model_name == 'textcnn':
            self.model = TextCNN()
        elif model_name == 'siagru':
            self.model = SiaGRU()
        elif model_name == 'x':
            self.model = XSSBert(
                tokenizer=self.tokenizer, config_path=bert_config_path, pre_trained_path=bert_pre_trained_path)
        elif model_name == 'x_k':
            self.model = XSSBert(tokenizer=self.tokenizer, config_path=bert_config_path,
                                 pre_trained_path=bert_pre_trained_path, mode='keywords_only')
        elif model_name == 'x_s':
            self.model = XSSBert(tokenizer=self.tokenizer, config_path=bert_config_path,
                                 pre_trained_path=bert_pre_trained_path, mode='seq_only')
        elif model_name == 'msim':
            self.model = MSIM(tokenizer=self.tokenizer, config_path=bert_config_path,
                              pre_trained_path=bert_pre_trained_path)
        elif model_name == 'simcse':
            self.model = SIMCSE(
                tokenizer=self.tokenizer, config_path=bert_config_path, pre_trained_path=bert_pre_trained_path)
        elif model_name == 'ernie3':
            self.model = Ernie(tokenizer=self.tokenizer, config_path=ernie3_config_path,
                               pre_trained_path=ernie3_pre_trained_path)
        elif model_name == 'ernie2':
            self.model = Ernie(tokenizer=self.tokenizer, config_path=ernie2_config_path,
                               pre_trained_path=ernie2_pre_trained_path)
        elif model_name == 'ernie':
            self.model = Ernie(tokenizer=self.tokenizer, config_path=ernie_config_path,
                               pre_trained_path=ernie_pre_trained_path)
        elif model_name == 'acbert':
            assert matched_word_vocab_file is not None
            assert emb_pretrained_path is not None
            embedding_size, embeddings = AutoModel.get_matched_word_embeddings(matched_word_vocab_file, emb_pretrained_path)
            self.model = ACBert(tokenizer=self.tokenizer, config_path=bert_config_path, pre_trained_path=bert_pre_trained_path,
                                word_embedding_size=embedding_size, pretrained_embeddings=embeddings)
        elif model_name == 'acernie':
            assert matched_word_vocab_file is not None
            assert emb_pretrained_path is not None
            embedding_size, embeddings = AutoModel.get_matched_word_embeddings(matched_word_vocab_file, emb_pretrained_path)
            self.model = ACErnie(tokenizer=self.tokenizer, config_path=ernie_config_path, pre_trained_path=ernie_pre_trained_path,
                                word_embedding_size=embedding_size, pretrained_embeddings=embeddings)
    
    @staticmethod
    def get_matched_word_embeddings(matched_word_vocab_file, emb_pretrained_path):
        file_name = os.path.basename(matched_word_vocab_file)
        cache_path = f'./embedding/ori_{file_name}'
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                pretrained_embeddings = pickle.load(f)
            return pretrained_embeddings.shape[0], pretrained_embeddings
        tokenizer = AutoTokenizer.from_pretrained(emb_pretrained_path)
        model = am.from_pretrained(emb_pretrained_path)
        BATCH_SIZE = 64

        model.eval()
        model.cuda()
        
        with open(matched_word_vocab_file) as f:
            ori_list = f.readlines()

        result_list = []

        result_list.append([0 for _ in range(768)])

        num_batches = len(ori_list) / BATCH_SIZE if len(ori_list) % BATCH_SIZE == 0 else int(len(ori_list) / BATCH_SIZE + 1)

        for idx in tqdm(range(num_batches)):
            lines = ori_list[idx*BATCH_SIZE : (idx+1)*BATCH_SIZE]
            entity_list = [line.split(',')[0] for line in lines]
            T = tokenizer(entity_list, padding=True, truncation=True, return_tensors="pt")
            T = {k: v.cuda() for k, v in T.items()}
            output = model(**T)
            result_list += output.pooler_output.tolist()

        if not os.path.exists(f'./embedding'):
            os.mkdir(f'./embedding')

        result_list_num = np.array(result_list)

        with open(cache_path, 'wb') as f:
            pickle.dump(result_list_num, f, 2)
        return len(ori_list) + 1, result_list_num

    def get_model(self):
        return self.model

    def optim_model(self):
        if self.model_name == 'wssbert':
            return self.model.get_model()
        return self.model

    def __call__(self):
        return self.get_model()
