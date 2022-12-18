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


class AutoModel(IModel):

    def __init__(self, tokenizer, model_name, from_pretrained=None):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.from_pretrained = from_pretrained
        self.load_model(model_name)

    def load_model(self, model_name):
        bert_config_path = './model/chinese_wwm_ext/bert_config.json'
        bert_pre_trained_path = self.from_pretrained if self.from_pretrained is not None else './model/chinese_wwm_ext/pytorch_model.bin'
        albert_config_path = './model/albert_chinese_base/config.json'
        albert_pre_trained_path = self.from_pretrained if self.from_pretrained is not None else './model/albert_chinese_base/pytorch_model.bin'
        roberta_config_path = './model/roberta_chinese_base/config.json'
        roberta_pre_trained_path = self.from_pretrained if self.from_pretrained is not None else './model/roberta_chinese_base/pytorch_model.bin'
        xlnet_config_path = './model/chinese-xlnet-base/config.json'
        xlnet_pre_trained_path = self.from_pretrained if self.from_pretrained is not None else './model/chinese-xlnet-base/pytorch_model.bin'
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
        elif model_name == 'acbert':
            import pickle
            with open('./embedding/CNSTS/ori.numpy', 'rb') as f:
                pretrained_embeddings = pickle.load(f)
            self.model = ACBert(tokenizer=self.tokenizer, config_path=bert_config_path, pre_trained_path=bert_pre_trained_path, word_embedding_size=40210, pretrained_embeddings=pretrained_embeddings)

    def get_model(self):
        return self.model

    def optim_model(self):
        if self.model_name == 'wssbert':
            return self.model.get_model()
        return self.model

    def __call__(self):
        return self.get_model()
