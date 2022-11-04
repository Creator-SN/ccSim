import os
import torch
from CC.sas_trainer import Trainer


class CLTrainer(Trainer):
    def save_model(self, current_step=0):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        if not os.path.exists('./save_model/{}/{}'.format(dir, self.model_name)):
            os.makedirs('./save_model/{}/{}'.format(dir, self.model_name))
        try:
            self.model.model.save_pretrained(
                './save_model/{}/{}/{}_{}'.format(dir, self.model_name, self.model_name, current_step))
        except:
            self.model.module.model.save_pretrained(
                './save_model/{}/{}/{}_{}'.format(dir, self.model_name, self.model_name, current_step))
        self.analysis.append_model_record(current_step)
        return current_step
