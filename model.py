import torch
import math
import copy

from transformers import get_linear_schedule_with_warmup, Adafactor
from torch.optim import AdamW

import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import tqdm as tqdm

from losses import InfoNCELoss, SupConLoss

from einops import rearrange

def switch_gradient(model, freeze: bool):
    for parameter in model.parameters():
        parameter.requires_grad_(freeze)

#auxiliary identity activation to delete tanh
class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class SupervisedContrastivePretrain(pl.LightningModule):
    def __init__(self, transformer,
                 learning_rate=5e-5,
                 weight_decay=.01,
                 num_warmup_steps=1000,
                 num_training_steps=10000,
                 enable_scheduler=True,
                 minibatch_size=128,
                 ):
        super().__init__()

        # Save hyperparameters for training

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.enable_scheduler = enable_scheduler
        self.minibatch_size = minibatch_size
        self.loss_func = SupConLoss(t_0=.07, eps=1e-12)
        self.loss_aux = InfoNCELoss(t_0=.07)
        self.save_hyperparameters()

        self.transformer = transformer

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=False,
                              relative_step=False,
                              warmup_init=False,
                              lr=self.learning_rate)

        if self.enable_scheduler:
            scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=self.num_warmup_steps,
                                                        num_training_steps=self.num_training_steps,
                                                        )
                                                        
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": 'linear_schedule_with_warmup',
            }
            return {'optimizer': optimizer,
                    'lr_scheduler': lr_scheduler_config,
                    }
        else:
            return {'optimizer': optimizer,
                    #'lr_scheduler': lr_scheduler_config,
                    }
    
    def training_step(self, train_batch, batch_idx):
        input_ids, attention_mask, labels = train_batch
        batch_size, view_size, _ = input_ids.shape

        optimizer = self.optimizers()
        optimizer.zero_grad()

        supcon_tracker, infonce_tracker, loss_tracker, acc_tracker = [], [], [], [] 

        n = int(math.ceil(batch_size*view_size/self.minibatch_size))

        simple_input_ids = rearrange(input_ids, 'b v i -> (b v) i')
        simple_mask = rearrange(attention_mask, 'b v i -> (b v) i')

        mb_input_ids = torch.chunk(simple_input_ids, n)
        mb_attention_mask = torch.chunk(simple_mask, n)

        with torch.no_grad():
            anchors = torch.cat([self(id, msk) for id, msk in zip(mb_input_ids, mb_attention_mask)], dim=0)

        for j, (a_ids, a_msk) in enumerate(zip(mb_input_ids, mb_attention_mask)):
            rep = copy.deepcopy(anchors)
            rep[(j * self.minibatch_size):((j+1) * self.minibatch_size)] = self(a_ids, a_msk)

            representation_views = rearrange(rep, '(b v) f -> b v f', b=batch_size, v=view_size)
            loss = self.loss_func(representation_views, labels)

            loss_tracker.append(loss)

            self.manual_backward(loss)

        with torch.no_grad():
            self.log(f'train/supcon_loss', sum(loss_tracker) / len(loss_tracker))
        
        optimizer.step()

        if self.enable_scheduler:
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()

        return loss

    def validation_step(self, val_batch, batch_idx):

        input_ids, attention_mask, labels = val_batch
        batch_size, view_size, _ = input_ids.shape

        n = int(math.ceil(len(input_ids) / self.minibatch_size*4))

        simple_input_ids = rearrange(input_ids, 'b v i -> (b v) i')
        simple_mask = rearrange(attention_mask, 'b v i -> (b v) i')

        mb_input_ids = torch.chunk(simple_input_ids, n)
        mb_attention_mask = torch.chunk(simple_mask, n)

        anchors = torch.cat([self(id_, msk) for id_, msk in zip(mb_input_ids, mb_attention_mask)], dim=0)
        
        representation_views = rearrange(anchors, '(b v) f -> b v f', b=batch_size, v=view_size)
        
        loss = self.loss_func(representation_views, labels)

        anchors, replicas, *_ = torch.chunk(representation_views, 4, dim=1)

        _, acc = self.loss_aux(anchors.mean(1), replicas.mean(1))

        self.log(f'valid/loss', loss)
        self.log(f'valid/accuracy', acc)

        return loss

    def predict_step(self, pred_batch, batch_idx):
        anchors, _, _ = pred_batch

        return self(anchors.input_ids, anchors.attention_mask)

class SupervisedContrastiveTransformer(SupervisedContrastivePretrain):
    def forward(self, input_ids, attention_mask=None):
        return  self.transformer(input_ids, attention_mask=attention_mask).pooler_output
        

class SupervisedContrastiveNoPoolerLast(SupervisedContrastiveTransformer):
    def __init__(self, transformer,
                 learning_rate=5e-5,
                 weight_decay=.01,
                 num_warmup_steps=1000,
                 num_training_steps=10000,
                 enable_scheduler=False,
                 minibatch_size=256,
                 unfreeze = 0,
                 **kwargs,
                 ):
        super().__init__(transformer,
                         learning_rate,
                         weight_decay,
                         num_warmup_steps,
                         num_training_steps,
                         enable_scheduler,
                         minibatch_size,
                         )
        if unfreeze > 0:
            for param in self.transformer.parameters():
                param.requires_grad = False

            for layer in self.transformer.encoder.layer[-unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.pooler = torch.nn.Sequential(torch.nn.Linear(1024, 1024),
                                          )
    
    def forward(self, input_ids, attention_mask=None):
        '''
        embedding =  self.transformer(input_ids, attention_mask=attention_mask).last_hidden_state
        if attention_mask is not None:
            expanded_attention_mask = attention_mask.unsqueeze(-1)
            reduced_embedding = (embedding * expanded_attention_mask).sum(1) / expanded_attention_mask.sum(1)
            return self.pooler(reduced_embedding)
        else:
            return self.pooler(embedding.mean(1))'''
        
        embedding =  self.transformer(input_ids, attention_mask=attention_mask).last_hidden_state
        if attention_mask is not None:
            expanded_attention_mask = attention_mask.unsqueeze(-1)
            reduced_embedding = (embedding * expanded_attention_mask).sum(1) / expanded_attention_mask.sum(1)
            return self.pooler(reduced_embedding)
        else:
            return self.pooler(embedding.mean(1))