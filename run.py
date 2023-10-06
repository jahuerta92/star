from datetime import datetime

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModel, AutoTokenizer
import torch
import wandb
from data import (build_dataset, build_interleaved_supervised_dataset,
                  build_supervised_dataset)
from model import SupervisedContrastiveNoPoolerLast

USED_FILES = ['local_data/reddit_train.csv',
              'local_data/book_train_for_reddit.csv',
              'local_data/blog_train_for_reddit.csv',
              'local_data/twitter_train.csv',
             ]
    
train_datasets = []
print('Reading training files...')
for file in USED_FILES:
    train_file = pd.read_csv(file)
    train_file['unique_id'] = train_file.index.astype(str) + f'{file}'
    train_file = train_file.drop_duplicates(subset=["decoded_text"], keep=False)
    train_datasets.append(train_file[['unique_id', 'id', 'decoded_text']])
    
train = pd.concat(train_datasets).sample(frac=1.)
test = pd.read_csv('local_data/reddit_test.csv')

test.columns = ['id', 'decoded_text', 'subreddit']

test['unique_id'] = test.index.astype(str)

print('Setting parameters...')
BATCH_SIZE = 1024
VIEW_SIZE = 16
MINIBATCH_SIZE = 128 // 4
VALID_BATCH_SIZE = 64
CHUNK_SIZE = 512
TRAINING_STEPS = 3000
VALIDATION_STEPS = 100
WARMUP_STEPS = .06
WEIGHT_DECAY = 1e-4
UNFREEZE = 0
LR = 1e-2
DEVICE = 1
MODEL_CODE = 'roberta-large'

print('Building dataset...')
train_data = build_interleaved_supervised_dataset(train_datasets,
                           steps=TRAINING_STEPS*BATCH_SIZE,
                           batch_size=BATCH_SIZE,
                           num_workers=8, 
                           prefetch_factor=8,
                           max_len=CHUNK_SIZE,
                           views=VIEW_SIZE,
                           tokenizer = AutoTokenizer.from_pretrained(MODEL_CODE),
                           )
test_data = build_supervised_dataset(test, 
                          steps=VALIDATION_STEPS*VALID_BATCH_SIZE, 
                          batch_size=VALID_BATCH_SIZE, 
                          num_workers=4, 
                          prefetch_factor=4, 
                          max_len=CHUNK_SIZE,
                          views=VIEW_SIZE,
                          shuffle=False,
                          tokenizer = AutoTokenizer.from_pretrained(MODEL_CODE),
                          )

print('Setup trainer...')
# Name model
date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_name = f'{date_time}_scl_roberta_bs{BATCH_SIZE}'
print(f'Saving model to {save_name}')

checkpoint_callback = ModelCheckpoint('model',
                                      filename=save_name,
                                      monitor=None,
                                      every_n_train_steps=250
                                      )
lr_monitor = LearningRateMonitor('step')

# Define training arguments
trainer = Trainer(devices=[DEVICE],
                  max_steps=TRAINING_STEPS,
                  accelerator='gpu',
                  log_every_n_steps=1,
                  precision=16,
                  val_check_interval=50,
                  callbacks=[checkpoint_callback, lr_monitor],
                  )

# Define model
print('Defining model...')
base_transformer = AutoModel.from_pretrained(MODEL_CODE)
train_model = SupervisedContrastiveTransformerLast(base_transformer, #base_transformer,
                                                   learning_rate=LR,
                                                   weight_decay=WEIGHT_DECAY,
                                                   num_warmup_steps=TRAINING_STEPS*WARMUP_STEPS,
                                                   num_training_steps=TRAINING_STEPS,
                                                   enable_scheduler=True,
                                                   minibatch_size=MINIBATCH_SIZE,
                                                   unfreeze=UNFREEZE,)

print('Begin training...')
trainer.fit(train_model, train_data, test_data)
wandb.finish()
