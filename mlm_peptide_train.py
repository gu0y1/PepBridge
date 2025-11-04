from script.dataset import MaskedLMDataSet
from script.model.encoder import MaskedLM

import pandas as pd 
import numpy as np
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from script.dataprocess import mk_aa_dict
from script.utils import EarlyStopping

os.chdir('/data5/tem/laiwp131/pMHC_TCR_20251103/')

df = pd.read_csv('unique_peptide_train.csv',
                   header=0,index_col=0)
df_valid = df.sample(frac=0.1, random_state=42)
df_train = df.drop(df_valid.index)

aa_dict = mk_aa_dict()
lm_config = {
    'max_len':10,'d_seq':128, 'd_pair':64,
    'd_head':32, 'dropout':0.1, 'n_layers':6
}

model = MaskedLM(aa_size=len(aa_dict), max_len=lm_config['max_len'], 
                d_seq=lm_config['d_seq'], d_head_seq=lm_config['d_head'], 
                d_pair=lm_config['d_pair'], d_head_pair=lm_config['d_head'], 
                dropout=lm_config['dropout'], n_layers=lm_config['n_layers'])

if torch.cuda.is_available(): # cuda device
    device = 'cuda'
    torch.cuda.set_device(0)
else:
    device = 'cpu'

train_config = {
    'masked_rate':0.15,'batch_size':256, 'contiguous_prob':0.5,
    'lr':5e-5, 'weight_decay':0.01, 'max_epoch':500, 
     'patience':50, 'save_path':'peptide_mlm.pt'
}

model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'], 
                              weight_decay=train_config['weight_decay'])
early_stopping = EarlyStopping(patience=train_config['patience'], verbose=False, 
                                path=train_config['save_path'])

loss_dict = {}
t = tqdm(range(train_config['max_epoch']), desc="Epochs")
for epoch in t: 
    epoch_df_train = df_train.sample(n=100000, replace=False, random_state=None, axis=0)
    epoch_df_valid = df_valid.sample(n=20000, replace=False, random_state=None, axis=0)

    train_dataloader = DataLoader(MaskedLMDataSet(
        seq_list=epoch_df_train['peptide'].tolist(), 
        max_len=lm_config['max_len'], masked_rate=train_config['masked_rate'], 
        contiguous_prob=train_config['contiguous_prob']
        ), batch_size=train_config['batch_size'], shuffle=False, num_workers=0)
    
    valid_dataloader = DataLoader(MaskedLMDataSet(
        seq_list=epoch_df_valid['peptide'].tolist(), 
        max_len=lm_config['max_len'], masked_rate=train_config['masked_rate'], 
        contiguous_prob=train_config['contiguous_prob']
        ), batch_size=train_config['batch_size'] // 4, shuffle=False, num_workers=0)
    
    model.train()
    train_loss = 0.0
    for idx, (x, _) in enumerate(train_dataloader):
        input = x['mlm_input'].to(device)
        label = x['mlm_label'].to(device)
 
        _, mlm_loss = model(input, label)
        
        optimizer.zero_grad()
        mlm_loss.backward()
        optimizer.step()
        
        train_loss += mlm_loss.item()
    train_loss = train_loss / (idx+1)

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for idx, (x, _) in enumerate(valid_dataloader):
            input = x['mlm_input'].to(device)
            label = x['mlm_label'].to(device)
            _, mlm_loss = model(input, label)
            valid_loss += mlm_loss.item()
    valid_loss = valid_loss / (idx+1)

    epoch_loss = {'train_loss':train_loss, 
                  'valid_loss':valid_loss}
    info = ','.join(['{}={:.3f}'.format(key, value) for key, value in epoch_loss.items()])
    t.set_postfix_str(info)

    loss_dict[epoch]  = epoch_loss
    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
        print('EarlyStopping: run {} epoch'.format(epoch+1))
        break

import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context

epochs_sorted = sorted(loss_dict.keys())
train_loss_list = [loss_dict[e]['train_loss'] for e in epochs_sorted]
valid_loss_list = [loss_dict[e]['valid_loss'] for e in epochs_sorted]

with rc_context({'figure.figsize': (5, 4)}):
    plt.figure()
    plt.plot(epochs_sorted, train_loss_list, label="Train loss", color='#836AA2',linewidth=1.5)
    plt.plot(epochs_sorted, valid_loss_list, label="Valid loss", color='#85A446',linewidth=1.5)
    plt.title("Training & Validation Loss for Peptide LM")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(size = 14)
    plt.yticks(size = 14)
    plt.savefig('peptide_loss_curve.png', dpi=200, bbox_inches='tight')
    plt.savefig('peptide_loss_curve.pdf',bbox_inches='tight')
    plt.close()

pd.DataFrame({
    "epoch": epochs_sorted,
    "train_loss": train_loss_list,
    "valid_loss": valid_loss_list
}).to_csv("peptide_loss_curve.csv", index=False)
