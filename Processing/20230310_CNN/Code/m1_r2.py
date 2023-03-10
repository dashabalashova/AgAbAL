model = 'AgAb_m1'
model_run = 'r2'
lines_num = 10
Agn_lst = [1, 2, 5, 10, 20, 50, 100]
epochs_num = 20

import sys
sys.path.append('../Code')

import AgAbCNN
import importlib
importlib.reload(AgAbCNN)

from AgAbCNN import CNNModel, AbLightDataset, f1
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
import torch

torch.set_num_threads(5)
os.makedirs('../Results/'+model+'_'+model_run+'/', exist_ok = True)

df = pd.read_csv('../Data/1ADQ_train.tsv', sep='\t')
df2 = df[['Ag']].drop_duplicates().sort_values(by='Ag').reset_index(drop=True)
dx_val = AbLightDataset(batch_size = 16, random_seed = 0, data_f = '1ADQ_val.tsv')
dx_val.setup()

roc_aucs = []
for l in range(lines_num):
    for Agn in Agn_lst:
        df3 = df[df.Ag.isin(df2.sample(frac=1, random_state=l).Ag[:Agn])]
        df3.to_csv('../Data/1ADQ_train_'+model+'_l'+str(l)+'_Agn'+str(Agn)+'.tsv', sep='\t', index=None)
        model_descr = model+'_l'+str(l)+'_Agn'+str(Agn)
        f_train = '1ADQ_train_'+model+'_l'+str(l)+'_Agn'+str(Agn)+'.tsv'
        seed_everything(0, workers = True)
        mod = CNNModel(learning_rate = 0.000075)
        for k in range(5):
            dx = AbLightDataset(k = k, batch_size = 16, random_seed = 0, data_f = f_train)
            dx.setup()
            early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
            trainer = Trainer(max_epochs = epochs_num, deterministic=True, callbacks=[early_stop_callback], \
                              num_sanity_val_steps=0, check_val_every_n_epoch=1)
            trainer.fit(model=mod, datamodule=dx)
            roc_auc, lloss = f1(mod, dx, trainer)
            print('model: %s, fold: %s, ROC_AUC: %s' %(model_descr, k, roc_auc))
            mod.eval()
            roc_auc_val, lloss_val = f1(mod, dx_val, trainer)
            print('model: %s, fold: %s, ROC_AUC_val: %s' %(model_descr, k, roc_auc_val))
            roc_aucs.append([model_descr, k, roc_auc, lloss, roc_auc_val, lloss_val])
            df4 = pd.DataFrame(roc_aucs, columns = ['model_descr', 'fold', 'roc_auc', 'llos', 'roc_auc_val', 'lloss_val'])
            df4['x'] = df4.apply(lambda x: int(x.model_descr.split('Agn')[-1]), axis=1)
            df4['line'] = df4.apply(lambda x: int(x.model_descr.split('_')[2][1:]), axis=1)
            df4.to_csv('../Results/'+model+'_'+model_run+'/accuracy.tsv', sep='\t', index=None)
        os.remove('../Data/1ADQ_train_'+model+'_l'+str(l)+'_Agn'+str(Agn)+'.tsv')
      