import os
import torch
import torch.nn as nn
from dataset import GraphDataset_seq_smi, collate_fn_seq
import numpy as np
from model import *
from torch.utils.data import DataLoader
import torch
from thop import profile
import pandas as pd
import torch.optim as optim
from utils import *
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from IPython.display import SVG
from tqdm import tqdm 
from early_stopping import * 
from lion_pytorch import Lion

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(42)


def val(model, dataloader, device):
    model.eval()
    pred_list = []
    for data in dataloader:
        mg, mg2, hg, seq, smi, label = data
        mg, mg2, hg, seq, smi, label = mg.to(device), mg2.to(device), hg.to(device), seq.to(device), smi.to(device), label.to(device)

        with torch.no_grad():
            pred, _ = model(mg, mg2, hg, seq, smi)
            pred_list.append(pred.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0).flatten()

    return pred



if __name__ == '__main__':
     
    repeats = 3
    for repeat in range(repeats):
         
        data_root = './pdbbind/'
        train_dir = os.path.join(data_root, 'train')
        valid_dir = os.path.join(data_root, 'valid')
        test2013_dir = os.path.join(data_root, 'test2013')
        test2016_dir = os.path.join(data_root, 'test2016')
        test2019_dir = os.path.join(data_root, 'test2019')

        train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(data_root, 'valid.csv'))
        test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
        test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))
        test2019_df = pd.read_csv(os.path.join(data_root, 'test2019.csv'))
        graph_type='Graph_EHIGN'
        train_set = GraphDataset_seq_smi(train_dir, train_df, graph_type=graph_type, MAX_SEQ_LEN = 1300,  max_smi_len= 80)
        valid_set = GraphDataset_seq_smi(valid_dir, valid_df, graph_type=graph_type, MAX_SEQ_LEN = 1300,max_smi_len = 80)
        test2013_set = GraphDataset_seq_smi(test2013_dir, test2013_df, graph_type=graph_type, MAX_SEQ_LEN = 1300,max_smi_len = 80)
        test2016_set = GraphDataset_seq_smi(test2016_dir, test2016_df, graph_type=graph_type, MAX_SEQ_LEN = 1300,max_smi_len = 80)
        test2019_set = GraphDataset_seq_smi(test2019_dir, test2019_df, graph_type=graph_type, MAX_SEQ_LEN = 1300,max_smi_len = 80)
        batch_size = 128
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_seq, num_workers=12)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_seq, num_workers=12)
        test2013_loader = DataLoader(test2013_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_seq, num_workers=12)
        test2016_loader = DataLoader(test2016_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_seq, num_workers=12)
        test2019_loader = DataLoader(test2019_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_seq, num_workers=12)
 

        device = torch.device('cuda:0')
        model = PLAPredictor(node_feat_size=35, edge_feat_size=17, hidden_feat_size=256, layer_num=3).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
         

        criterion = nn.MSELoss()

        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []
        stopper = EarlyStopping(mode='lower', patience=30, filename='./model')
        # start training
        epochs = 200
        model.train()
        for epoch in range(epochs):
            tbar = tqdm(train_loader)
            for data in tbar:
                mg, mg2, hg, seq, smi, label = data
                mg, mg2, hg, seq, smi, label = mg.to(device), mg2.to(device), hg.to(device), seq.to(device), smi.to(device), label.to(device)
                pred , cl_loss = model(mg, mg2, hg, seq, smi)

                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), label.size(0)) 
                tbar.set_postfix(loss=f' * Train Epoch {epoch} Loss={loss.item()  :.3f} ')

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()


            # start validating
            valid_rmse, valid_pr = val(model, valid_loader, device)
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            print(msg)
            early_stop = stopper.step(valid_rmse, model)
            if early_stop:
                break
        stopper.load_checkpoint(model)
        valid_rmse, valid_pr = val(model, valid_loader, device)
        test2013_rmse, test2013_pr = val(model, test2013_loader, device)
        test2016_rmse, test2016_pr = val(model, test2016_loader, device)
        test2019_rmse, test2019_pr = val(model, test2019_loader, device)

        msg = "valid_rmse-%.4f, valid_pr-%.4f, test2013_rmse-%.4f, test2013_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f, test2019_rmse-%.4f, test2019_pr-%.4f,"\
                    % (valid_rmse, valid_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, test2019_rmse, test2019_pr)

        print(msg)#
        model_path = os.path.join("./models", msg + '.pt')
        torch.save(model.state_dict(), model_path)
        print("model has been saved to %s." % (model_path))
