import os
from dataset import GraphDataset_seq_smi, collate_fn_seq
import numpy as np
from model import *
from torch.utils.data import DataLoader
from rdkit import Chem
import torch
import pandas as pd
from utils import *

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def val(model, dataloader):
    model.eval()
    pred_list = []
    for data in dataloader:
        mg, mg2, hg, seq, smi, label = data
        with torch.no_grad():
            pred, _ = model(mg, mg2, hg, seq, smi)
            pred_list.append(pred.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0).flatten()
    return pred

test_dir = os.path.join('./sample_data/')
test_df = pd.read_csv(os.path.join('./sample_data', 'samples.csv'))
graph_type='Graph_EHIGN'
test_set = GraphDataset_seq_smi(test_dir, test_df, graph_type=graph_type, MAX_SEQ_LEN = 1300,max_smi_len = 80)
test_loader = DataLoader(test_set, batch_size=10, shuffle=False, collate_fn=collate_fn_seq, num_workers=8)
model = PLAPredictor(node_feat_size=35, edge_feat_size=17, hidden_feat_size=256, layer_num=3)
model_path = os.path.join('./model.pt')
load_model_dict(model, model_path)
device = torch.device('cuda:0')
model = model.to(device)
pred = val(model, test_loader)
