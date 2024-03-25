import os
import numpy as np
import torch
import dgl
from torch.utils.data import DataLoader
from itertools import repeat

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)
PT_FEATURE_SIZE = 40


def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch]
    return X


CHARPROTSET = { "A": 1, "C": 2,  "D": 3, "E": 4, "F": 5,"G": 6, 
                 "H": 7, "I": 8, "K": 9, "L": 10, "M": 11, 
                 "N": 12,"P": 13, "Q": 14, "R": 15, "S": 16,  
                 "T": 17, "V": 18, "W": 19, 
                "Y": 20, "X": 21,"O": 22,"U": 23,"Z": 24, "X": 25
                 }

def label_sequence(line, MAX_SEQ_LEN):   
    X = np.zeros(MAX_SEQ_LEN, dtype=int)
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = CHARPROTSET[ch]
    return X

def collate_fn_seq(data_batch):
    mg, mg2, g, seq, smi,label = map(list, zip(*data_batch))
    bg = dgl.batch(g)
    mbg = dgl.batch(mg)
    mbg2 = dgl.batch(mg2)
    s = torch.from_numpy(np.array(seq))
    d = torch.from_numpy(np.array(smi))
    y = torch.cat(label, dim=0)
    
    return mbg, mbg2, bg, s, d, y


class GraphDataset_seq_smi(object):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, dis_threshold=6.0, MAX_SEQ_LEN=1000,  max_smi_len = 120, graph_type='Graph_EHIGN'):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.max_smi_len = max_smi_len
        self.graph_paths = None
        self.seq_list = None
        self._pre_process()

    def _pre_process(self):

        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        dis_thresholds = repeat(self.dis_threshold, len(data_df))

        pKa_list = []
        graph_path_list = []
        seq_list = []
        smi_list = []
        for i, row in data_df.iterrows():
            cid, pKa, seq, smi = row['pdbid'], float(row['-logKd/Ki']), row['Sequence'], row['smiles']
            complex_dir = os.path.join(data_dir, cid)
            graph_path = os.path.join(complex_dir, f"{graph_type}-{cid}-6.dgl")
            complex_path = os.path.join(complex_dir, f"{cid}-6.rdkit")
            pKa_list.append(pKa)
            seq_list.append(seq)
            smi_list.append(smi)

            graph_path_list.append(graph_path)

        self.graph_paths = graph_path_list
        self.seq_list = seq_list
        self.smi_list = smi_list



    def __getitem__(self, idx):
        bg, label = torch.load(self.graph_paths[idx])
        mp = ['inter_l2p', 'inter_p2l']
        mp2 = ['inter_p2l', 'inter_l2p']
        mgs = dgl.metapath_reachable_graph(bg, mp)
        mgs2 = dgl.metapath_reachable_graph(bg, mp2)
        mgs = dgl.add_self_loop(mgs) 
        mgs2 = dgl.add_self_loop(mgs2)
        seq = label_sequence(self.seq_list[idx], self.MAX_SEQ_LEN)  
        smi = label_smiles(self.smi_list[idx], self.max_smi_len)
        return mgs2, mgs, bg, seq, smi, label

    def __len__(self):
        return len(self.data_df)


