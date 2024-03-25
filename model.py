import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv
from heterograh import HeteroGraphConv
from dgl.nn.pytorch import GraphConv, SGConv, GINConv
from torch.nn import init
from dgl.nn import MaxPooling,WeightAndSum
import dgl.nn.pytorch as dglnn
from rgat_layer import * 
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
import dgl.function as fn
from loss import * 

from torch.autograd import Variable
from einops.layers.torch import Rearrange, Reduce



class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.U = nn.Linear(self.dim, self.dim )
        self.V = nn.Linear(self.dim, self.dim )
        self.h1 = nn.Linear(self.dim, self.dim )
        self.h2 = nn.Linear(self.dim, self.dim  )

        self.dynamic_weights = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            # nn.GELU(),
            # # nn.Dropout(p=0.2),
            # nn.Linear(1024, self.dim),
            nn.Sigmoid()
        )
         
        self.P = nn.Linear(self.dim*2  , self.dim*2 )
        self.norm = nn.LayerNorm(self.dim*2)
        self.dropout = nn.Dropout(0.2)
    def forward(self, input1, input2):

        dynamic_weight = self.dynamic_weights(input1 + input2)

        output =  torch.cat((dynamic_weight *  input1,  (1-dynamic_weight) *  input2 ),1 ) # dynamic_weight *  input1 +  (1-dynamic_weight) *  input2
        output = self.P(output)
        output = self.norm(output)
        # output = self.dropout(output)
        return output

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        

    def forward(self, x):
        return x * torch.sigmoid( x)

    


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        mult = 4,
        dropout = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        activation = Swish()
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            activation
        )  

        self.ff = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim),
            # nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
            nn.LayerNorm(dim_out),
        )

        #self.ff.add_module("transition", project_in)

    def forward(self, x):
        return self.ff(x)


class GraphFeatureEncoder(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hetero_dim,
        dim_out,
        mult = 4,
        dropout = 0.0,
        
    ):
        super().__init__()

        ligand_encoder = FeedForward(node_dim, dim_out, mult, dropout)
        self.pocket_encoder = FeedForward(node_dim, dim_out, mult, dropout)
        ligand_edge_encoder =  FeedForward(edge_dim, dim_out, mult, dropout)
        self.pocket_edge_encoder = FeedForward(edge_dim, dim_out, mult, dropout)

        self.hetero_edge_encoder1 = FeedForward(hetero_dim, dim_out, mult, dropout)
        self.hetero_edge_encoder2 = FeedForward(hetero_dim, dim_out, mult, dropout)

        self.add_module('ligand_encoder', ligand_encoder)
        self.add_module('ligand_edge_encoder', ligand_edge_encoder)
         

    def forward(self, bg):
        atom_feats = bg.ndata['h']
        bond_feats = bg.edata['e']

        atom_feats = {
            'ligand':self.ligand_encoder(atom_feats['ligand']),
            'pocket':self.pocket_encoder(atom_feats['pocket'])
        }
        
        bond_feats = {
            ('ligand', 'intra_l', 'ligand'):self.ligand_edge_encoder(bond_feats[('ligand', 'intra_l', 'ligand')]),
            ('pocket', 'intra_p', 'pocket'):self.pocket_edge_encoder(bond_feats[('pocket', 'intra_p', 'pocket')]),       
            ('ligand', 'inter_l2p', 'pocket'):self.hetero_edge_encoder1(bond_feats[('ligand', 'inter_l2p', 'pocket')]),    
            ('pocket', 'inter_p2l', 'ligand'):self.hetero_edge_encoder2(bond_feats[('pocket', 'inter_p2l', 'ligand')]),        
        }
        


        bg.edata['e'] = bond_feats
        bg.nodes['ligand'].data['h'] = atom_feats['ligand']
        bg.nodes['pocket'].data['h'] = atom_feats['pocket']
        return bg

         

class GINConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, **kwargs):
        super(GINConv, self).__init__()
        lin = nn.Sequential(nn.Linear(dim_in, dim_out, bias), nn.ReLU(),
                               nn.Linear(dim_out, dim_out))
        self.model = dgl.nn.pytorch.GINConv(lin, 'max')

    def forward(self, g, h):
        h = self.model(g, h)
        return h

class SGCLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=True):
        super(SGCLayer, self).__init__()
        self.gcn_layer = SGConv(in_feats,
                                out_feats,
                                k,
                                cached,
                                bias,
                                norm,
                                allow_zero_in_degree)
        self.res = nn.Linear(in_feats, out_feats)

    def forward(self, bg, feat):
        return self.gcn_layer(bg, feat) + self.res(feat) 

class MetapathEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        
        super().__init__()
        self.lin_node_l = FeedForward(input_dim, hidden_dim)

        self.gcn = SGCLayer(hidden_dim, hidden_dim )
        self.gcn2 = SGCLayer(hidden_dim, hidden_dim )
        self.gcn3 = SGCLayer(hidden_dim, hidden_dim )
        self.gcn4 = SGCLayer(hidden_dim, hidden_dim )
        self.pool = MaxPooling()
        self.pred = nn.Linear(hidden_dim, 1)

    def forward(self, mgs):
        feat = mgs.ndata['h']
        feat = self.lin_node_l(feat)
        h =  self.gcn(mgs, feat)
        h = self.gcn2(mgs, h)
        h = self.gcn3(mgs, h)
        h = self.gcn4(mgs, h)
        h = self.pool(mgs, h)

        return h

class EdgePredictLayer(nn.Module):
    def __init__(self, hidden_feat_size):
        super(EdgePredictLayer, self).__init__()
         
        self.graph_proj = GraphFeatureEncoder(hidden_feat_size, hidden_feat_size,hidden_feat_size, hidden_feat_size)
        self.pred1 = nn.Linear(hidden_feat_size, 1)
        self.pred2 = nn.Linear(hidden_feat_size, 1)
        
    def forward(self,  g):
        with g.local_scope():
            g = self.graph_proj(g)
             
            
            h_lp = dgl.max_edges(g, 'e', etype='inter_l2p')
            h_pl = dgl.max_edges(g, 'e', etype='inter_p2l')


            h =  torch.cat((h_lp, h_pl),1)
            return  self.pred1(h_lp), self.pred2(h_pl), h 



class NodePredictLayer(nn.Module):
    def __init__(self, hidden_feat_size):
        super(NodePredictLayer, self).__init__()
         
        self.graph_proj = GraphFeatureEncoder(hidden_feat_size, hidden_feat_size,hidden_feat_size, hidden_feat_size)
        self.pred = nn.Linear(hidden_feat_size*2, 1)

        
           
    def forward(self,  g):
        with g.local_scope():
            g = self.graph_proj(g)  
            hp = dgl.max_nodes(g, 'h', ntype='pocket')
            hl = dgl.max_nodes(g, 'h', ntype='ligand')
            h =  torch.cat((hp, hl),1)
            return  self.pred(h), h 

class CNNBlocks_seq(nn.Module):
    def __init__(self, input_dim=64,name='res_prot'):
        super(CNNBlocks_seq, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding='valid',
                                 strides=1,)
        self.relu1 = nn.ReLU()
        #self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding='valid',
                                 strides=1,)
        self.relu2 = nn.ReLU()
        #self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding='valid',
                                 strides=1,)
        self.relu3 = nn.ReLU()

        self.pool3 = nn.Pool1d(kernel_size=2, stride=2)

        
    def forward(self, seq):
        # 将第二维和第三维进行转置
        conv1_out = self.conv1(seq.transpose(1, 2))
        relu1_out = self.relu1(conv1_out)
        #pool1_out = self.pool1(relu1_out)
        conv2_out = self.conv1(relu1_out)
        relu2_out = self.relu1(conv2_out)


        conv3_out = self.conv3(relu2_out)
        relu3_out = self.relu3(conv3_out)
        pool3_out = self.pool3(relu3_out)
        #x = Reduce('b c t -> b c', 'max')(pool3_out)
        return pool3_out

class CNNBlocks_smi(nn.Module):
    def __init__(self, input_dim=64,name='res_drug'):
        super(CNNBlocks_smi, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3)
        self.relu1 = nn.ReLU()
        #self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        #self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3)
        self.relu3 = nn.ReLU()

        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        
    def forward(self, smi):
        conv1_out = self.conv1(smi.transpose(1, 2))
        relu1_out = self.relu1(conv1_out)
        #pool1_out = self.pool1(relu1_out)

        conv2_out = self.conv1(relu1_out)
        relu2_out = self.relu1(conv2_out)

        conv3_out = self.conv3(relu2_out)
        relu3_out = self.relu3(conv3_out)
        pool3_out = self.pool3(relu3_out)
        #x = Reduce('b c t -> b c', 'max')(pool3_out)
        return pool3_out

class CNNs_DeepDTA_Net(nn.Module):
    def __init__(self, num_filters, filter_length1, filter_length2):
        super(CNNs_DeepDTA_Net, self).__init__()

        # Drug representation
        self.d_input = nn.Embedding(num_embeddings=65, embedding_dim=128)
        self.d_cnn1 = nn.Conv1d(in_channels=128, out_channels=num_filters, kernel_size=filter_length1)
        self.d_cnn2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 1, kernel_size=filter_length1)
        self.d_cnn3 = nn.Conv1d(in_channels=num_filters * 1, out_channels=num_filters * 1, kernel_size=filter_length1)
        self.d_global_maxpool = nn.AdaptiveMaxPool1d(output_size=1)

        # Protein representation
        self.p_input = nn.Embedding(num_embeddings=26, embedding_dim=128)
        self.p_cnn1 = nn.Conv1d(in_channels=128, out_channels=num_filters, kernel_size=filter_length2)
        self.p_cnn2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=filter_length2)# * 2
        self.p_cnn3 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 1, kernel_size=filter_length2)# * 2
        self.p_global_maxpool = nn.AdaptiveMaxPool1d(output_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters * 2, 1)#

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, d_input, p_input):
        d_p = self.d_input(d_input)
        d_p = d_p.permute(0, 2, 1)
        d_p = self.d_cnn1(d_p)
        d_p = self.relu(d_p)
        d_p = self.d_cnn2(d_p)
        d_p = self.relu(d_p)
        d_p = self.d_cnn3(d_p)
        d_p = self.relu(d_p)
        d_p = self.d_global_maxpool(d_p).squeeze()

        p_p = self.p_input(p_input)
        p_p = p_p.permute(0, 2, 1)
        p_p = self.p_cnn1(p_p)
        p_p = self.relu(p_p)
        p_p = self.p_cnn2(p_p)
        p_p = self.relu(p_p)
        p_p = self.p_cnn3(p_p)
        p_p = self.relu(p_p)
        p_p = self.p_global_maxpool(p_p).squeeze()

        c_p = torch.cat((d_p, p_p), dim=-1)

        fc = self.fc1(c_p)
        y = self.relu(fc)


        return y


class PLAPredictor(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size, layer_num=3):
        super(PLAPredictor, self).__init__()

        #self.embed_prot = nn.Embedding(26, 128)
        #self.embed_drug = nn.Embedding(65, 128)
        #self.onehot_prot_net = CNNBlocks_seq(128, name='res_prot')
        #self.onehot_drug_net = CNNBlocks_smi(128, name='res_drug')
        self.deepdta = CNNs_DeepDTA_Net(32, 4, 8)

        self.graph_f_encoder = GraphFeatureEncoder(node_feat_size, edge_feat_size, 11, hidden_feat_size)
        self.convs = nn.ModuleList()
        rel_names = ['intra_l', 'intra_p', 'inter_l2p', 'inter_p2l'] 
        for _ in range(layer_num):
            # conv = HeteroGraphAttentionConv({
            # rel: dglnn.GraphConv(hidden_feat_size, hidden_feat_size)
            #     for rel in rel_names},  att_in_feats=hidden_feat_size, att_out_feats=hidden_feat_size)
            conv = HeteroGraphConv({
            rel: dglnn.GraphConv(hidden_feat_size, hidden_feat_size)
                for rel in rel_names})
            self.convs.append(conv)

        
        
        self.mg_encoder = MetapathEncoder(node_feat_size, hidden_feat_size)
        self.mg_encoder2 = MetapathEncoder(node_feat_size, hidden_feat_size)
        self.fcn = nn.Sequential( nn.Linear(hidden_feat_size*2, 1))

        self.edge_pred = EdgePredictLayer(hidden_feat_size)
        self.node_pred = NodePredictLayer(hidden_feat_size)

        self.meta_fusion = GatedFusion(hidden_feat_size)

        # self.ensemble_fcn = nn.Linear(6*hidden_feat_size,1,bias=False)

        self.transform = nn.Sequential(
             nn.LayerNorm(512),
             nn.Linear(512, 1),
            #  nn.Dropout(p=0.1),
            #  #nn.LayerNorm(64),
            #  nn.Linear(1024, 512),
            #  nn.LayerNorm(512),
            #  nn.Dropout(p=0.1),
            # nn.Linear(512, 1)

         )


 
    def forward(self, mg, mg2, bg, seq, smi):
        #proteinFeature_onehot = self.embed_prot(seq)
        #smilesFeature_onehot = self.embed_drug(smi)
        #seq_pre = self.onehot_prot_net(proteinFeature_onehot)
        #smi_pre = self.onehot_drug_net(smilesFeature_onehot)

        #combined_features = torch.cat((seq_pre, smi_pre), dim=1)

        #seq_pre = self.transform(combined_features)
        #smi_pre = self.transform(smi_pre)
        seq_smi = self.deepdta(smi, seq)
        

        bg = self.graph_f_encoder(bg)

        rsts = bg.ndata['h']
         
        for conv in self.convs:
            rsts = conv(bg, rsts)

        bg.nodes['ligand'].data['h'] = rsts['ligand']
        bg.nodes['pocket'].data['h'] = rsts['pocket']

        

        mg_f1 = self.mg_encoder(mg)
        mg_f2 = self.mg_encoder(mg2)
        mg_f = self.meta_fusion( mg_f1, mg_f2 )

        meta_pred = self.fcn( mg_f )
        hg_e_pred1, hg_e_pred2, e_f = self.edge_pred(bg)
        hg_n_pred, n_f = self.node_pred(bg)

        return   1/5 * (hg_e_pred1 + hg_e_pred2 + hg_n_pred+ meta_pred +seq_smi).view(-1), (mg_f, n_f, e_f) # cl_loss

    def cl_loss_f(self, x1, x2):
        batch_size = x1.size(0)
        device = x1.device 
        clt = NTXentLoss_poly(device, batch_size, 0.07, True)
        return clt(x1, x2)


