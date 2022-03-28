"""
    NRI with PyTorch Geometric
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from SourceModified.message_passing import MessagePassing2

import numpy as np

#MLP Block with XavierInit & BatchNorm1D (n_in, n_hid, n_out)
class MLPBlock(nn.Module):
    """
    MLP Block with XavierInit & BatchNorm1D (n_in, n_hid, n_out)
    """
    def __init__(self, n_in, n_hid, n_out, prob_drop=0.):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.prob_drop = prob_drop
        self.init_weights()
    
    def init_weights(self):
        '''
        Init weight with Xavier Nornalization
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        sizes = inputs.size()
        x = inputs.view(-1, inputs.size(-1))
        x = self.bn(x)
        return x.view(sizes)

    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.prob_drop, training=self.training) #training
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)

class MLPEncoder_NRI(MessagePassing2):
    """
        MLP Encoder from https://github.com/ethanfetaya/NRI using Pytorch Geometric
    """
    def __init__(self, n_in, n_hid, n_out, skip=True):
        super(MLPEncoder_NRI, self).__init__(aggr='add') # Eq 7 aggr part
        self.mlp_eq5_embedding = MLPBlock(n_in, n_hid, n_hid)
        self.mlp_eq6 = MLPBlock(n_hid*2, n_hid, n_hid)
        self.mlp_eq7 = MLPBlock(n_hid, n_hid, n_hid)
        self.mlp_eq8_skipcon = MLPBlock(n_hid*3, n_hid, n_hid)
        self.fc_out = nn.Linear(n_hid, n_out)
        self.skip = skip
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, edge_index):
        # Eq 5
        x = self.mlp_eq5_embedding(inputs) # [sims, nodes, hid_dim]
        # x: aggr_msg, x_edge: message output for skip-con
        x, x_edge = self.propagate(edge_index, x=x, skip=not self.skip) #[sims, nodes, hid_dim], [sims, edges, hid_dim] 
        _, x = self.propagate(edge_index, x=x, skip=self.skip, x_skip=x_edge) #[batch_size * edges, hid_dim]
        x = self.fc_out(x) #[batch_size * edges, edge_features]
        return x

    def message(self, x_i, x_j, skip, x_skip=None):
        # Eq 6
        if not skip:
            x_edge = torch.cat([x_i, x_j], dim=-1)
            x_edge = self.mlp_eq6(x_edge)
        # Eq 8
        else:
            x_edge = torch.cat([x_i, x_j, x_skip], dim=-1)
            x_edge = self.mlp_eq8_skipcon(x_edge)
        return x_edge

    def update(self, aggr_out):
        # Eq 7 MLP part
        new_embedding = self.mlp_eq7(aggr_out)
        return new_embedding

class MLPDecoder_NRI(MessagePassing):
    """
        MLP Encoder from https://github.com/ethanfetaya/NRI using Pytorch Geometric.\n
        Edge types are not explictly hard-coded by default.\n
        To explicitly hard code first edge-type as 'non-edge', set 'edge_type_explicit = True' 
        in config.yml.\n
    """
    def __init__(self, features_dims, num_edge_types, msg_hid, msg_out, 
                 n_hid, config, do_prob=0.):
        super(MLPDecoder_NRI, self).__init__(aggr='add')
        self.fc1_eq10 = nn.ModuleList(
            [nn.Linear(2*features_dims, msg_hid) for _ in range(num_edge_types)])
        self.fc2_eq10 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(num_edge_types)])
        self.eq10_out_dim = msg_out

        self.fc1_eq11 = nn.Linear(features_dims + msg_out, n_hid)
        self.fc2_eq11 = nn.Linear(n_hid, n_hid)
        self.fc3_eq11 = nn.Linear(n_hid, features_dims)
        self.dropout_prob = do_prob

        #config (num_nodes, node_feature_dim, num_timesteps, pred_step, num_chunk, device)
        self.edge_type_explicit = config.get('edge_type_explicit')
        self.num_nodes = config.get('num_nodes')
        self.node_feature_dim = config.get('num_node_features')
        self.num_timesteps = config.get('num_timesteps')
        self.edge_types = config.get('edge_types')
        self.pred_step = config.get('pred_step')
        self.num_chunk = np.ceil(config.get('num_timesteps') / 
                                 config.get('pred_step')).astype(int)

    def single_step_pred(self, idxed_input, edge_index, z):
        # idxed_input [sims * nodes, idxed_tsteps, features] | z [edges, edge_types] 
        idxed_input = idxed_input.transpose(0,1).contiguous()
        x = self.propagate(edge_index, x=idxed_input, z=z, indexed_input=idxed_input, 
                           edge_explicit=self.edge_type_explicit)
        x = x.transpose(0,1).contiguous()

        return x #[sims * nodes, tsteps, features]

    def forward(self, inputs, edge_index, z, pred_steps):
        ''' '''
        # inputs [nodes, tsteps *features] / z [edges, edge_types]
        inputs = inputs.view(-1, self.num_timesteps, self.node_feature_dim)

        assert(pred_steps <= self.num_timesteps)
        preds = []

        # inputs [sims * nodes, tsteps, features]
        idxed_inputs = inputs[:, 0::pred_steps, :]

        for _ in range(0, pred_steps):
            idxed_inputs = self.single_step_pred(idxed_inputs, edge_index, z)
            preds.append(idxed_inputs)
            
        pred_all = torch.stack(preds, dim=2) # sims*nodes, num_chunks, pred_step, feature_dim
        pred_all = pred_all.view(-1, self.num_chunk*pred_steps, self.node_feature_dim)
        return pred_all[:, :self.num_timesteps - 1, :] #[sims * nodes, tsteps, features]

    # Eq 10
    def message(self, x_i, x_j, z, edge_explicit):        
        x_edge = torch.cat([x_i, x_j], dim=-1) #[sims * edges, features]
        start_idx = 1 if edge_explicit else 0

        for i in range(start_idx, len(self.fc1_eq10)):
            scale_factor = start_idx * (i - 1) + 1
            if i == start_idx:
                msg = F.relu(self.fc1_eq10[i](x_edge))
                msg = F.dropout(msg, p=self.dropout_prob)
                msg = F.relu(self.fc2_eq10[i](msg))
                msg = msg * z[:, i:i + 1] * scale_factor #element-wise product with broadcast
                all_msgs = msg
            else:
                msg = F.relu(self.fc1_eq10[i](x_edge))
                msg = F.dropout(msg, p=self.dropout_prob)
                msg = F.relu(self.fc2_eq10[i](msg))
                msg = msg * z[:, i:i + 1] * scale_factor #element-wise product with broadcast
                all_msgs = all_msgs + msg
        return all_msgs

    # Eq 11
    def update(self, aggr_out, indexed_input):
        x_cat = torch.cat([indexed_input, aggr_out], dim=-1)
        feature_diff = F.dropout(F.relu(self.fc1_eq11(x_cat)), p=self.dropout_prob)
        feature_diff = F.dropout(F.relu(self.fc2_eq11(feature_diff)), p=self.dropout_prob)
        feature_diff = self.fc3_eq11(feature_diff)
        return indexed_input + feature_diff

class RNNDecoder_NRI(MessagePassing):
    def __init__(self, feature_dim, edge_types, n_hid, config, do_prob=0.):
        super(RNNDecoder_NRI, self).__init__(aggr='add') # Eq 14
        self.fc1_eq13 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.fc2_eq13 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_dim = n_hid
        self.do_prob = do_prob
        # Eq 15
        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(feature_dim, n_hid, bias=True)
        self.input_i = nn.Linear(feature_dim, n_hid, bias=True)
        self.input_n = nn.Linear(feature_dim, n_hid, bias=True)
        # Eq 16
        self.fc1_eq16 = nn.Linear(n_hid, n_hid)
        self.fc2_eq16 = nn.Linear(n_hid, n_hid)
        self.fc3_eq16 = nn.Linear(n_hid, feature_dim)
        #config
        self.node_feature_dim = config.get('num_node_features')
        self.num_timesteps = config.get('num_timesteps')

    def forward(self, inputs, edge_index, z, pred_steps=1):
        #[num_graphs * num_nodes, tsteps, feature_dim]
        inputs = inputs.reshape(-1, self.num_timesteps, self.node_feature_dim)
        pred_all = []
        
        for step in range(self.num_timesteps - 1):
            ins = inputs[:, step, :]
            if step == 0:
                pred, hidden = self.propagate(edge_index, x=ins, z=z, inputs=ins)
            else:
                pred, hidden = self.propagate(edge_index, x=ins, z=z, inputs=ins, hidden=hidden)
            pred_all.append(pred)
        preds = torch.stack(pred_all, dim=1)
        return preds

    def message(self, x_i, x_j, z):
        x_edge = torch.cat((x_i, x_j), dim=-1)
        norm = float(len(self.fc1_eq13))
        for i in range(len(self.fc1_eq13)):
            if i == 0:
                msg = F.tanh(self.fc1_eq13[i](x_edge))
                msg = F.dropout(msg, p=self.do_prob)
                msg = F.tanh(self.fc2_eq13[i](msg))
                msg = msg * z[:, :, i:i+1]
                all_msgs = msg / norm
            else:
                msg = F.tanh(self.fc1_eq13[i](x_edge))
                msg = F.dropout(msg, p=self.do_prob)
                msg = F.tanh(self.fc2_eq13[i](msg))
                msg = msg * z[:, :, i:i+1]
                all_msgs = all_msgs + (msg / norm)
        return all_msgs

    def update(self, aggr_msg, inputs, hidden=0):
        # Eq 15
        r = F.sigmoid(self.input_r(inputs) + self.hidden_r(aggr_msg))
        i = F.sigmoid(self.input_i(inputs) + self.hidden_i(aggr_msg))
        n = F.tanh(self.input_n(inputs) + r * self.hidden_h(aggr_msg))
        hidden = (1 - i) * n + i * hidden
        # Eq 16
        pred = F.dropout(F.relu(self.fc1_eq16(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.fc2_eq16(pred)), p=self.dropout_prob)
        pred = self.fc1_eq16(pred)
        pred = inputs + pred
        # mu, hidden(t+1)
        return pred, hidden