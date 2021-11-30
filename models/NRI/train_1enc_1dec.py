import time
import os
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch_geometric.data import Data

import yaml
from generate_data import *
from modules import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='0')

args = parser.parse_args()

with open('config/settings{}.yml'.format(args.config), encoding='UTF8') as f:
    setting = yaml.load(f, Loader=yaml.FullLoader)
settings = setting.get('train')

# Hyper Parameter Settings
sim_setting = settings.get('sim_setting')

data_suffix = sim_setting.get('data')
num_nodes = sim_setting.get('num_nodes')
num_node_features = sim_setting.get('num_node_features')
num_timesteps = sim_setting.get('num_timesteps')
edge_types = sim_setting.get('edge_types')
device = sim_setting.get('device')
pred_step = sim_setting.get('pred_step')
batch_size = sim_setting.get('batch_size')
edge_type_explicit = sim_setting.get('edge_type_explicit')
prior_not_uniform = sim_setting.get('prior_not_uniform')

model_params = settings.get('model_params')

enc_hid = model_params.get('encoder').get('num_hidden')
dec_msg_hid = model_params.get('decoder').get('msg_hidden')
dec_msg_out = model_params.get('decoder').get('msg_out')
dec_hid = model_params.get('decoder').get('num_hidden')
dec_drop = model_params.get('decoder').get('do_prob')
lr = model_params.get('training').get('lr')
lr_decay = model_params.get('training').get('lr_decay')
gamma = model_params.get('training').get('gamma')
save_folder = model_params.get('model').get('save_folder') + data_suffix
var = model_params.get('model').get('output_var')
epochs = model_params.get('model').get('epoch')

# Change suffix with intrested hyper-params (enc, dec, edge_explicit, prior_not_uniform, ...)
edge_explicit = 'T' if edge_type_explicit else 'F'
suffix = '{}_edge{}_expl{}'.format(data_suffix, edge_types, edge_explicit)

encoder_file = os.path.join(save_folder, 'encoder_{}.pt'.format(suffix))
decoder_file = os.path.join(save_folder, 'decoder_{}.pt'.format(suffix))
log_file = os.path.join(save_folder, 'log_{}.txt'.format(suffix))
log_degenerate = os.path.join(save_folder, 'log_degeneracy_{}.txt'.format(suffix))

if not os.path.exists(save_folder):
    os.mkdir(save_folder)
log = open(log_file, 'w')
log_degen = open(log_degenerate, 'w')

device = torch.device(device if torch.cuda.is_available() else 'cpu')

encoder = MLPEncoder_NRI(num_node_features*num_timesteps, enc_hid, edge_types)
decoder = MLPDecoder_NRI(num_node_features, edge_types, dec_msg_hid, 
                         dec_msg_out, dec_hid, sim_setting)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=gamma)

# Loading datset
if edge_types == 2:
    data_train = spring5_edge2_train(data_dir)
    data_valid = spring5_edge2_valid(data_dir)
    data_test = spring5_edge2_test(data_dir)

elif edge_types == 3:
    data_train = spring5_edge3_train(data_dir)
    data_valid = spring5_edge3_valid(data_dir)
    data_test = spring5_edge3_test(data_dir)

# batch_size: num of graphs(sims)
train_loader = DL_PyG(data_train, batch_size=batch_size, shuffle=False) 
valid_loader = DL_PyG(data_valid, batch_size=batch_size, shuffle=False)
test_loader = DL_PyG(data_test, batch_size=batch_size, shuffle=False)

def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []

    encoder.train()
    decoder.train()
    
    encoder.to(device)
    decoder.to(device)

    if prior_not_uniform:
        prior_z = np.array(sim_setting.get('prior_z'))
        prior_z = prior_z / prior_z.sum()
        log_prior_z = torch.from_numpy(np.log(prior_z)).float()
        log_prior_z = torch.unsqueeze(log_prior_z, 0).to(device)
    # _ > batch_index (use batch_idx, control iteration with lambda)
    for _, data in tqdm(enumerate(train_loader)):
        data = data.to(device)
        x_data, edge_index, edge_target = data.x, data.edge_index, data.y

        optimizer.zero_grad()
        logits = encoder(x_data, edge_index)
        
        z = F.gumbel_softmax(logits, hard=False)
        prob = F.softmax(z, dim=-1) # prob of discrete relation

        output = decoder(x_data, edge_index, z, pred_step) #prob here

        target = x_data.reshape((-1, num_timesteps, num_node_features))[:, 1:, :]

        loss_reconstruction = nll_gaussian(output, target, var)
        if prior_not_uniform:
            loss_kl = kl_categorical(prob, log_prior_z, num_nodes)
        else:
            loss_kl = kl_categorical_uniform(prob, num_nodes, edge_types)        
        loss = loss_reconstruction + loss_kl

        if edge_type_explicit:
            acc = edge_accuracy(logits, edge_target)
            acc_train.append(acc)
        elif (not edge_type_explicit):
            acc = edge_accuracy_index(logits, edge_target)
            acc_train.append(acc)

        loss.backward()
        optimizer.step()

        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_reconstruction.item())
        kl_train.append(loss_kl.item())

    
    scheduler.step()

    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            data = data.to(device)
            x_data, edge_index, edge_target = data.x, data.edge_index, data.y
            
            logits = encoder(x_data, edge_index)
            
            z = F.gumbel_softmax(logits, hard=True)
            prob = F.softmax(z, dim=-1) # prob of discrete relation

            output = decoder(x_data, edge_index, z, pred_step) #prob here

            target = x_data.reshape((-1, num_timesteps, num_node_features))[:, 1:, :]

            loss_reconstruction = nll_gaussian(output, target, var)
            if prior_not_uniform:
                loss_kl = kl_categorical(prob, log_prior_z, num_nodes)
            else:
                loss_kl = kl_categorical_uniform(prob, num_nodes, edge_types)
            loss = loss_reconstruction + loss_kl

            if batch_idx % 100 == 0 and edge_type_explicit:
                acc = edge_accuracy(logits, edge_target)
                acc_val.append(acc)
            elif batch_idx % 100 == 0 and (not edge_type_explicit):
                acc = edge_accuracy_index(logits, edge_target)
                acc_val.append(acc)

            mse_val.append(F.mse_loss(output, target).item())
            nll_val.append(loss_reconstruction.item())
            kl_val.append(loss_kl.item())
        
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
              'kl_val: {:.10f}'.format(np.mean(kl_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              'acc_val: {:.5f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t))
        
        if np.mean(nll_val) < best_val_loss:
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            print('Best model so far, saving...')
            print('Epoch: {:04d}'.format(epoch),
                  'nll_train: {:.10f}'.format(np.mean(nll_train)),
                  'kl_train: {:.10f}'.format(np.mean(kl_train)),
                  'mse_train: {:.10f}'.format(np.mean(mse_train)),
                  'acc_train: {:.10f}'.format(np.mean(acc_train)),
                  'nll_val: {:.10f}'.format(np.mean(nll_val)),
                  'kl_val: {:.10f}'.format(np.mean(kl_val)),
                  'mse_val: {:.10f}'.format(np.mean(mse_val)),
                  'acc_val: {:.5f}'.format(np.mean(acc_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()
    return np.mean(nll_val)
'''
def test_degeneracy(pred_steps):
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []

    encoder.train()
    decoder.train()
    
    encoder.to(device)
    decoder.to(device)

    for batch_idx, data in tqdm(enumerate(train_loader)):
        data = data.to(device)
        x_data, edge_index, edge_target = data.x, data.edge_index, data.y

        optimizer.zero_grad()
        logits = encoder(x_data, edge_index)
        z = F.gumbel_softmax(logits)
        
        prob = F.softmax(z, dim=-1)
        output = decoder(x_data, edge_index, z, pred_steps) #prob here

        target = x_data.reshape((-1, num_timesteps, num_node_features))[:, 1:, :]
        
        loss_reconstruction = nll_gaussian(output, target, var)
        loss_kl_uniform_prior = kl_categorical_uniform(prob, num_nodes, edge_types)
        loss = loss_reconstruction + loss_kl_uniform_prior

        if batch_idx % 100 == 0:
            acc = edge_accuracy_index(logits, edge_target)
            acc_train.append(acc)

        loss.backward()
        optimizer.step()
cat
        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_reconstruction.item())
        kl_train.append(loss_kl_uniform_prior.item())
    
    scheduler.step()

    print('nll_train: {:.10f}'.format(np.mean(nll_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'acc_train: {:.10f}'.format(np.mean(acc_train)))

    return np.mean(acc_train)
'''
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0

for epoch in range(epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
'''
cnt_degenerate = 0

for sim in range(100):
    encoder = MLPEncoder_NRI(num_node_features*num_timesteps, enc_hid, edge_types)
    decoder = MLPDecoder_NRI(num_node_features, edge_types, dec_msg_hid, dec_msg_out, dec_hid, sim_setting)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=gamma)

    acc_degeneracy = []
    for _ in range(2):
        acc_degeneracy.append(test_degeneracy(pred_step))
    if acc_degeneracy[0] > acc_degeneracy[1]:
        print('sim: {} encoder degenerated'.format(sim))
        cnt_degenerate += 1
    else:
        print('sim: {} encoder well trained, {}'.format(sim, 100-cnt_degenerate))


print('Pred {} steps:'.format(pred_step), file=log_degen) 
log_degen.flush()
print('Degeneracy percentage: {}%'.format(cnt_degenerate), file=log_degen)
log_degen.flush()
'''