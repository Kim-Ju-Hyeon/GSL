import torch
import numpy as np

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))

def kl_categorical(preds, log_prior, num_nodes, eps=1e-16):
    num_edges = num_nodes * (num_nodes - 1)
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / preds.size(0) * num_edges

def kl_categorical_uniform(preds, num_nodes, num_edge_types, add_const=False, eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    num_edges = num_nodes * (num_nodes - 1)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / preds.size(0) * num_edges # sims * edges

def kl_gaussian(preds_mu, preds_log_var):
    kl_div = 0.5 * (preds_log_var.exp() + preds_mu**2 - 1 - preds_log_var)
    return kl_div.sum() / preds_mu.size(0)

def edge_accuracy(preds, target):
    #preds는 index를 return [#sims, #edges, #edge_types] -> [#sims, #edges] / element -1 dim에서의 max index
    preds_max = preds.argmax(-1) 
    correct = preds_max.float().eq(
        target.float().view_as(preds_max)).cpu().sum()
    
    return np.float(correct) / (target.size(0))

# Too slow...
def edge_accuracy_index(preds, target):
    edge_types = target.size(-1)
    target_size = target.size(0)

    preds = preds.argmax(-1).float().cpu().detach()
    target = target.float().view_as(preds).cpu().detach().numpy()
    preds = preds.numpy()

    preds_per = np.zeros_like(preds)

    for i in range(edge_types):
        idx = np.where(preds == i)
        permute = np.mean(target[idx] - preds[idx])
        preds_per[idx] = preds[idx] + np.round(permute, 0)

    correct = np.sum(preds_per == target)

    return correct / target_size

def edge_accuracy_con(preds, target):
    edge_types = target.size(-1)
    target_size = target.size(0)

    preds = preds.float().cpu().detach() #continuous value
    target = target.float().view_as(preds).cpu().detach().numpy()
    preds = preds.numpy()

    preds_per = np.zeros_like(preds)

    for i in range(edge_types):
        idx = np.where(preds == i)
        permute = np.mean(target[idx] - preds[idx])
        preds_per[idx] = preds[idx] + np.round(permute, 0)

    correct = np.sum(preds_per == target)

    return correct / target_size

def kl_pseudo(u_star, mu_sig, log_spike, mu_sig_p, log_spike_p, n_edges):
    eps = 1e-10
    batch_size = u_star.shape[0]
    T_p = torch.cat((mu_sig_p, log_spike_p), dim=-1).reshape(-1, n_edges, 3)
    expand_size = [batch_size] + list(T_p.shape)
    T_p = T_p.unsqueeze(0).expand(expand_size)

    u_s = u_star.unsqueeze(-1).unsqueeze(-1).expand_as(T_p)
    x_u = (u_s * T_p).sum(dim=1).reshape(-1, 3)    

    spike = log_spike.exp()
    mu, logvar = mu_sig[:, 0:1], mu_sig[:, 1:2]
    mu_p, logvar_p, log_gamma_p = x_u[:, 0:1], x_u[:, 1:2], x_u[:, 2:3]
    spike_p = log_gamma_p.exp()
    KL_mu_sig = spike.mul(logvar_p - logvar + 
                          (logvar.exp() + (mu - mu_p).pow(2))/(2*(logvar_p.exp())) - 0.5)
    KL_spike = (1 - spike) * (torch.log(1 - spike + eps) -  torch.log(1 - spike_p + eps))
    KL_slab = spike * (torch.log(spike + eps) - torch.log(spike_p + eps))
    KL = KL_mu_sig + KL_spike + KL_slab
    
    return KL.sum() / batch_size

def kl_spike_only(mu_sig, log_spike, alpha, n_edges, device):
    eps = 1e-6
    batch_size = int(mu_sig.shape[0] / n_edges)
    mu, logvar = mu_sig[:, 0:1], mu_sig[:, 1:2]
    alpha = alpha.to(device) #[num_edges] > [1, edges, 1] > [batch, edges, 1] > [batch * edges, 1]
    alpha_expand = alpha.unsqueeze(0).unsqueeze(-1)
    alpha_expand = alpha_expand.expand(batch_size, n_edges, 1).reshape(-1, 1) 
    alpha_expand = torch.clamp(alpha_expand, eps, 1 - eps)
    spike = torch.clamp(log_spike.exp(), eps, 1-eps) #[batch * edges, 1]

    KL_mu_sig = -0.5 * spike.mul(1 + logvar - logvar.exp().sqrt() - mu.pow(2))
    KL_spike = (1 - spike) * (torch.log(1 - spike) - torch.log(1 - alpha_expand))
    KL_slab = spike * (torch.log(spike) - torch.log(alpha_expand))
    KL = KL_mu_sig + KL_slab + KL_spike
    return KL.sum() / mu_sig.shape[0]

def penalty_term(log_spike_p, u_star, alpha, n_edges, device):
    eps = 1e-10
    batch_size = u_star.shape[0]
    expand_size = [batch_size] + list(log_spike_p.reshape(-1, n_edges, 1).shape)
    gamma = log_spike_p.exp().reshape(-1, n_edges, 1).unsqueeze(0).expand(expand_size) # 128 20 20 1

    u_s = u_star.unsqueeze(-1).unsqueeze(-1).expand_as(gamma)
    gamma_bar = (u_s * gamma).sum(1).squeeze() #128 20
    alpha = alpha.unsqueeze(0).expand_as(gamma_bar).to(device)
    KL = gamma_bar.mul(torch.log(gamma_bar + eps) - torch.log(alpha + eps)).sum()
    return KL * n_edges / log_spike_p.shape[0]