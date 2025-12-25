import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.FloatTensor')


def sparsity(arr, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def recon_loss(de_features, nor_features):
    recon_loss = torch.nn.MSELoss()
    loss_recon = recon_loss(de_features, nor_features)
    return loss_recon


def mil_loss(args, y_pred, segment, ncrop):
    topk_pred = torch.max(y_pred.view([args.batch_size * 2, ncrop, segment]), dim=-1, keepdim=False)[0]
    nor_max = topk_pred[:args.batch_size]
    abn_max = topk_pred[args.batch_size:]
    err = 0
    for i in range(args.batch_size):
        err += torch.sum(F.relu(1-abn_max+nor_max[i]))
    err = err/(args.batch_size**2)
    loss = err
    return loss


class bce_loss(torch.nn.Module):
    def __init__(self, batch_size, ncrop, device):
        super(bce_loss, self).__init__()
        self.batch_size = batch_size
        self.ncrop = ncrop
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

    def forward(self, score_normal, score_abnormal, nlabel, alabel, model, ncrop, lamda5):
        score = torch.cat((score_normal, score_abnormal), 0)
        nlabel = torch.zeros_like(alabel)
        if model == 'lin':
            alabel = torch.where(alabel > 0.85, torch.tensor(1.0), torch.tensor(0.0))
        else:
            alabel = torch.where(alabel > 0.45, torch.tensor(1.0), torch.tensor(0.0))
        anum = torch.sum(alabel.eq(1.0))
        nnum = torch.sum(alabel.eq(0.0)) + self.batch_size * ncrop * 64
        label = torch.cat((nlabel, alabel), 0).to(self.device)
        loss_cls = torch.mean(-(1 - label) * torch.log(1 - score + (1e-8)) - nnum/anum * label * torch.log(score + (1e-8)))

        return lamda5*loss_cls


def train(nloader, aloader, model_lin, model_mem, args, optimizer_lin, optimizer_mem, viz, step, device):
    with torch.set_grad_enabled(True):
        model_lin.train()
        model_mem.train()

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)

        input = torch.cat((ninput, ainput), 0).to(device)

        scores_lin, feat_lin = model_lin(input, train=True)

        scores_mem, nor_features, de_features, feat_mem = model_mem(input, train=True)

        scores_mem = scores_mem.view(args.batch_size * args.plot_freq * 32 * 2, -1)
        scores_mem = scores_mem.squeeze()
        abn_scores_mem = scores_mem[args.batch_size * args.plot_freq * 32:]
        nor_scores_mem = scores_mem[0:args.batch_size * args.plot_freq * 32]

        scores_lin = scores_lin.view(args.batch_size * args.plot_freq * 32 * 2, -1)
        scores_lin = scores_lin.squeeze()
        abn_scores_lin = scores_lin[args.batch_size * args.plot_freq * 32:]
        nor_scores_lin = scores_lin[0:args.batch_size * args.plot_freq * 32]

        nlabel = nlabel[0:args.batch_size]
        alabel = alabel[0:args.batch_size]

        loss_criterion = bce_loss(args.batch_size, args.plot_freq, device)

        loss_mil_mem = mil_loss(args, scores_mem, 32, args.plot_freq)
        loss_sparse_mem = sparsity(abn_scores_mem, 8e-3)

        loss_mil_lin = mil_loss(args, scores_lin, 32, args.plot_freq)
        loss_sparse_lin = sparsity(abn_scores_lin, 8e-3)

        loss_recon = recon_loss(de_features, nor_features)

        if step < 0:
            cost_mem = loss_mil_mem + loss_sparse_mem
            cost_lin = loss_mil_lin + loss_sparse_lin

            viz.plot_lines('loss_mem', cost_mem.item())
            viz.plot_lines('loss_lin', cost_lin.item())

            optimizer_lin.zero_grad()
            cost_lin.backward()
            optimizer_lin.step()

            optimizer_mem.zero_grad()
            cost_mem.backward()
            optimizer_mem.step()

        else:
            loss_bce_mem = loss_criterion(nor_scores_mem, abn_scores_mem, nlabel, abn_scores_lin, 'lin',  args.plot_freq, 0.8)
            cost_mem = loss_mil_mem + loss_recon + loss_bce_mem + loss_sparse_mem

            loss_bce_lin = loss_criterion(nor_scores_lin, abn_scores_lin, nlabel, abn_scores_mem, 'mem', args.plot_freq, 0.8)
            cost_lin = loss_mil_lin + loss_bce_lin + loss_sparse_lin

            viz.plot_lines('loss_mem', cost_mem.item())
            viz.plot_lines('loss_lin', cost_lin.item())

            optimizer_lin.zero_grad()
            cost_lin.backward()
            optimizer_lin.step()

            optimizer_mem.zero_grad()
            cost_mem.backward()
            optimizer_mem.step()
