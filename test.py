import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from dataset import Dataset
import option
from config import *
from utils import Visualizer
from model import Model_lin
from memory import Model_mem
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test(dataloader, model_lin, model_mem, args, viz, device):
    with torch.no_grad():
        model_lin.eval()
        model_mem.eval()
        pred = torch.zeros(0, device=device)

        if args.dataset in ['sh', 'shanghai']:
            gt = np.load('list/gt-sh.npy')
        elif args.dataset == "ucf":
            gt = np.load('list/gt-ucf.npy')
        elif args.dataset == "xd":
            gt = np.load('list/gt-xd.npy')

        result = {}
        for i, inputs in enumerate(dataloader):
            input, name = inputs
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            logits_lin = model_lin(inputs=input, train=False)
            logits_mem = model_mem(inputs=input, train=False)
            logits = (logits_lin+logits_mem)/2
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))
            result[name] = logits

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('outputs/fpr.npy', fpr)
        np.save('outputs/tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('outputs/precision.npy', precision)
        np.save('outputs/recall.npy', recall)

        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)

        if args.dataset == "xd":
            np.save(f"outputs/pr_auc.npy", pr_auc)
            print('AP: ' + str(pr_auc))
            return pr_auc, list(gt), pred, result
        else:
            print('AUC @ ROC: ' + str(rec_auc))
            return rec_auc, list(gt), pred, result


if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    viz = Visualizer(env='Test', use_incoming_socket=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_lin = Model_lin(args.feature_size, args.batch_size).to(device)
    model_mem = Model_mem(args.feature_size, args.batch_size).to(device)

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    save_root = "/home/xinghongjie/xuchen/Project/CoLG"

    model_lin_dict = model_lin.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_root + '/models/CoLG-xd-lin.pkl').items()})
    model_mem_dict = model_mem.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_root + '/models/CoLG-xd-mem.pkl').items()})

    auc, gt, pred, result = test(test_loader, model_lin, model_mem, args, viz, device)