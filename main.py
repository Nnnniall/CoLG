from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model_lin
from memory import Model_mem
from dataset import Dataset
from train import train
from test import test
import option
from tqdm import tqdm
from utils import Visualizer
from config import *
import numpy as np
import torch.nn.functional as F

viz = Visualizer(env='CoLG', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model_lin = Model_lin(args.feature_size, args.batch_size)
    model_mem = Model_mem(args.feature_size, args.batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_lin = model_lin.to(device)
    model_mem = model_mem.to(device)

    if not os.path.exists('models'):
        os.makedirs('models')

    optimizer_lin = optim.Adam(model_lin.parameters(), lr=config.lr[0], weight_decay=0.0005)
    optimizer_mem = optim.Adam(model_mem.parameters(), lr=config.lr[0], weight_decay=0.0005)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = 'outputs/'  # put your own path here

    auc, gt, pred, result = test(test_loader, model_lin, model_mem, args, viz, device)

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer_lin.param_groups:
                param_group["lr"] = config.lr[step - 1]
            for param_group in optimizer_mem.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(loadern_iter, loadera_iter, model_lin, model_mem, args, optimizer_lin, optimizer_mem, viz, step, device)

        if step % 5 == 0 and step > 200:

            auc, gt, pred, result = test(test_loader, model_lin, model_mem, args, viz, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save(model_lin.state_dict(), './models/' + args.model_name + '{}-lin.pkl'.format(step))
                torch.save(model_mem.state_dict(), './models/' + args.model_name + '{}-mem.pkl'.format(step))
                save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(step)))
    torch.save(model_lin.state_dict(), './models/' + args.model_name + 'lin_final.pkl')
    torch.save(model_mem.state_dict(), './models/' + args.model_name + 'mem_final.pkl')
