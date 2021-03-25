# coding: utf-8
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model_tesan import TESAN
from parameter import parse_args
from load_data import load_data


args = parse_args()
w2v_model = pickle.load(open(args.WORD2VEC_DIR, 'rb'))

# load data for NTM
data_bow = np.load('dataset/data_bow_ltf.npy')
data_tr = data_bow[:args.train_size]
data_te = data_bow[args.train_size:]
tensor_tr = torch.from_numpy(data_tr).float()
tensor_te = torch.from_numpy(data_te).float()

# load data for san
text_data, label = load_data()
label_tr = torch.from_numpy(label[:args.train_size]).float()
label_te = torch.from_numpy(label[args.train_size:]).float()
print('Data loaded')


def get_batch(text_data, w2v_model, indices):
    batch_size = len(indices)
    text_length = []
    for idx in indices:
        text_length.append(len(text_data[idx]))
    batch_x = np.zeros((batch_size, max(text_length), args.in_dim), dtype=np.float32)
    for i, idx in enumerate(indices, 0):
        for j, word in enumerate(text_data[idx], 0):
            batch_x[i][j] = w2v_model[word]

    return batch_x

def make_mask(text_data, indices, sent_length):
    batch_size = len(indices)
    text_length = [len(text_data[idx]) for idx in indices]
    mask = np.full((batch_size, sent_length, 1), float('-inf'), dtype=np.float32)
    for i in range(batch_size):
        mask[i][0:text_length[i]] = 0.0
    return mask


# ---------- network ----------
net_arch = args
net_arch.num_input = tensor_tr.shape[1]
net = TESAN(net_arch).cuda()
optimizer = optim.Adam(net.parameters(), args.lr, betas=(args.momentum, 0.999), weight_decay=args.wd)
criterion = nn.KLDivLoss()

# train
for epoch in range(args.num_epoch):
    print('Epoch: ', epoch+1)
    all_indices = torch.randperm(tensor_tr.size(0)).split(args.batch_size)
    loss_epoch = 0.0
    acc = 0.0
    net.train()
    for i, batch_indices in enumerate(all_indices, 1):
        # get a batch of wordvecs
        batch_x = get_batch(text_data, w2v_model, batch_indices)
        batch_x = Variable(torch.from_numpy(batch_x).float()).cuda()
        batch_mask = make_mask(text_data, batch_indices, batch_x.shape[1])
        batch_mask = Variable(torch.from_numpy(batch_mask).float()).cuda()

        input = Variable(tensor_tr[batch_indices]).cuda()
        y = Variable(label_tr[batch_indices]).cuda()

        recon, loss, out = net(input, batch_x, batch_mask, compute_loss=True)
        _, pred = torch.max(out, dim=1)
        _, truth = torch.max(y, dim=1)
        num_correct = (pred==truth).sum()
        acc += num_correct.data[0]
        # sentiment loss
        sent_loss = criterion(out, y)
        total_loss = args.L*loss+sent_loss
        # optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # report
        loss_epoch += total_loss.data[0]
    print('Train Loss={:.4f}, Train Acc={:.4f}'.format(loss_epoch/len(all_indices), acc/args.train_size))

    # test
    all_indices = torch.arange(tensor_te.size(0)).long().split(args.batch_size)
    loss_epoch = []
    acc = []
    test_ap = []
    net.eval()
    for batch_indices in all_indices:
        # get a batch of wordvecs
        batch_x = get_batch(text_data, w2v_model, batch_indices+tensor_tr.shape[0])
        batch_x = Variable(torch.from_numpy(batch_x).float(), volatile=True).cuda()
        batch_mask = make_mask(text_data, batch_indices+tensor_tr.shape[0], batch_x.shape[1])
        batch_mask = Variable(torch.from_numpy(batch_mask).float(), volatile=True).cuda()

        input = Variable(tensor_te[batch_indices], volatile=True).cuda()
        y = Variable(label_te[batch_indices], volatile=True).cuda()

        recon, loss, out = net(input, batch_x, batch_mask, compute_loss=True)
        _, pred = torch.max(out, dim=1)
        _, truth = torch.max(y, dim=1)
        acc.extend((pred==truth).cpu().data.tolist())
        sent_loss = criterion(out, y)
        # AP
        out_exp = np.power(np.e, out.cpu().data.numpy())
        y_numpy = y.cpu().data.numpy()
        test_ap.extend([np.corrcoef(out_exp[i], y_numpy[i])[0, 1] for i in range(out_exp.shape[0])])
        # loss
        loss_epoch.extend((args.L*loss+sent_loss).cpu().data.tolist())
    print('Test Loss ={:.2f},  Test Acc ={:.4f}, Test AP={:.4f}'.format(np.mean(loss_epoch),np.mean(acc),np.mean(test_ap)))