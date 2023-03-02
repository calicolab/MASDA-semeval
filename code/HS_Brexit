import json
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as op
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer


NUM_ANN = 6
LM_DIM, ANN_LAT_DIM = 384, 5
DEC, GOLD = -2, -1
fp = ''


def calc_eval(y_true, y_soft, y_gold, y_pred_soft, y_pred_gold):
    yps_pos = torch.sum(y_pred_soft, dim=1) / y_pred_soft.shape[1]
    yps_neg = 1 - yps_pos
    yps = torch.cat((yps_neg[:, None], yps_pos[:, None]), dim=1)
    predictions = np.clip(yps.numpy(), 1e-12, 1 - 1e-12) + 1e-9
    cross_entropy = -1 * (np.sum(y_soft.numpy() * np.log(predictions)) / predictions.shape[0])

    y_pred_gold.apply_(lambda z: 0 if z < 0.5 else 1)
    f1 = f1_score(y_gold.tolist(), y_pred_gold.tolist(), average='micro')

    y_pred_soft.apply_(lambda z: 1 if z > 0.5 else 0)
    y_true_ = y_true.detach().cpu()
    accuracy = (torch.sum(y_pred_soft == y_true_).item() / (y_true_.shape[0] * y_true_.shape[1])) * 100

    return cross_entropy, accuracy, f1


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class GoldNet(nn.Module):
    def __init__(self):
        super(GoldNet, self).__init__()
        self.attn = nn.Linear(LM_DIM, NUM_ANN)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x_list, x_):
        x_ = self.dropout(x_)
        attn = self.attn(x_)
        attn = torch.softmax(attn, dim=1)
        x = torch.sum(x_list * attn[:, None, :], dim=2)

        return x


class AnnNet(nn.Module):
    def __init__(self):
        super(AnnNet, self).__init__()
        self.fc = nn.Linear(LM_DIM, ANN_LAT_DIM)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)

        return x


class DecNet(nn.Module):
    def __init__(self):
        super(DecNet, self).__init__()
        self.fc = nn.Linear(ANN_LAT_DIM, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)

        return x


class NNModel:
    def __init__(self):
        self.lm = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.nets = nn.ModuleList([AnnNet() for _ in range(NUM_ANN)] + [DecNet(), GoldNet()])
        self.loss = nn.BCEWithLogitsLoss()
  
    def train(self, x, y, y_s, y_h, lr=1e-3, lr_lm=1e-4, weight_decay=0, patience=100, print_rate=10):
        self.nets.train()
        self.lm.train()
        optimizer = op.Adam(self.nets.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_lm = op.Adam(self.lm.parameters(), lr=lr_lm, weight_decay=weight_decay)

        target_loss = 0.15
        loss_val = 100
        i = 0

        while loss_val > target_loss:
            h_soft, h_hard = self.exec(x, x['attention_mask'])
            loss = self.loss(torch.cat((h_soft, h_hard[:, None]), dim=1), torch.cat((y, y_h[:, None]), dim=1))
            optimizer.zero_grad()
            optimizer_lm.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_lm.step()

            loss_val = loss.item()

            if i % print_rate == print_rate - 1:
                print('Epoch: ' + str(i + 1))
                print('Loss: ' + str(loss_val))
                print()

            i += 1

        val_loss, val_s, val_h = self.test(x, y, y_h)
        ce, acc, f1 = calc_eval(y, y_s, y_h, val_s, val_h)

        print('\nFinal train loss: ' + str(val_loss))
        print('CE: ' + str(ce))
        print('Accuracy: ' + str(acc))
        print('F1: ' + str(f1))

    def test(self, x, y_s, y_h, eval=False):
        self.nets.eval()
        self.lm.eval()

        with torch.no_grad():
            y_soft, y_hard = self.exec(x, x['attention_mask'])

        if eval:
            return torch.sigmoid(y_soft).detach().cpu(), torch.sigmoid(y_hard).detach().cpu()

        with torch.no_grad():
            loss = self.loss(torch.cat((y_soft, y_hard[:, None]), dim=1), torch.cat((y_s, y_h[:, None]), dim=1))

        return loss.item(), torch.sigmoid(y_soft).detach().cpu(), torch.sigmoid(y_hard).detach().cpu()


    def exec(self, x, attn_mask):
        x_ = self.lm(**x)
        h_ = mean_pooling(x_, attn_mask)
        ann_out, ann_lat = [], []
        
        for k in range(NUM_ANN):
            out_ = self.nets[k](h_)
            ann_lat.append(out_.detach()[:, :, None])
            ann_out.append(self.nets[DEC](out_))

        pred_hard = self.nets[GOLD](torch.cat(ann_lat, dim=2), h_.detach())
        pred_hard = self.nets[DEC](pred_hard)

        return torch.cat(ann_out, dim=1), torch.flatten(pred_hard)


with open(fp + 'train_x.json', 'r') as f:
    train_file = json.load(f)

with open(fp + 'test_x.json', 'r') as f:
    test_file = json.load(f)

with open(fp + 'eval_x.json', 'r') as f:
    eval_file = json.load(f)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

with torch.no_grad():
    x_train = tokenizer(train_file + test_file, padding=True, truncation=True, return_tensors='pt')
    x_eval = tokenizer(eval_file, padding=True, truncation=True, return_tensors='pt')

y_train = torch.load(fp + 'y_train.pt')
y_test = torch.load(fp + 'y_test.pt')
y_train = torch.cat((y_train, y_test), dim=0)

y_train_soft = torch.load(fp + 'soft_train_y.pt')
y_test_soft = torch.load(fp + 'soft_test_y.pt')
y_train = torch.cat((y_train_soft, y_test_soft), dim=0)

y_train_gold = torch.load(fp + 'gold_train_y.pt')
y_test_gold = torch.load(fp + 'gold_test_y.pt')
y_train_gold = torch.cat((y_train_gold, y_test_gold), dim=0)

DEVICE, dtype = ('cuda:0', torch.cuda.FloatTensor) if torch.cuda.is_available() else ('cpu', torch.float32)

model = NNModel()
model.nets.to(DEVICE)
model.lm.to(DEVICE)
x_train.to(DEVICE)
y_train.to(DEVICE)
y_train_soft.to(DEVICE)
y_train_gold.to(DEVICE)
y_test = y_test.type(dtype)
y_train = y_train.type(dtype)
y_train_gold = y_train_gold.type(dtype)
y_train_soft = y_train_soft.type(dtype)

model.train(x_train, y_train, y_train_soft, y_train_gold, lr=1e-3, lr_lm=1e-5, weight_decay=1e-4, print_rate=10)

x_eval.to(DEVICE)
eval_soft, eval_hard = model.test(x_eval, None, None, eval=True)
te_soft_list = eval_soft.tolist()
te_hard_list = eval_hard.tolist()

with open(fp + 'HS-Brexit_results.tsv', 'w') as f:
    for i_ in range(len(te_hard_list) - 1):
        pos = sum(te_soft_list[i_]) / len(te_soft_list[i_])
        hard = 0 if te_hard_list[i_] < 0.5 else 1
        f.write(str(hard) + '\t' + str(1 - pos) + '\t' + str(pos) + '\n')

    pos = sum(te_soft_list[-1]) / len(te_soft_list[-1])
    hard = 0 if te_hard_list[-1] < 0.5 else 1
    f.write(str(hard) + '\t' + str(1 - pos) + '\t' + str(pos))
