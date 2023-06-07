import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import SequentialSampler, RandomSampler
import utils
import torch.nn.functional as F

max_length = 80
num_point  = 82
embed_dim  = 512
num_head   = 4
num_block  = 1
num_class  = 250
num_landmark = 543
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, x):
        return self.mlp(x)

#https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
class MultiHeadAttention(nn.Module):
    def __init__(self,
            embed_dim,
            num_head,
            batch_first,
        ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_head,
            bias=True,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            dropout=0.0,
            batch_first=batch_first,
        )

    def forward(self, x, x_mask):
        out, _ = self.mha(x,x,x, key_padding_mask=x_mask)
        return out

class TransformerBlock(nn.Module):
    def __init__(self,
        embed_dim,
        num_head,
        out_dim,
        batch_first=True,
    ):
        super().__init__()
        self.attn  = MultiHeadAttention(embed_dim, num_head,batch_first)
        self.ffn   = FeedForward(embed_dim, out_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x, x_mask=None):
        x = x + self.attn((self.norm1(x)), x_mask)
        x = x + self.ffn((self.norm2(x)))
        return x


def positional_encoding(length, embed_dim):
    dim = embed_dim//2
    position = np.arange(length)[:, np.newaxis]     # (seq, 1)
    dim = np.arange(dim)[np.newaxis, :]/dim   # (1, dim)
    angle = 1 / (10000**dim)         # (1, dim)
    angle = position * angle    # (pos, dim)
    pos_embed = np.concatenate(
        [np.sin(angle), np.cos(angle)],
        axis=-1
    )
    pos_embed = torch.from_numpy(pos_embed).float()
    return pos_embed

def pack_seq(
    seq,
):
    length = [min(s.shape[0], max_length)  for s in seq]
    batch_size = len(seq)
    K = seq[0].shape[1]
    L = max(length)

    x = torch.zeros((batch_size, L, K, 3)).to(seq[0].device)
    x_mask = torch.zeros((batch_size, L)).to(seq[0].device)
    for b in range(batch_size):
        l = length[b]
        x[b, :l] = seq[b][:l]
        x_mask[b, l:] = 1
    x_mask = (x_mask>0.5)
    x = x.reshape(batch_size,-1,K*3)
    return x, x_mask

class Net(nn.Module):

    def __init__(self, num_class=num_class):
        super().__init__()
        self.output_type = ['inference', 'loss']

        pos_embed = positional_encoding(max_length, embed_dim)
        # self.register_buffer('pos_embed', pos_embed)
        self.pos_embed = nn.Parameter(pos_embed)

        self.cls_embed = nn.Parameter(torch.zeros((1, embed_dim)))
        self.x_embed = nn.Sequential(
            nn.Linear(num_point * 3, embed_dim, bias=False),
        )

        self.encoder = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_head,
                embed_dim,
            ) for i in range(num_block)
        ])
        self.logit = nn.Linear(embed_dim, num_class)

    def forward(self, batch):
        xyz = batch['xyz']
        x, x_mask = pack_seq(xyz)
        B,L,_ = x.shape
        x = self.x_embed(x)
        x = x + self.pos_embed[:L].unsqueeze(0)

        x = torch.cat([
            self.cls_embed.unsqueeze(0).repeat(B,1,1),
            x
        ],1)
        x_mask = torch.cat([
            torch.zeros(B,1).to(x_mask),
            x_mask
        ],1)


        #x = F.dropout(x,p=0.25,training=self.training)
        for block in self.encoder:
            x = block(x,x_mask)

        cls = x[:,0]
        cls = F.dropout(cls,p=0.4,training=self.training)
        logit = self.logit(cls)

        output = {}
        if 'loss' in self.output_type:
            output['label_loss'] = F.cross_entropy(logit, batch['label'])

        if 'inference' in self.output_type:
            output['sign'] = torch.softmax(logit,-1)

        return output
    
class IncludeDataset(Dataset):
    def __init__(self, df, augment=None):
        self.df = df
        self.augment = augment
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]

        csv_file = d.path
        xyz = utils.load_relevant_data_subset(csv_file, type="csv")
        xyz = xyz - xyz[~np.isnan(xyz)].mean(0,keepdims=True) #noramlisation to common maen
        xyz = xyz / xyz[~np.isnan(xyz)].std(0, keepdims=True)

        if self.augment is not None:
            xyz = self.augment(xyz)
        xyz = torch.from_numpy(xyz).float()
        xyz = utils.pre_process(xyz)

        r = {}
        r['index'] = index
        r['d'    ] = d
        r['xyz'  ] = xyz
        r['label'] = d.label
        r['category'] = d.category
        return r

tensor_key = ['xyz', 'label']
def null_collate(batch):
    batch_size = len(batch)
    d = {}
    key = batch[0].keys()
    for k in key:
        d[k] = [b[k] for b in batch]
    d['label'] = torch.LongTensor(d['label'])
    return d

