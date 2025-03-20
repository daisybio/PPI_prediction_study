import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.data as d


class baseline2d(nn.Module):
    def __init__(self, embed_dim, h3 = 64, kernel_size = 2, pooling = 'avg'):
        super(baseline2d, self).__init__()

        if embed_dim < 30:  #usually just the case for one-hot-encoding / might need to check this differently
            h = h3
            h2 = h3
            h3 = h3
        else:
            h = int(embed_dim//4)
            h2 = int(h//4)   
            h3 = h3 

        self.conv = nn.Conv2d(h3, 1, kernel_size=kernel_size, padding='same')
        if pooling == 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size)
        elif pooling == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)  

        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, h)
        #self.bn1 = nn.BatchNorm1d(h)
        self.fc2 = nn.Linear(h, h2)
        #self.bn2 = nn.BatchNorm1d(h2)

        self.fc3 = nn.Linear(h2, h3)
        #self.bn3 = nn.BatchNorm1d(h3)
        #self.fc4 = nn.Linear(h3, 1)    

        self.sigmoid = nn.Sigmoid()


    def forward(self, x1 = None, x2 = None):

        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        x1 = self.ReLU(self.fc1(x1))
        x1 = self.ReLU(self.fc2(x1))
        x1 = self.ReLU(self.fc3(x1))
        
        x2 = self.ReLU(self.fc1(x2))
        x2 = self.ReLU(self.fc2(x2))
        x2 = self.ReLU(self.fc3(x2))



        mat = torch.einsum('ik,jk->ijk', x1, x2)    # normale matrix multiplikation
        mat = mat.permute(2, 0, 1)
        
        mat = self.conv(mat.unsqueeze(0))

        x = self.pool(mat)    
                   
        m = torch.max(x)

        pred = self.sigmoid(m)
        pred = pred[None]

        return pred, mat



    def shifted_sigmoid(x, balance_point):
        x = balance_point + torch.sigmoid((x - balance_point) * 10)
        return x
    

    def batch_iterate(self, batch, device, layer, emb_dir, embedding=True):
            pred = []
            for i in range(len(batch['interaction'])):
                id1 = batch['name1'][i]
                id2 = batch['name2'][i]
                if embedding:
                    seq1 = d.get_embedding_per_tok(emb_dir, id1, layer).to(device)
                    seq2 = d.get_embedding_per_tok(emb_dir, id2, layer).to(device)
                else:
                    seq1 = batch['sequence_a'][i]
                    seq2 = batch['sequence_b'][i]
                    seq1 = d.sequence_to_vector(seq1)
                    seq2 = d.sequence_to_vector(seq2)
                    seq1 = torch.tensor(np.array(seq1)).to(device)
                    seq2 = torch.tensor(np.array(seq2)).to(device)
                p, cm = self.forward(seq1, seq2)
                pred.append(p)
            return torch.stack(pred)  

