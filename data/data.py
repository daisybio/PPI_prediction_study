import torch
import torch.utils.data as data
import h5py
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np

import os

       
def one_hotting(letter):
    hot_vector = np.zeros(24)
    if letter == 'A':
        hot_vector[0] = 1
    elif letter == 'C':
        hot_vector[1] = 1
    elif letter == 'D':
        hot_vector[2] = 1
    elif letter == 'E':
        hot_vector[3] = 1
    elif letter == 'F':
        hot_vector[4] = 1
    elif letter == 'G':
        hot_vector[5] = 1
    elif letter == 'H':
        hot_vector[6] = 1
    elif letter == 'I':
        hot_vector[7] = 1
    elif letter == 'K':
        hot_vector[8] = 1
    elif letter == 'L':
        hot_vector[9] = 1
    elif letter == 'M':
        hot_vector[10] = 1
    elif letter == 'N':
        hot_vector[11] = 1
    elif letter == 'P':
        hot_vector[12] = 1
    elif letter == 'Q':
        hot_vector[13] = 1
    elif letter == 'R':
        hot_vector[14] = 1
    elif letter == 'S':
        hot_vector[15] = 1
    elif letter == 'T':
        hot_vector[16] = 1
    elif letter == 'V':
        hot_vector[17] = 1
    elif letter == 'W':
        hot_vector[18] = 1
    elif letter == 'Y':
        hot_vector[19] = 1
    elif letter == 'B':
        hot_vector[20] = 1
    elif letter == 'U':
        hot_vector[21] = 1
    elif letter == 'X':
        hot_vector[22] = 1
    elif letter == 'Z':
        hot_vector[23] = 1

    return hot_vector

        
def sequence_to_vector(seq):
    vec = []
    for item in seq:
        if item != None:
            # one-hot the sequences
            onehot = one_hotting(item)
            vec.append(onehot)
    return vec
    

def padd_sequence(sequence, maximum):
    while len(sequence) < maximum:
        sequence.extend(np.zeros((1,24)))


def padd_embedding(embedding, maximum):
    padding = torch.zeros((maximum - embedding.shape[0], embedding.shape[1]))
    padded_embedding = torch.cat((embedding, padding), dim=0)


    return padded_embedding

def get_embedding_per_tok(dirpath, protein_id, layer):
    if dirpath.endswith(".h5"):
        return get_embedding_per_tok_h5(dirpath, protein_id).squeeze(0)
    else:
        embedding = torch.load(os.path.join(dirpath, protein_id + ".pt"))
        return embedding['representations'][layer]

def get_embedding_mean(dirpath, protein_id, layer):
    embedding = torch.load(os.path.join(dirpath, protein_id + ".pt"))
    return embedding['mean_representations'][layer]          

def get_embedding_per_tok_h5(file_path, protein_id):
    with h5py.File(file_path, 'r') as f:
        embedding = torch.from_numpy(f[protein_id][:])
    return embedding


class MyDataset(data.Dataset):
    def __init__(self, filename, layer, max_len=10000, embedding=True, mean=True,
                  embedding_directory="/nfs/scratch/t.reim/embeddings/esm2_t36_3B/"):
        self.df = pd.read_csv(filename)  # Load the data from the CSV file
        if max_len is None:
            self.max = max(max(self.df['sequence_a'].apply(len)), max(self.df['sequence_b'].apply(len)))
        else:
            self.max = max_len
        self.embedding = embedding
        self.mean = mean
        self.embedding_directory = embedding_directory
        self.df = self.df[(self.df['sequence_a'].apply(len) <= self.max) & (self.df['sequence_b'].apply(len) <= self.max)]
        self.df = self.df.reset_index(drop=True)
        self.layer = layer
       
    def __len__(self):
        return len(self.df)
 
    def __max__(self):	
        return self.max

    def __getitem__(self, index):
        data = self.df.iloc[index]
        if self.embedding == True:
            
            if self.mean == False:
                seq1 = get_embedding_per_tok(self.embedding_directory, data['Id1'], self.layer)
                seq2 = get_embedding_per_tok(self.embedding_directory, data['Id2'], self.layer)
                
                seq1 = padd_embedding(seq1, self.max)
                seq2 = padd_embedding(seq2, self.max)
                tensor = torch.stack([seq1, seq2])
            else:
                seq1 = get_embedding_mean(self.embedding_directory, data['Id1'], self.layer)
                seq2 = get_embedding_mean(self.embedding_directory, data['Id2'], self.layer)
                tensor = torch.stack([seq1, seq2])
                tensor = tensor.squeeze(0)
                
        else:    
            seq1 = sequence_to_vector(data['sequence_a'])
            seq2 = sequence_to_vector(data['sequence_b'])
            padd_sequence(seq1, self.max)
            padd_sequence(seq2, self.max)
            seq_array = np.array([seq1, seq2])
            tensor = torch.tensor(seq_array)

        sample = {'name1': data['Id1'], 'name2': data['Id2'], 'tensor': tensor, 'interaction': data['Interact']}
        return sample
    
        
class dataset2d(data.Dataset):
    def __init__(self, filename, layer, max_len=None, embedding_directory="/nfs/scratch/t.reim/embeddings/esm2_t36_3B/"):
        
        self.df = pd.read_csv(filename)

        if max_len is None:
            self.max = max(max(self.df['sequence_a'].apply(len)), max(self.df['sequence_b'].apply(len)))
        else:
            self.max = max_len

        
        self.embedding_directory = embedding_directory
        self.df = self.df[(self.df['sequence_a'].apply(len) <= self.max) & (self.df['sequence_b'].apply(len) <= self.max)]
        self.df = self.df.reset_index(drop=True)
        self.layer = layer
       
        
    def __len__(self):
        return len(self.df)
 
    def __max__(self):	
        return self.max

    def __getitem__(self, index):
        data = self.df.iloc[index]
          
        #seq1 = get_embedding_per_tok(self.embedding_directory, data['Id1'], self.layer)
        #seq2 = get_embedding_per_tok(self.embedding_directory, data['Id2'], self.layer)
        if self.embedding_directory is None:
            sample = {'name1': data['Id1'], 'name2': data['Id2'], 'interaction': data['Interact'], 'sequence_a': data['sequence_a'], 'sequence_b': data['sequence_b']}
        else:
            sample = {'name1': data['Id1'], 'name2': data['Id2'], 'interaction': data['Interact']}
        return sample


# Test area
'''
train_data = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/gold_stand_train_all_seq.csv"
embedding_dir = "/nfs/scratch/t.reim/embeddings/esm2_t33_650/per_tok/"
dataset = MyDataset(train_data, 33, max_len=10000, embedding=True, mean=False, embedding_directory=embedding_dir)
dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=padded_permuted_collate)
for batch in dataloader:
    print(batch['tensor'].shape)
    break
'''