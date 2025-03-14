import torch
import torch.utils.data as data
import h5py
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np

import os


class DataFileFormatException(Exception):
    '''Exception class to handle errors in files'''

    def __init__(self, value):
        self.value = value
        self.msg = "File format is not compliant."

    def __str__(self):
        return repr(self.value)

def padded_permuted_collate(batch: dict) -> dict:
    """
    Takes list of tuples with embeddings of variable sizes and pads them with zeros
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of tensor of embeddings with [batchsize, length_of_longest_sequence, embeddings_dim]
    and tensor of labels [batchsize, labels_dim] and metadate collated according to default collate

    """
    embeddings = batch['tensor']
    embeddings = pad_sequence(embeddings, batch_first=True)
    batch['tensor'] = embeddings.permute(0, 2, 1)
    return batch

def max_sequence_size(filename):
    max = 0
    if (has_proper_structure(filename)):
        data_file = open(filename, 'r')
        data_content = data_file.read()
        data_content = data_content.strip()
        lines = data_content.splitlines()
        for line in lines:
            if ',' in line:
                sep = ','
            else:
                sep = ' '
            content = line.split(sep)
            if len(content[4]) > max:
                max = len(content[4])
            if len(content[5]) > max:
                max = len(content[5])
        return max
    
        
def has_proper_structure(filename):
    '''Checks that the file structure corresponds to
    a multiline file with lines structured like this :
    name1, name2, seq1, seq2, interaction
    Names are not checked 
    seq1 and seq2 should be strings
    interaction just one character with either 0 or 1 
    '''
    data_file = open(filename, 'r')
    data_content = data_file.read()
    data_content = data_content.strip()
    lines = data_content.splitlines()[1:]
    temp = 0
    for line in lines:
        if ',' in line:
            sep = ','
        else:
            sep = ' '
        temp = temp + 1 
        content = line.split(sep)
        if line != "\n":
            for i in range(1, len(content)):
                if len(content) != 6:
                    print("Problem line :", temp)
                    print(len(content))
                    print(content)
                    raise DataFileFormatException('Length of line is wrong')
                elif content[4].isalpha() != True:
                    raise DataFileFormatException('First sequence is not in alpha format'+line)
                elif content[5].isalpha() != True:
                    raise DataFileFormatException('Second sequence is not in alpha format')
    return True

# Functions with '_better' at the end aimed to handle one-hotting by passing vectors of integers to GPU cards
# and let these card to convert such vectors into one-hot vectors (here, vectors of vectors of 20 Booleans).

def one_hotting_better(letter):
    '''
    Same as below but with dense vectors (better for RAM)
    '''
    hot = -1
    if letter == 'A':
        hot = 19
    elif letter == 'C':
        hot = 18
    elif letter == 'D':
        hot = 17
    elif letter == 'E':
        hot = 16
    elif letter == 'F':
        hot = 15
    elif letter == 'G':
        hot = 14
    elif letter == 'H':
        hot = 13
    elif letter == 'I':
        hot = 12
    elif letter == 'K':
        hot = 11
    elif letter == 'L':
        hot = 10
    elif letter == 'M':
        hot = 9
    elif letter == 'N':
        hot = 8
    elif letter == 'O':
        hot = 7
    elif letter == 'P':
        hot = 6
    elif letter == 'Q':
        hot = 5
    elif letter == 'R':
        hot = 4
    elif letter == 'T':
        hot = 3
    elif letter == 'V':
        hot = 2
    elif letter == 'W':
        hot = 1
    elif letter == 'Y':
        hot = 0

    return hot
    

def one_hotting(letter):
    '''one-hots values for aa
    The dataset has 20 aas, encoded with letters, and 6 letters are
    not used, but not the 6 lasts (or firsts). Instead, the letters
    B, J, S, U, X, W are not being used. 
    No switch case in python, hence the ugly if elifs 
    TODO: find a nicer way to do it ?
    '''
    hot_vector = np.zeros(24)
    if letter == 'A':
        hot_vector[19] = 1
    elif letter == 'C':
        hot_vector[18] = 1
    elif letter == 'D':
        hot_vector[17] = 1
    elif letter == 'E':
        hot_vector[16] = 1
    elif letter == 'F':
        hot_vector[15] = 1
    elif letter == 'G':
        hot_vector[14] = 1
    elif letter == 'H':
        hot_vector[13] = 1
    elif letter == 'I':
        hot_vector[12] = 1
    elif letter == 'K':
        hot_vector[11] = 1
    elif letter == 'L':
        hot_vector[10] = 1
    elif letter == 'M':
        hot_vector[9] = 1
    elif letter == 'N':
        hot_vector[8] = 1
    elif letter == 'O':
        hot_vector[7] = 1
    elif letter == 'P':
        hot_vector[6] = 1
    elif letter == 'Q':
        hot_vector[5] = 1
    elif letter == 'R':
        hot_vector[4] = 1
    elif letter == 'T':
        hot_vector[3] = 1
    elif letter == 'V':
        hot_vector[2] = 1
    elif letter == 'W':
        hot_vector[1] = 1
    elif letter == 'Y':
        hot_vector[0] = 1      
    return hot_vector

        
def sequence_to_vector(seq):
    '''
    Turns a sequence of letters into a sequence of numbers
    to be used in a vector afterwards
    '''
    vec = []
    for item in seq:
        if item != None:
            # one-hot the sequences
            onehot = one_hotting(item)
            vec.append(onehot)
    return vec



def sequence_to_vector_better(seq):
    '''
    Turns a sequence of letters into a sequence of numbers
    to be used in a vector afterwards
    '''
    vec = []
    for item in seq:
        if item != None:
            #onehot the sequences
            onehot = one_hotting_better(item)
            vec.append(onehot)                        
    return vec
    

def padd_sequence(sequence, maximum):
    '''
    padds the data with 0s to have them all have the same size
    according to the maximum parameter
    '''
    while len(sequence) < maximum:
        sequence.extend(np.zeros((1,24)))

def padd_sequence_better(sequence, maximum):
    '''
    Padds the data with 0s to have them all be the same size
    following the better functions
    '''
    while len(sequence) < maximum:
        sequence.extend(0)

def padd_embedding(embedding, maximum):
    '''
    Padds the data with 0s to have them all be the same size
    following the better functions
    '''
    padding = torch.zeros((maximum - embedding.shape[0], embedding.shape[1]))
    padded_embedding = torch.cat((embedding, padding), dim=0)


    return padded_embedding


def tensorize(sequence1, sequence2):
    '''Make a tensor with two sequences of numbers 
    returns a FloatTensor of dimension 3xMax_size
    note: removing the cuda tensor from here so the dataset loads
    the tensors in the cpu and only when training is it loaded to gpu
    '''
    return torch.FloatTensor([sequence1, sequence2]).unsqueeze(0)


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