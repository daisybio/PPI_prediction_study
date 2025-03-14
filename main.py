import data.data as d
import models.baseline2d as baseline2d
import models.fc2_20_2_dense as richoux
import models.dscript_like as dscript_like
import models.attention as attention
import torch
import torch.nn as nn
import torch.utils.data as data
import wandb
import argparse
import random
import numpy as np
import traceback


def confmat(y_true, y_pred):
    true_positive = torch.sum((y_true == 1) & (y_pred == 1)).item()
    false_positive = torch.sum((y_true == 0) & (y_pred == 1)).item()
    true_negative = torch.sum((y_true == 0) & (y_pred == 0)).item()
    false_negative = torch.sum((y_true == 1) & (y_pred == 0)).item()
    return true_positive, false_positive, true_negative, false_negative


def metrics(true_positive, false_positive, true_negative, false_negative):
    N = true_positive + false_positive + true_negative + false_negative

    pred_positive = true_positive + false_positive
    real_positive = true_positive + false_negative

    if real_positive == 0:
        accuracy = precision = recall = f1_score = 0
    else:
        accuracy = (true_positive + true_negative) / N
        precision = true_positive / pred_positive if pred_positive > 0 else 0
        recall = true_positive / real_positive if real_positive > 0 else 0
        f1_score = (2 * (precision * recall)) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score

sweep_config = {
    'crossattention':{
        "name": "CrossAttention_sweep_bayesian_val",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "data_name": {"value": "gold_stand"},
            "num_epochs": {"value": 20},
            "subset": {"value": True},
            "subset_size": {"value": 0.5},
            "use_embeddings": {"value": True},
            "mean_embedding": {"value": False},
            "embedding_dim": {"value": 1280},
            "use_wandb": {"value": True},
            "early_stopping": {"value": 8},
            "model_name": {"value": "crossattention"},
            "run_name": {"value": ''},
            "batch_size": {"value": 16},
            "max_seq_len": {"value": 1000},

            "learning_rate": {
                "distribution": "uniform",
                "min": 0.0001,
                "max": 0.01
            },
            "dropout": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.5
            },
            "num_heads": {
                "values": [2, 4, 8, 16]
            },
            "attention_dim": {
                "distribution": "q_uniform",
                "min": 16,
                "max": 1024,
                "q": 16
            },
            "ff_dim":{
                "values": [64, 128, 256, 512, 1024]
            },
            "pooling":{
                "values": ['avg', 'max']
            },
            "kernel_size":{
                "values": [1, 2, 3, 4, 5, 6]
            }
        }
    },
    'TUnA':{
        "name": "TUnA_sweep_bayesian_val",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "data_name": {"value": "gold_stand"},
            "num_epochs": {"value": 20},
            "subset": {"value": True},
            "subset_size": {"value": 0.5},
            "use_embeddings": {"value": True},
            "mean_embedding": {"value": False},
            "embedding_dim": {"value": 1280},
            "use_wandb": {"value": True},
            "early_stopping": {"value": 8},
            "model_name": {"value": "TUnA"},
            "run_name": {"value": ''},
            "batch_size": {"value": 16},
            "max_seq_len": {"value": 1000},
            "learning_rate": {
                "distribution": "uniform",
                "min": 0.0001,
                "max": 0.01
            },
            "dropout": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.5
            },
            "num_heads": {
                "values": [2, 4, 8, 16]
            },
            "attention_dim": {
                "distribution": "q_uniform",
                "min": 16,
                "max": 512,
                "q": 16
            },
            "ff_dim":{
                "values": [64, 128, 256, 512, 1024]
            },
            "rffs":{
                "values": [64, 128, 256, 512, 1024, 2048]
            },
            'cross':{
                "values": [True, False]
            }
        }
    },
    'richoux':{
        "name": "Richoux_sweep_bayesian_val",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "data_name": {"value": "gold_stand"},
            "num_epochs": {"value": 20},
            "subset": {"value": True},
            "subset_size": {"value": 0.5},
            "use_embeddings": {"value": True},
            "mean_embedding": {"value": True},
            "use_wandb": {"value": True},
            "early_stopping": {"value": 8},
            "model_name": {"value": "richoux"},
            "run_name": {"value": ''},
            "batch_size": {"value": 512},
            "max_seq_len": {"value": 10000},
            "embedding_dim": {"values": [1280, 2560, 5120]},
            "learning_rate": {
                "distribution": "uniform",
                "min": 0.0001,
                "max": 0.01
            },
            "ff_dim1":{
                "distribution": "int_uniform",
                "min": 10,
                "max": 1000
            },
            "ff_dim2":{
                "distribution": "int_uniform",
                "min": 10,
                "max": 500
            },
            "ff_dim3":{
                "distribution": "int_uniform",
                "min": 10,
                "max": 500
            },
            "spec_norm":{
                "values": [True, False]
            }
        }
    },
    'dscript_like':{
        "name": "dscript_like_sweep_bayesian_val",
        'method': 'bayes',
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        'parameters': {
            "data_name": {"value": "gold_stand"},
            "num_epochs": {"value": 20},
            "subset": {"value": True},
            "subset_size": {"value": 0.5},
            "use_embeddings": {"value": True},
            "mean_embedding": {"value": False},
            "use_wandb": {"value": True},
            "early_stopping": {"value": 8},
            "model_name": {"value": "dscript_like"},
            "run_name": {"value": ''},
            "batch_size": {"value": 16},
            "max_seq_len": {"value": 1000},
            "embedding_dim": {"values": [1280, 2560, 5120]},
            "learning_rate": {
                "distribution": "uniform",
                "min": 0.0001,
                "max": 0.01
            },
            'd': {
                'distribution': 'int_uniform',
                'min': 20,
                'max': 300
            },
            'w': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 10
            },
            'h': {
                'distribution': 'int_uniform',
                'min': 10,
                'max': 150
            },
            'x0': {
                'distribution': 'uniform',
                'min': -1.0,
                'max': 1.0
            },
            'k': {
                'distribution': 'uniform',
                'min': 1,
                'max': 50
            },
            'pool_size': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 10
            },
            'do_pool': {
                'values': [True, False]
            },
            'do_w': {
                'values': [True, False]
            },
            'theta_init': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 1.0
            },
            'lambda_init': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 1.0
            },
            'gamma_init': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 1.0
            },
        }
    },
    'selfattention':{
        "name": "SelfAttention_sweep_bayesian_val",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "data_name": {"value": "gold_stand"},
            "num_epochs": {"value": 20},
            "subset": {"value": True},
            "subset_size": {"value": 0.5},
            "use_embeddings": {"value": True},
            "mean_embedding": {"value": False},
            "embedding_dim": {"value": 1280},
            "use_wandb": {"value": True},
            "early_stopping": {"value": 8},
            "model_name": {"value": "selfattention"},
            "run_name": {"value": ''},
            "batch_size": {"value": 16},
            "max_seq_len": {"value": 1000},

            "learning_rate": {
                "distribution": "uniform",
                "min": 0.0001,
                "max": 0.01
            },
            "dropout": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.5
            },
            "num_heads": {
                "values": [2, 4, 8, 16]
            },
            "attention_dim": {
                "distribution": "q_uniform",
                "min": 16,
                "max": 1024,
                "q": 16
            },
            "ff_dim":{
                "values": [64, 128, 256, 512, 1024]
            },
            "pooling":{
                "values": ['avg', 'max']
            },
            "kernel_size":{
                "values": [1, 2, 3, 4, 5, 6]
            }
        }
    }
}

best_configs = {
    "crossattention": {
        "seed": 3292779418,
        "save_confpred": True,
        "save_model": True,
        "test": True,
        "data_name": "gold_stand",
        "num_epochs": 25,
        "subset": True,
        "subset_size": 0.5,
        "use_embeddings": True,
        "mean_embedding": False,
        "embedding_dim": 1280,
        "use_wandb": True,
        "early_stopping": 8,
        "model_name": "crossattention",
        "run_name": 'best_crossattention',
        "batch_size": 16,
        "max_seq_len": 1000,
        "learning_rate": 0.0001,
        "dropout": 0.1,
        "num_heads": 8,
        "attention_dim": 64,
        "ff_dim": 256,
        "pooling": 'avg',
        "kernel_size": 3
        },
    "dscript_like": {
    "seed": 3292779418,
    "save_confpred": True,
    "save_model": True,
    "test": True,
    "data_name": "gold_stand",
    "num_epochs": 20,
    "subset": True,
    "subset_size": 0.5,
    "use_embeddings": True,
    "mean_embedding": False,
    "use_wandb": True,
    "early_stopping": 8,
    "model_name": "dscript_like",
    "run_name": "best_dscript_like",
    "batch_size": 16,
    "max_seq_len": 1000,
    "embedding_dim": 1280,
    "learning_rate": 0.008056061488991427,
    "d": 112,
    "w": 4,
    "h": 44,
    "x0": 0.4989645451223388,
    "k": 20.296019995569164,
    "pool_size": 8,
    "do_pool": False,
    "do_w": True,
    "theta_init": 0.9537458250320732,
    "lambda_init": 0.42347718679986945,
    "gamma_init": 0.22024583140255627
    },
    "selfattention": {
        "seed": 3292779418,
        "save_confpred": True,
        "save_model": True,
        "test": True,
        "data_name": "gold_stand",
        "num_epochs": 25,
        "subset": True,
        "subset_size": 0.5,
        "use_embeddings": True,
        "mean_embedding": False,
        "embedding_dim": 1280,
        "use_wandb": True,
        "early_stopping": 8,
        "model_name": "selfattention",
        "run_name": 'best_selfattention',
        "batch_size": 16,
        "max_seq_len": 1000,
        "learning_rate": 0.0001,
        "dropout": 0.1,
        "num_heads": 8,
        "attention_dim": 64,
        "ff_dim": 256,
        "pooling": 'max',
        "kernel_size": 3
    },
    "richoux": {
        "seed": 3292779418,
        "save_confpred": True,
        "save_model": True,
        "test": True,
        "data_name": "gold_stand",
        "num_epochs": 25,
        "subset": True,
        "subset_size": 0.5,
        "use_embeddings": True,
        "mean_embedding": True,
        "use_wandb": True,
        "early_stopping": 8,
        "model_name": "richoux",
        "run_name": 'best_rich',
        "batch_size": 1024,
        "max_seq_len": 10000,
        "embedding_dim": 2560,
        "learning_rate": 0.001,
        "ff_dim1": 20,
        "ff_dim2": 20,
        "ff_dim3": 20,
        "spec_norm": True
    },
    "TUnA": {
        "seed": 3292779418,
        "save_confpred": True,
        "save_model": True,
        "test": True,
        "data_name": "gold_stand",
        "num_epochs": 25,
        "subset": True,
        "subset_size": 0.5,
        "use_embeddings": True,
        "mean_embedding": False,
        "embedding_dim": 1280,
        "use_wandb": True,
        "early_stopping": 8,
        "model_name": "TUnA",
        "run_name": 'best_TUnA',
        "batch_size": 32,
        "max_seq_len": 1000,
        "learning_rate": 0.0005,
        "dropout": 0.15,
        "num_heads": 8,
        "attention_dim": 64,
        "ff_dim": 256,
        "rffs": 1028,
        'cross': False
    },
    "baseline2d": {
        "seed": None,
        "save_confpred": True,
        "save_model": True,
        "test": True,
        "data_name": "gold_stand",
        "num_epochs": 25,
        "subset": True,
        "subset_size": 0.5,
        "use_embeddings": True,
        "mean_embedding": False,
        "embedding_dim": 1280,
        "use_wandb": True,
        "early_stopping": 8,
        "model_name": "baseline2d",
        "run_name": 'best_baseline2d',
        "batch_size": 40,
        "max_seq_len": 1000,
        "learning_rate": 0.001,
        "h3": 64,
        "kernel_size": 2,
        "pooling": 'avg'
    }
}

def train(config=None, sweep=False):
    try:
        seed = None
        if config is not None:
            seed = config.get('seed', None)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        print(f"Random seed: {seed}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    ############################################################__Parameters__#################################################################
        if config is None:
            wandb.init(config=config, project="hypertune") 
            config = wandb.config
            config['seed'] = seed

        print(config)

        #general params
        test = config.get('test', False)
        data_name = config.get('data_name')
        model_name = config.get('model_name')
        learning_rate = config.get('learning_rate')
        num_epochs = config.get('num_epochs')
        bs = config.get('batch_size')
        max = config.get('max_seq_len')
        subset = config.get('subset')
        subset_size = config.get('subset_size')
        use_embeddings = config.get('use_embeddings')
        mean_embedding = config.get('mean_embedding')
        embedding_dim = config.get('embedding_dim')
        use_wandb = config.get('use_wandb')
        early_stopping = config.get('early_stopping', 0)
        save_confpred = config.get('save_confpred', False)
        save_model = config.get('save_model', False)
        #params for attention models
        dropout = config.get('dropout', 0.2) #for tuna #for cross 
        num_heads = config.get('num_heads', 8) #for tuna #for cross 
        run_name = config.get('run_name')
        h3 = config.get('attention_dim', 64) #for tuna #for cross 
        ff_dim = config.get('ff_dim', 256) #for cross 
        pooling = config.get('pooling', 'avg') #for cross 
        kernel_size = config.get('kernel_size', 2) #for cross  
        rffs = config.get('rffs', 1028) #for tuna
        cross = config.get('cross', False) #for tuna
        #params for richoux
        ff_dim1 = config.get('ff_dim1', 20) #for rich
        ff_dim2 = config.get('ff_dim2', 20) #for rich
        ff_dim3 = config.get('ff_dim3', 20) #for rich
        spec_norm = config.get('spec_norm', False) #for rich
        #params for dscript_like
        d_ = config.get('d',128)
        w = config.get('w', 7)
        h = config.get('h', 50)
        x0 = config.get('x0', 0.5)
        k = config.get('k', 20)
        pool_size = config.get('pool_size', 9)
        do_pool = config.get('do_pool', False)
        do_w = config.get('do_w', True)
        theta_init = config.get('theta_init', 1.0)
        lambda_init = config.get('lambda_init', 0.0)
        gamma_init = config.get('gamma_init', 0.0)


        per_tok_models = ["baseline2d", "dscript_like", "selfattention", "crossattention",
                        "ICAN_cross", "AttDscript", "Rich-ATT", "TUnA"]
        mean_models = ["richoux"]
        padded_models = ["richoux", "TUnA"]

        # some cases, could be done better or more elegantly
        if model_name in per_tok_models:
            use_2d_data = True
        elif model_name in mean_models:
            use_2d_data = False
        else:
            raise ValueError(f"The model name {model_name}  does not exist. Please check for spelling errors.")

        if model_name in padded_models:
            use_2d_data = False    

        train_data = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/" + data_name + "/" + data_name + "_train_all_seq.csv"
        
        test_data = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/" + data_name + "/" + data_name + "_test_all_seq.csv"
    
        val_data = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/" + data_name + "/" + data_name + "_val_all_seq.csv"
        
        if embedding_dim == 2560:
            emb_name = 'esm2_t36_3B'
            layer = 36
        elif embedding_dim == 1280: 
            emb_name = 'esm2_t33_650'
            layer = 33
        elif embedding_dim == 5120:
            emb_name = 'esm2_t48_15B'
            layer = 48
        elif embedding_dim == 6165:
            emb_name = 'bepler_berger_2019'
            layer = None  # Assuming there's no specific layer concept for this model
            num_heads = 1
        elif embedding_dim == 320:
            emb_name = "esm2_t6_8M"
            layer = 6

            
        if mean_embedding:
            emb_type = 'mean'
        else:
            emb_type = 'per_tok'


        if use_embeddings:
            embedding_dir = "/nfs/scratch/t.reim/embeddings/" + emb_name + "/" + emb_type + "/"
        else:
            embedding_dir = None

        if embedding_dim == 6165:
            embedding_dir = "/nfs/scratch/jbernett/bepler_berger_embeddings/human_embedding.h5" 

        if run_name is None:
            run_name = f"{model_name}_{emb_name}_{subset_size}"

        if not sweep:
            if use_wandb == True:
                wandb.init(project="bachelor",
                        name=run_name,
                        config=config)    
            
        print("Parameters:")
        for key, value in config.items():
            print(f"{key}: {value}")


        ############################################################__Data__and_Models__#################################################################

        if use_2d_data:
            dataset = d.dataset2d(train_data, layer, max, embedding_dir)
        else:
            dataset = d.MyDataset(train_data, layer, max, use_embeddings, mean_embedding, embedding_dir)

        if subset:
            sampler = data.RandomSampler(dataset, num_samples=int(len(dataset)*subset_size), replacement=True)
            dataloader = data.DataLoader(dataset, batch_size=bs, sampler=sampler)
            print("Using Subset")
            print("Subset Size: ", int(len(dataset)*subset_size))
        else:
            dataloader = data.DataLoader(dataset, batch_size=bs, shuffle=True)


        if use_2d_data:
            vdataset = d.dataset2d(val_data, layer, max, embedding_dir)
        else:
            vdataset = d.MyDataset(val_data, layer, max, use_embeddings, mean_embedding, embedding_dir)

        if subset:
            sampler = data.RandomSampler(vdataset, num_samples=int(len(vdataset)*subset_size), replacement=True)
            vdataloader = data.DataLoader(vdataset, batch_size=bs, sampler=sampler)
            print("Val Subset Size: ", int(len(vdataset)*subset_size))
        else:
            vdataloader = data.DataLoader(vdataset, batch_size=bs, shuffle=True)
        

        if use_2d_data:
            test_dataset = d.dataset2d(test_data, layer, max, embedding_dir)
        else:
            test_dataset = d.MyDataset(test_data, layer, max, use_embeddings, mean_embedding, embedding_dir)

        test_dataloader = data.DataLoader(test_dataset, batch_size=bs, shuffle=True)
        print("Test Data Size: ", int(len(test_dataset)))

        if use_embeddings:
            print("Using Embeddings: ", emb_name, " Mean: ", mean_embedding)
            insize = embedding_dim
        else:     
            print("Not Using Embeddings")  
            print("Max Sequence Length: ", dataset.__max__())
            insize = dataset.__max__()*24

        model_mapping = {
            "dscript_like": dscript_like.DScriptLike(embed_dim=insize, d=d_, w=w, h=h, x0=x0, k=k, pool_size=pool_size, do_pool=do_pool,
                                                      do_w=do_w, theta_init=theta_init, lambda_init=lambda_init, gamma_init=gamma_init),
            "richoux": richoux.FC2_20_2Dense(embed_dim=insize, ff_dim1=ff_dim1, ff_dim2=ff_dim2, ff_dim3=ff_dim3, spec_norm=spec_norm),
            "baseline2d": baseline2d.baseline2d(embed_dim=insize, h3=h3, kernel_size=kernel_size, pooling=pooling),
            "selfattention": attention.SelfAttInteraction(embed_dim=insize, num_heads=num_heads, h3=h3, dropout=dropout,
                                                            ff_dim=ff_dim, pooling=pooling, kernel_size=kernel_size),
            "crossattention": attention.CrossAttInteraction(embed_dim=insize, num_heads=num_heads, h3=h3, dropout=dropout,
                                                            ff_dim=ff_dim, pooling=pooling, kernel_size=kernel_size),
            "ICAN_cross": attention.ICAN_cross(embed_dim=insize, num_heads=num_heads, cnn_drop=dropout, transformer_drop=dropout, hid_dim=h3, ff_dim=ff_dim),
            "AttDscript": attention.AttentionDscript(embed_dim=insize, num_heads=num_heads, dropout=dropout, d=h3, w=w, h=h, x0=x0, k=k,
                                                    pool_size=pool_size, do_pool=do_pool, do_w=do_w, theta_init=theta_init,
                                                    lambda_init=lambda_init, gamma_init=gamma_init),
            "Rich-ATT": attention.AttentionRichoux(embed_dim=insize, num_heads=num_heads, dropout=dropout),
            "TUnA": attention.TUnA(embed_dim=insize, num_heads=num_heads, dropout=dropout, rffs=rffs, cross=cross, hid_dim=h3)
        }

        model = model_mapping.get(model_name)
        if model is None:
            raise ValueError(f"Invalid model_name: {model_name}")

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)  

        if torch.cuda.is_available():
            print("Using CUDA")
            model = model.cuda()
            criterion = criterion.cuda()
            device = torch.device("cuda")
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:    
            print("Using CPU")
            device = torch.device("cpu")    

        ############################################################__Training__#################################################################

        best_val_acc = 0.0
        best_model = None
        es = early_stopping != 0
        patience = early_stopping
        counter = 0
        confident_train_predictions = []
        confident_val_predictions = []
        confident_test_predictions = []
        threshold = 0.9

        for epoch in range(num_epochs):
            epoch_loss= 0.0
            tp, fp, tn, fn = 0, 0, 0, 0
            model.train()
            for batch in dataloader:
                optimizer.zero_grad()

                if use_2d_data:
                    outputs = model.batch_iterate(batch, device, layer, embedding_dir)
                else:
                    tensor = batch['tensor'].to(device)
                    outputs = model(tensor) 
                
                if save_confpred:
                    names1 = batch['name1']
                    names2 = batch['name2']
                    output_values = outputs.detach().cpu().numpy()
                    predicted_labels = torch.round(outputs.float()).detach().cpu().numpy()
                    labels = batch['interaction'].numpy()

                    for value, predicted_label, label, id1, id2 in zip(output_values, predicted_labels, labels, names1, names2):
                        if value > threshold and predicted_label == 1 and label == 1:
                            confident_train_predictions.append((id1, id2))

                labels = batch['interaction']
                labels = labels.unsqueeze(1).float()
                predicted_labels = torch.round(outputs.float())

                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                met = confmat(labels.to(device), predicted_labels)
                tp, fp, tn, fn = tp + met[0], fp + met[1], tn + met[2], fn + met[3]

            acc, prec, rec, f1 = metrics(tp, fp, tn, fn)
            avg_loss = epoch_loss / len(dataloader)
            if use_wandb == True:
                wandb.log({
                    "epoch_loss": avg_loss,
                    "epoch_accuracy": acc,
                    "epoch_precision": prec,
                    "epoch_recall": rec,
                    "epoch_f1_score": f1
                })
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}, Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1}")
            print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

        ############################################################__Validation__#################################################################    

            model.eval()
            val_loss = 0.0
            tp, fp, tn, fn = 0, 0, 0, 0
            with torch.no_grad():
                for val_batch in vdataloader:

                    if use_2d_data:
                        val_outputs = model.batch_iterate(val_batch, device, layer, embedding_dir)
                    else:
                        val_inputs = val_batch['tensor'].to(device)
                        val_outputs = model(val_inputs)

                    if save_confpred:
                        names1 = val_batch['name1']
                        names2 = val_batch['name2']
                        output_values = val_outputs.detach().cpu().numpy()
                        interactions = torch.round(val_outputs).detach().cpu().numpy()
                        labels = batch['interaction'].numpy()

                        for value, interaction, label, name1, name2 in zip(output_values, interactions, labels, names1, names2):
                            if value > threshold and interaction == 1 and label == 1:
                                confident_val_predictions.append((name1, name2))
                            
                    val_labels = val_batch['interaction']
                    val_labels = val_labels.unsqueeze(1).float()
                    predicted_labels = torch.round(val_outputs.float())

                    met = confmat(val_labels.to(device), predicted_labels)
                    val_loss += criterion(val_outputs, val_labels.to(device))
                    tp, fp, tn, fn = tp + met[0], fp + met[1], tn + met[2], fn + met[3]

            avg_loss = val_loss / len(vdataloader)
            acc, prec, rec, f1 = metrics(tp, fp, tn, fn)
            if use_wandb == True:
                wandb.log({
                    "val_loss": avg_loss,
                    "val_accuracy": acc,
                    "val_precision": prec,
                    "val_recall": rec,
                    "val_f1_score": f1
                })  
            print(f"Epoch {epoch+1}/{num_epochs}, Average Val Loss: {avg_loss}, Val Accuracy: {acc}, Val Precision: {prec}, Val Recall: {rec}, Val F1 Score: {f1}")
            print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            if test:
                test_loss = 0.0
                tp, fp, tn, fn = 0, 0, 0, 0
                with torch.no_grad():
                    for test_batch in test_dataloader:

                        if use_2d_data:
                            test_outputs = model.batch_iterate(test_batch, device, layer, embedding_dir)
                        else:
                            test_inputs = test_batch['tensor'].to(device)
                            test_outputs = model(test_inputs)

                        if save_confpred:
                            names1 = test_batch['name1']
                            names2 = test_batch['name2']
                            output_values = test_outputs.detach().cpu().numpy()
                            interactions = torch.round(test_outputs).detach().cpu().numpy()
                            labels = batch['interaction'].numpy()

                            for value, interaction, label, name1, name2 in zip(output_values, interactions, labels, names1, names2):
                                if value > threshold and interaction == 1 and label == 1:
                                    confident_test_predictions.append((name1, name2))
                                
                        test_labels = test_batch['interaction']
                        test_labels = test_labels.unsqueeze(1).float()
                        predicted_labels = torch.round(test_outputs.float())

                        met = confmat(test_labels.to(device), predicted_labels)
                        test_loss += criterion(test_outputs, test_labels.to(device))
                        tp, fp, tn, fn = tp + met[0], fp + met[1], tn + met[2], fn + met[3]

                avg_loss = test_loss / len(test_dataloader)
                acc, prec, rec, f1 = metrics(tp, fp, tn, fn)
                if use_wandb == True:
                    wandb.log({
                        "test_loss": avg_loss,
                        "test_accuracy": acc,
                        "test_precision": prec,
                        "test_recall": rec,
                        "test_f1_score": f1
                    })  
                print(f"Epoch {epoch+1}/{num_epochs}, Average test Loss: {avg_loss}, Test Accuracy: {acc}, Test Precision: {prec}, Test Recall: {rec}, Test F1 Score: {f1}")
                print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
                
            if es:
                if acc > best_val_acc:
                    best_val_acc = acc
                    best_model = model.state_dict()
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping triggered. No improvement in validation accuracy. Best epoch: " + str(epoch+1 - patience))
                        break
            else:
                best_model = model.state_dict()


            if save_confpred:
                with open(f"/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/confident_train_pred_{model_name}.csv", 'w') as f:
                    for name1, name2 in confident_train_predictions:
                        f.write(f'{name1}\t{name2}\n')                

                with open(f"/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/confident_val_pred_{model_name}.csv", 'w') as f:
                    for name1, name2 in confident_val_predictions:
                        f.write(f'{name1}\t{name2}\n')
                with open(f"/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/confident_test_pred_{model_name}.csv", 'w') as f:
                    for name1, name2 in confident_test_predictions:
                        f.write(f'{name1}\t{name2}\n')

        if save_model:
            if test:
                torch.save(best_model, f"/nfs/home/students/t.reim/bachelor/pytorchtest/models/pretrained/{model_name}_{emb_name}_test.pt")
            else: 
                torch.save(best_model, f"/nfs/home/students/t.reim/bachelor/pytorchtest/models/pretrained/{model_name}_{emb_name}_.pt") 
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(e)
        tb = traceback.format_exc()
        print(tb)
        del model
        torch.cuda.empty_cache()
        raise e

def main():
    debug = True
    if debug:
        args = argparse.Namespace(
            data_name="gold_stand_dscript",
            model_name="baseline2d",
            learning_rate=0.001,
            num_epochs=25,
            batch_size=2,
            max_seq_len=1000,
            subset=False,
            subset_size=0.5,
            use_embeddings=True,
            mean_embedding=False,
            embedding_dim=6165,
            use_wandb=False,
            early_stopping=True,
            dropout=0.3,
            num_heads=8,
            run_name=None,
            attention_dim=64
        )
        params = vars(args)
        train(params)   
    else:
        parser = argparse.ArgumentParser(description='PyTorch Training')
        parser.add_argument('-data', '--data_name', type=str, default='gold_stand', help='name of dataset')
        parser.add_argument('-model', '--model_name', type=str, default='dscript_like', help='name of model')
        parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
        parser.add_argument('-epoch', '--num_epochs', type=int, default=25, help='number of epochs')
        parser.add_argument('-batch', '--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('-max', '--max_seq_len', type=int, default=10000, help='max sequence length')
        parser.add_argument('-sub', '--subset', action='store_true', help='use subset')
        parser.add_argument('-subsize', '--subset_size', type=float, default=0.5, help='subset size')
        parser.add_argument('-emb', '--use_embeddings', action='store_true', help='use embeddings')
        parser.add_argument('-mean', '--mean_embedding', action='store_true', help='use mean embedding')
        parser.add_argument('-emb_dim', '--embedding_dim', type=int, default=2560, help='embedding dimension')
        parser.add_argument('-wandb', '--use_wandb', action='store_true', help='use wandb')
        parser.add_argument('-es', '--early_stopping', type=int, default=0, help='early stopping patience')
        parser.add_argument('-dropout', '--dropout', type=float, default=0.2, help='dropout value between 0 and 1')
        parser.add_argument('-heads', '--num_heads', type=int, default=8, help='number of heads for attention')
        parser.add_argument('-run', '--run_name', type=str, default=None, help='name of the run to appear in wandb')
        parser.add_argument('-h3', '--attention_dim', type=int, default=64, help='embedding dimension before attention')
        parser.add_argument('-sweep', '--sweep', action='store_true', help='use wandb sweep/hyperparam_tuning')
        parser.add_argument('-save_confpred', '--save_confpred', action='store_true', help='save confident predictions')
        parser.add_argument('-save_model', '--save_model', action='store_true', help='save model')
        parser.add_argument('-pool', '--pooling', type=str, default='avg', help='pooling method')
        parser.add_argument('-tb', '--test_best', action='store_true', help='test best config')
        parser.add_argument('-test', '--test', action='store_true', help='evaluate model on testset')
        args = parser.parse_args()
        if args.sweep:
            model = args.model_name
            sweep_id = wandb.sweep(sweep_config[model], project="hypertune")
            print("Sweep ID: ", sweep_id)
            #wandb.agent(sweep_id, function=train(sweep=True), count=1)
        if args.test_best:
            model = args.model_name
            train(best_configs[model])    
        else:    
            params = vars(args)
            train(params)            

if __name__ == "__main__":
    main()

