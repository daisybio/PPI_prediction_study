import data.data as d
import models.baseline2d as baseline2d
import models.fc2_20_2_dense as richoux
import models.dscript_like as dscript_like
import models.attention as attention
import os
import torch
import torch.nn as nn
import torch.utils.data as data
import argparse
import numpy as np
import traceback
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import csv  # Add this import at the top of the file

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

def test_model(config):
    try:
        # Set random seed
        seed = config.get('seed', np.random.randint(0, 2**32 - 1))
        print(f"Random seed: {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Extract configuration values
        data_name = config.get('data_name')
        model_name = config.get('model_name')
        batch_size = config.get('batch_size')
        max_seq_len = config.get('max_seq_len')
        use_embeddings = config.get('use_embeddings')
        mean_embedding = config.get('mean_embedding')
        embedding_dim = config.get('embedding_dim')
        pretrained_model_path = config.get('pretrained_model_path')
        outpath = config.get('output_dir', "~/")

        # Additional model-specific parameters
        d_ = config.get('d', 128)
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
        attention_dim = config.get('attention_dim', 64)
        dropout = config.get('dropout', 0.2)
        ff_dim = config.get('ff_dim', 256)
        kernel_size = config.get('kernel_size', 2)
        pooling = config.get('pooling', 'avg')
        num_heads = config.get('num_heads', 8)
        rffs = config.get('rffs', 1028)
        cross = config.get('cross', False)

        # Load test data
        test_data = f"/nfs/home/students/t.reim/bachelor/pytorchtest/data/{data_name}/{data_name}_test_all_seq.csv"
        emb_type = 'mean' if mean_embedding else 'per_tok'
        os.makedirs(outpath, exist_ok=True)

        if use_embeddings:
            if embedding_dim == 2560:
                emb_name, layer = 'esm2_t36_3B', 36
            elif embedding_dim == 1280:
                emb_name, layer = 'esm2_t33_650', 33
            elif embedding_dim == 5120:
                emb_name, layer = 'esm2_t48_15B', 48
            elif embedding_dim == 320:
                emb_name, layer = "esm2_t6_8M", 6
            elif embedding_dim == 6165:
                emb_name, layer = 'bepler_berger_2019', None
                embedding_dir = "/nfs/scratch/jbernett/bepler_berger_embeddings/human_embedding.h5"
            else:
                raise ValueError(f"Unsupported embedding dimension: {embedding_dim}")
            embedding_dir = f"/nfs/scratch/t.reim/embeddings/{emb_name}/{emb_type}/"
        else:
            emb_name, layer, embedding_dir = "onehot", None, None

        # Check if model_name requires 2D dataset
        requires_2d_dataset = model_name in [
            "baseline2d", "dscript_like", "selfattention", "crossattention",
            "ICAN_cross", "AttDscript", "Rich-ATT", "TUnA"
        ]
        if model_name in ["richoux", "TUnA"]:
            requires_2d_dataset = False

        # Load test dataset
        if requires_2d_dataset:
            test_dataset = d.dataset2d(test_data, layer, max_seq_len, embedding_dir)
        else:
            test_dataset = d.MyDataset(test_data, layer, max_seq_len, use_embeddings, mean_embedding, embedding_dir)

        test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        print(f"Test Data Size: {len(test_dataset)}")

        # Determine input size
        if use_embeddings:
            print("Using Embeddings: ", emb_name, " Mean: ", mean_embedding)
            insize = embedding_dim
        else:
            print("Not Using Embeddings")
            if mean_embedding:
                print("Using Flattened One-Hot Encoding")
                print("Max Sequence Length: ", test_dataset.__max__())
                insize = test_dataset.__max__() * 24
            else:
                print("Using 2D One-Hot Encoding")
                print("Max Sequence Length: ", test_dataset.__max__())
                insize = 24

        # Model mapping
        model_mapping = {
            "dscript_like": dscript_like.DScriptLike(embed_dim=insize, d=d_, w=w, h=h, x0=x0, k=k, pool_size=pool_size, do_pool=do_pool, do_w=do_w, theta_init=theta_init, lambda_init=lambda_init, gamma_init=gamma_init),
            "richoux": richoux.FC2_20_2Dense(embed_dim=insize, ff_dim1=20, ff_dim2=20, ff_dim3=20, spec_norm=False),
            "baseline2d": baseline2d.baseline2d(embed_dim=insize, h3=attention_dim, kernel_size=kernel_size, pooling=pooling),
            "selfattention": attention.SelfAttInteraction(embed_dim=insize, num_heads=num_heads, h3=attention_dim, dropout=dropout, ff_dim=ff_dim, pooling=pooling, kernel_size=kernel_size),
            "crossattention": attention.CrossAttInteraction(embed_dim=insize, num_heads=num_heads, h3=attention_dim, dropout=dropout, ff_dim=ff_dim, pooling=pooling, kernel_size=kernel_size),
            "ICAN_cross": attention.ICAN_cross(embed_dim=insize, num_heads=num_heads, cnn_drop=dropout, transformer_drop=dropout, hid_dim=attention_dim, ff_dim=ff_dim),
            "AttDscript": attention.AttentionDscript(embed_dim=insize, num_heads=num_heads, dropout=dropout, d=attention_dim, w=w, h=h, x0=x0, k=k, pool_size=pool_size, do_pool=do_pool, do_w=do_w, theta_init=theta_init, lambda_init=lambda_init, gamma_init=gamma_init),
            "Rich-ATT": attention.AttentionRichoux(embed_dim=insize, num_heads=num_heads, dropout=dropout),
            "TUnA": attention.TUnA(embed_dim=insize, num_heads=num_heads, dropout=dropout, rffs=rffs, cross=cross, hid_dim=attention_dim)
        }

        model = model_mapping.get(model_name)
        if model is None:
            raise ValueError(f"Invalid model_name: {model_name}")

        # Load pretrained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        model.to(device)
        model.eval()

        criterion = nn.BCELoss().to(device)

        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # File to save predictions
        predictions_file = f"{outpath}predictions.csv"
        with open(predictions_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id1", "id2", "pred", "label"])  # Header row

            # Testing
            test_loss = 0.0
            tp, fp, tn, fn = 0, 0, 0, 0
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for idx, test_batch in enumerate(test_dataloader):
                    if requires_2d_dataset:
                        test_outputs = model.batch_iterate(test_batch, device, layer, embedding_dir)
                    else:
                        test_inputs = test_batch['tensor'].to(device)
                        test_outputs = model(test_inputs)

                    test_labels = test_batch['interaction']
                    test_labels = test_labels.unsqueeze(1).float()
                    predicted_labels = torch.round(test_outputs.float())

                    met = confmat(test_labels.to(device), predicted_labels)
                    test_loss += criterion(test_outputs, test_labels.to(device))
                    tp, fp, tn, fn = tp + met[0], fp + met[1], tn + met[2], fn + met[3]

                    all_labels.extend(test_labels.cpu().numpy())
                    all_probs.extend(test_outputs.cpu().numpy())

                    # Write predictions to CSV
                    for i in range(len(test_labels)):
                        id1 = test_batch['id1'][i]  # Assuming 'id1' is in the batch
                        id2 = test_batch['id2'][i]  # Assuming 'id2' is in the batch
                        pred = test_outputs[i].item()
                        label = test_labels[i].item()
                        writer.writerow([id1, id2, pred, label])

            avg_loss = test_loss / len(test_dataloader)
            acc, prec, rec, f1 = metrics(tp, fp, tn, fn)
            print(f"Test Loss: {avg_loss}, Test Accuracy: {acc}, Test Precision: {prec}, Test Recall: {rec}, Test F1 Score: {f1}")
            print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

            # Compute precision-recall curve and AUPR
            precision, recall, _ = precision_recall_curve(all_labels, all_probs)
            aupr = auc(recall, precision)
            print(f"AUPR: {aupr}")

        # Plot precision-recall curve
        plt.figure()
        plt.plot(recall, precision, label=f'AUPR = {aupr:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {model_name}')
        plt.legend(loc='best')

        # Save the plot
        plot_file = f"{outpath}precision_recall_curve.png"
        plt.savefig(plot_file)
        print(f"Precision-Recall Curve saved to {plot_file}")

        # Optionally show the plot
        plt.show()

        print(f"Predictions saved to {predictions_file}")

    except Exception as e:
        print(e)
        tb = traceback.format_exc()
        print(tb)
        raise e

def main():
    debug = True
    if debug:
        args = argparse.Namespace(
            data_name="gold_stand",
            model_name="TUnA",
            batch_size=16,
            max_seq_len=1000,
            use_embeddings=True,
            mean_embedding=False,
            embedding_dim=1280,
            pretrained_model_path="/nfs/home/students/t.reim/bachelor/pytorchtest/models/pretrained/TUnA_esm2_t33_650_test.pt",
            output_dir="/nfs/home/students/t.reim/bachelor/pytorchtest/models/out/",
        )
        params = vars(args)
        test_model(params) 
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('-data', '--data_name', type=str, default='gold_stand', help='name of dataset')
    parser.add_argument('-model', '--model_name', type=str, default='dscript_like', help='name of model')
    parser.add_argument('-batch', '--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('-max', '--max_seq_len', type=int, default=10000, help='max sequence length')
    parser.add_argument('-emb', '--use_embeddings', action='store_true', help='use embeddings')
    parser.add_argument('-mean', '--mean_embedding', action='store_true', help='use mean embedding')
    parser.add_argument('-emb_dim', '--embedding_dim', type=int, default=2560, help='embedding dimension')
    parser.add_argument('-pretrained', '--pretrained_model_path', type=str, required=True, help='path to the pretrained model')
    parser.add_argument('-out', '--output_dir', type=str, help='output directory')
    args = parser.parse_args()
    config = vars(args)
    test_model(config)

if __name__ == "__main__":
    main()