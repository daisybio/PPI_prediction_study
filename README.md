# Deep learning models for unbiased sequence-based PPI prediction plateau at an accuracy of 0.65

![Figure 1 embedding paper](https://github.com/user-attachments/assets/23bed63f-5e0f-467b-8fd6-5262104196c6)

## Data
We used the data leakage-free PPI dataset from [Bernett et al](https://figshare.com/articles/dataset/PPI_prediction_from_sequence_gold_standard_dataset/21591618) available on figshare (DOI: 10.6084/m9.figshare.21591618.v3). 
For the proteins contained in the dataset, we generated ESM-2 per-token and per-protein embeddings [data/extract_esm.py](data/extract_esm.py) for `esm2_t33_650M_UR50D`, `esm2_t36_3B_UR50D`, and `esm2_t48_15B_UR50D`.

## Models
### Per-protein models
As a baseline, we used a Random Forest Classifier - with full embeddings and PCA-reduced embeddings (400 and 40 components). The associated code is in [models/baselineRFC.py](models/baselineRFC.py).

As an advanced model, we re-implemented the fully connected model by Richoux et al. [models/fc2_20_2_dense.py](models/fc2_20_2_dense.py). A version including a Transformer encoder is contained in [models/attention.py#L595](models/attention.py#L595).

### Per-token models
The 2d-baseline model is implemented in [models/baseline2d.py](models/baseline2d.py).

We extended the 2d-baseline by inserting a Transformer encoder with self- [models/attention.py#L90](models/attention.py#L90) or cross-attention [models/attention.py#L15](models/attention.py#L15). 

Further, we re-implemented D-SCRIPT [models/dscript_like.py](models/dscript_like.py). Also, we implemented a version that included a Transformer encoder [models/attention.py#L157](models/attention.py#L157).

We also re-implemented TUnA [models/attention.py#L321](models/attention.py#L321).

## Tests
Hyperparameter tuning was done with wandb. All code necessary for repeating the analyses is found in [main.py](main.py).

## Distance maps

PPIs included in the PDB were identified in [data/get_contact.py](data/get_contact.py). Those were filtered for confident predictions made by the models [data/find_confpreds_with_structure.py](data/find_confpreds_with_structure.py). 

Finally, the distance maps were calculated and their correlations to the predicted distance maps were obtained in [data/get_cmap.py](data/get_cmap.py).

## Visualizations
All other visualizations can be found in [plots/](plots/). 

## Contact

- [Timo Reim](https://github.com/BlackCetus) T.Reim@campus.lmu.de (Developer)
- [Judith Bernett](https://github.com/JudithBernett) judith.bernett@tum.de 
- [Markus List](https://github.com/mlist) markus.list@tum.de
