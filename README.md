# GIF-PLA
## Structure-aware heterogeneous graph neural networks for protein-ligand binding affinity prediction

This repository contains the code for the work on protein-ligand binding affinty with heterogeneous graph.

## Dependencies
- python 3.8
- pytorch = 1.10.0
- numpy = 1.20.0
- RDkit = 2019.09.3
- pandas
- networkx = 2.8.4
- scipy = 1.7.3
- seaborn = 0.12.1
- torch-geometric = 2.2.0
- tqdm = 4.62.3


# Repository Structure

## `data` Folder
The `data` folder contains the datasets utilized in our study. Each file includes the PDB IDs and binding affinity values of the complexes. You can download the protein's PDB files and the compound's mol2 files corresponding to these PDB IDs from the [PDBbind database](http://www.pdbbind.org.cn). Additionally, protein sequences can be obtained from the [UniProt website](https://www.uniprot.org), and compound SMILES strings can be downloaded from the [PubChem database](https://pubchem.ncbi.nlm.nih.gov).

## `sample_data` Folder
The `sample_data` folder contains ten example samples that you can use to familiarize yourself with the data format and test the functionality of our model. These samples are a subset of the larger dataset and are provided to offer a quick and easy way to see GIF-PLA in action.

## Step-by-step running:

### 1. Process raw data
- Firstly, run preprocess_complex.py using `python preprocess_complex.py`.
Through this step, you will obtain the protein-ligand complexes saved as rdkit files.
 
- Secondly, run graph_construct.py using `python graph_construct.py`.
This step will transform the complexes saved in the first step into a graphical representation.

### 2. Running GIF-PLA

## Making Predictions with Our Trained Model
To make predictions using our trained model:
- Start by preparing your data as described in "Step 1: Process Raw Data."
- We have provided a sample dataset for your convenience. To use this dataset, run the following command:
```bash
python predict.py
```
This command will employ the sample data and our trained model.pt to perform predictions, outputting the results directly.


## Training a New Model with Your Data
To train a new model using your dataset:
- Confirm that you have all the required inputs ready.
- Initiate the training process with the following command:
```bash
python train.py
```


