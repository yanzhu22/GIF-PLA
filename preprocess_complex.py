# %%
import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import logging

# %%

def generate_pocket(data_dir, distance=6):
    complex_id = os.listdir(data_dir)
    for cid in complex_id:
        print("cid: ", cid)
        complex_dir = os.path.join(data_dir, cid)
        lig_native_path = os.path.join(complex_dir, f"{cid}_ligand.mol2")
        protein_path= os.path.join(complex_dir, f"{cid}_protein.pdb")

        if os.path.exists(os.path.join(complex_dir, f'Pocket_{distance}A.pdb')):
            continue

        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_native_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {cid}_ligand around {distance}')
        pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
        pymol.cmd.delete('all')

def generate_complex(data_dir, data_df, distance=4, input_ligand_format='mol2'):
    j=0
    #pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        cid, pKa = row['pdbid'], float(row['num'])
        complex_dir = os.path.join(data_dir, cid)
        pocket_path = os.path.join(data_dir, cid, f'Pocket_{distance}A.pdb')
        if input_ligand_format != 'pdb':
            ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.{input_ligand_format}')
            ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
            os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
        else:
            ligand_path = os.path.join(data_dir, cid, f'{cid}_ligand.pdb')

        save_path = os.path.join(complex_dir, f"{cid}-4.rdkit")
        ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)

        if ligand == None:
            print(f"Unable to process ligand of {cid}")
            j = j+1
            continue

        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)

        if pocket == None:
            print(f"Unable to process protein of {cid}")
            j= j+1
            continue
        complex = (ligand, pocket)
        with open(save_path, 'wb') as f:
            pickle.dump(complex, f)

if __name__ == '__main__':
    distance = 6
    input_ligand_format = 'mol2'
    data_root = './PDBbindv2016/'
    data_dir = os.path.join(data_root, 'pdbbind_files')
    data_df = pd.read_csv(os.path.join(data_root, './train_add_seq_smiles.csv'))

    ## generate pocket within 6 Ångström around ligand 
    generate_pocket(data_dir=data_dir, distance=distance)
    generate_complex(data_dir, data_df, distance=distance, input_ligand_format=input_ligand_format)
    



# %%
