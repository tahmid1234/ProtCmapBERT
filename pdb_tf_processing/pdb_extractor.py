import numpy as np
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
def calc_distance_matrix(atoms):
    """Calculate the distance matrix for a list of atoms."""
    num_atoms = len(atoms)
    distance_matrix = np.zeros((num_atoms, num_atoms))

    for i, atom1 in enumerate(atoms):
        for j, atom2 in enumerate(atoms):
            distance_matrix[i, j] = atom1 - atom2  # This uses the overridden '-' operator to calculate distance

    return distance_matrix



def load_predicted_clean_PDB(pdb_filename):
    # Create a parser and parse the structure

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_filename)

    # Get the first model and first chain
    model = structure[0]
    _chain = list(model.get_chains())[0]
   
    chain_id = pdb_filename.split('-')[-1].split('.')[0]
    
    if chain_id in model:
        chain = model[chain_id]
    else:
        raise ValueError(f"Chain {chain_id} not found in {pdb_file}")


    seq = ""
    for residue in chain:
        seq= seq+d3to1[residue.resname]

    # Extract C-alpha atoms
    ca_atoms = [atom for atom in chain.get_atoms() if atom.get_id() == 'CA' and is_aa(atom.get_parent())]

    # Calculate the distance matrix
    distance_matrix = calc_distance_matrix(ca_atoms)
    
    #print dis\tance matrix
    
    
    return distance_matrix,seq