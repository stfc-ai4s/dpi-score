
from itertools import combinations

import os
import json
import gemmi
import numpy as np
import numpy.typing as npt
from copy import deepcopy


ELEMENTS = ["C", "N", "O", "S", "P", "H"]

def __vector_ca_atoms(
    vmp: gemmi.VectorMarkPtr, st: gemmi.Structure
) -> gemmi.VectorMarkPtr:
    """
    Given a VectorMarkPtr (array of Marks, which contain atoms) return
    the VectorMarkPtr of CA marks.

    Inputs:
        vmp (gemmi.VectorMarkPtr): vector of marks generated from populated
            search

    Returns:
        gemmi.VectorMarkPtr of CA atoms
    """
    ret = gemmi.VectorMarkPtr()
    for mark in vmp:
        cra = mark.to_cra(st[0])
        if cra.atom.name == "CA":
            ret.append(mark)
    return ret


def residue_to_string(res: gemmi.Residue, author: bool = False) -> str:
    """
    Given a residue, return the amino acid code and chain position.
    Example output: 'SER248'
    Defaults to mmCIF renumbered notation.  Flag changes to author augmented
    notation.

    Input:
        res (gemmi.Residue): target residue object
        author (bool): denotes if author denoted sequences should be used

    Returns:
        res_str (str): formatted string
    """
    if author:
        if res.label_seq is not None:
            return res.name + str(res.label_seq)
        else:
            return res.name + str(res.seqid)
    else:
        return res.name + str(res.seqid)


def get_atom_feats(atom, chainID, resID, resNAME):

    onehot_atom = np.zeros(len(ELEMENTS), dtype=int)
    onehot_atom[ELEMENTS.index(atom.element.name)] = 1
    
    atom_features = {
        'x': atom.pos[0], 
        'y': atom.pos[1], 
        'z': atom.pos[2],
        'C': onehot_atom[0],
        'N': onehot_atom[1],
        'O': onehot_atom[2],
        'S': onehot_atom[3],
        'P': onehot_atom[4],
        'H': onehot_atom[5],
        'charge': atom.charge,
        'occ': atom.occ,
        'biso': atom.b_iso,
        'CA': 1 if atom.name=='CA' else 0,
        'chainID': chainID+1,
        'resID': resID,
    }
    return atom_features

    
def _get_residue_atoms(
    chain: gemmi.Chain, 
    chain_id: int, 
    res_list: list[str],
) -> npt.NDArray[any]:
    """
    Given a chain and a list of target residues, return a list of the atoms
    involved in the interface.  Ugly, as calculates both with sequence ID
    and label ID and considers both.  If both are valid targets prioritises
    label.

    Inputs:
        chain (gemmi.Chain): target chain
        res_list (list[str]): residues in chain, can be disordered

    Returns:
        NDArray[any]
    """
    
    atoms_label = []
    atoms_seq = []
    # insertion code fix
    res_list_label = [
        int(res[3:]) if not res[-1].isalpha() else int(res[3:-1]) for res in res_list
    ]
    res_list_label.sort()
    res_list_seq = deepcopy(res_list_label)

    for res in chain:
        if len(res_list_label) > 0 and len(res_list_seq) > 0:
            res_id_label = res_list_label[0]
            res_id_seq = res_list_seq[0]
        else:
            break

        
        # Default to label_seq, but if not exists use inbuilt numbering
        if res.label_seq == res_id_label:
            for atom in res:
                if atom.element.name in ELEMENTS:
                    atom_label = get_atom_feats(atom, chain_id, res_id_label, res.name)
                    atoms_label.append(atom_label)
                
            res_list_label.pop(0)

        if res.seqid.num == res_id_seq:
            for atom in res:
                if atom.element.name in ELEMENTS:
                    atom_label = get_atom_feats(atom, chain_id, res_id_label, res.name)
                    atoms_seq.append(atom_label)
                    # print(atom_label)
                    
            res_list_seq.pop(0)

    
    if len(res_list_label) <= len(res_list_seq):
        return atoms_label
    else:
        return atoms_seq

def get_residue_atoms(
    chain: gemmi.Chain, 
    chain_id: int, 
    res_list: list[str],
) -> npt.NDArray[any]:
    """
    Clean version of older get_residue_atoms.
    Given a chain and a list of target residues, return a list of the atoms
    involved in the interface.  Ugly, as calculates both with sequence ID
    and label ID and considers both.  If both are valid targets prioritises
    label.

    Inputs:
        chain (gemmi.Chain): target chain
        res_list (list[str]): residues in chain, can be disordered

    Returns:
        NDArray[any]
    """
    
    atoms_label = []
    atoms_seq = []
    res_list_label = [
        int(res[3:]) if not res[-1].isalpha() else int(res[3:-1]) for res in res_list
    ]
    
    for res in chain:
        res_id = None
        if res.label_seq in res_list_label:
            res_id = res.label_seq
        elif res.seqid.num in res_list_label:
            res_id = res.seqid.num

        # print(res.name, res.label_seq, res.seqid.num, res_label)
        if res_id is not None:
            for atom in res:
                if atom.element.name in ELEMENTS:
                    atoms_label.append(get_atom_feats(atom, chain_id, res_id, res.name))
                
            res_list_label.remove(res_id)
    
    return atoms_label
        
def get_interface_residues(
    st: gemmi.Structure,
    chain0_id: int,
    chain1_id: int,
    dist_cut: float,
    nres_cut: int,
    dict_res_contacts: dict,
    dict_intf: dict,
    author: bool = True,
) -> tuple[dict, dict]:
    """
    Given two chains of the same structure, finds the interface residues.
    Uses CA atoms as a search feature.

    Input:
        st (gemmi.Structure): structure that the chains are in
        chain0_id (int): id of the chain within the structure that populates
                         the neighbour search
        chain1_id (int): id of the other chain within the structure that is
                         compared atom-by-atom
        dist_cut (float): angstrom distance cutoff to detect interfaces
        nres_cut (int): number of residues that are minimum in an interface
        dict_res_contacts (dict): dictionary of residue contacts
        dict_intf (dict): dictionary of interfaces
        author (bool): denotes if author denoted sequences should be used

    Returns:
        tuple[dict_res_contacts, dict_intf] where
            dict_res_contacts (dict): same dict with new residues added
            dict_intf (dict): same dict with new interfaces added
    """
    res_list_0 = []
    res_list_1 = []
    intf_ch = st[0][chain0_id].name + "_" + st[0][chain1_id].name

    # Perform the neighboUr search
    ns = gemmi.NeighborSearch(st[0], st.cell, dist_cut)
    ns.add_chain(st[0][chain0_id], include_h=False)

    for res in st[0][chain1_id]:
        for atom in res:
            if atom.name == "CA":
                marks = ns.find_neighbors(atom, min_dist=0.1, max_dist=dist_cut)
                mark_vector = __vector_ca_atoms(marks, st)
                if mark_vector:
                    res_str_1 = residue_to_string(res, author)
                    res_list_1.append(res_str_1)
                    t = []
                    for mark in mark_vector:
                        cra = mark.to_cra(st[0])
                        res_list_0.append(residue_to_string(cra.residue, author))
                        t.append(residue_to_string(cra.residue, author))
                    if intf_ch in list(dict_res_contacts.keys()):
                        dict_res_contacts[intf_ch].update({res_str_1: t})
                    else:
                        dict_res_contacts[intf_ch] = {res_str_1: t}

    # Number of interface residues from each chain is >=10
    if len(set(res_list_0)) >= nres_cut and len(set(res_list_1)) >= nres_cut:
        dict_intf[intf_ch] = [list(set(res_list_0)), list(set(res_list_1))]

    return dict_res_contacts, dict_intf


def get_interface_dict(
    file: str,
    len_chains: int = 30,
    dist_cut: float = 7.0,
    nres_cut: int = 10,
    num_chains: int = 30,
    author: bool = True,
) -> None:
    """
    Given a filepath, calculate and dump the interface dict next to it.

    Inputs:
        file (str): path of target .cif | .pdb | .ent +- *.gz
        len_chains (int): minimum length of chains for neighboUr search
        dist_cut (float): cutoff distance for neighboUr search
        nres_cut (int): minimum number of residues required for an interface
        num_chains (int): maximum number of chains within file
        author (bool): denotes if author denoted sequences should be used

    Returns:
        None
    """
    # print(f"{len_chains=}, {dist_cut=}, {nres_cut=}, {num_chains=}")

    st = gemmi.read_structure(file)

    if file.endswith((".pdb", ".ent")):
        st.setup_entities()
        st.assign_label_seq_id()
    st.remove_ligands_and_waters()

    # Ignore structure with chains>max_chains
    if len(st[0]) > num_chains:
        return -1

    chain_comb_list = list(combinations(range(len(st[0])), 2))
    dict_res_contacts = {}
    dict_intf = {}

    for chain_comb in chain_comb_list:
        ca_0 = st[0][chain_comb[0]]
        ca_1 = st[0][chain_comb[1]]
        # verify length of chains
        if len(ca_0) < len_chains or len(ca_1) < len_chains:
            # insufficient chain length in file
            continue

        poly_0, poly_1 = ca_0.get_polymer(), ca_1.get_polymer()
        if not poly_0 or not poly_1:
            continue
        if (not poly_0.check_polymer_type() == gemmi.PolymerType.PeptideL) or (
            not poly_1.check_polymer_type() == gemmi.PolymerType.PeptideL
        ):
            continue

        dict_res_contacts, dict_intf = get_interface_residues(
            st=st,
            chain0_id=chain_comb[0],
            chain1_id=chain_comb[1],
            dist_cut=dist_cut,
            nres_cut=nres_cut,
            author=author,
            dict_res_contacts=dict_res_contacts,
            dict_intf=dict_intf,
        )

    # # Cluttered the original directory
    # intf_outfle_name = os.path.splitext(file)[0] + "_interface_residues_dict.json"
    # if dict_intf:
    #     with open(intf_outfle_name, "w") as outfle:
    #         json.dump(dict_intf, outfle)
            
    return dict_res_contacts, dict_intf