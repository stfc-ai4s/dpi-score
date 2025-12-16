"""
    @author: Nicholas Whyatt UKRI-STFC on behalf of CCP-EM
"""

from re import match


import os
import csv
import gzip
import gemmi
import shutil
import numpy as np
import numpy.typing as npt
from typing import Tuple, Union, List


def gather_files(
    target_dir: str, dest_dir: str, file_end: str, exclude=np.array([])
) -> None:
    """
    os.walks through a target directory completely, gathers all files
    of the correct file ending and writes them to dest_dir.
    Highly inefficient in large directories due to os.walk.

    Inputs:
        target_dir (str): target directory (to copy from)
        dest_dir (str): destination directory (to copy to)
        file_end (str): file ending to copy
        exclude (ndarray): files not to copy, typically duplicates

    Returns:
        None
    """
    count = 0

    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(file_end) and not np.isin(file, exclude):
                shutil.copy(src=os.path.join(root, file), dst=dest_dir)
                count += 1

    print(f"Moved {count} files from {target_dir} to {dest_dir}")


def __reset_dir(path: str) -> None:
    """
    Obliterates a local directory with style and panache, then makes a
    new directory of the same name in the same place.

    Inputs:
        path (str): Directory (and all files, dirs inside) you wish to burn

    Returns:
        None, but the sweet scent of file-related ozone
    """
    num_files = len(
        [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    )
    print(f"Resetting directory {path} with {num_files} files")
    shutil.rmtree(path=path)
    os.makedirs(name=path)
    print("Directory remade.")


def create_dir(path: str, clean=False) -> None:
    """
    Cakes a new directory. If clean flag, remove the directory and create new directory of the same name.

    Inputs:
        path (str): Directory (and all files, dirs inside) you wish to burn

    Returns:
        path
    """

    if clean:
        print(f"Removing directory {path}")
        shutil.rmtree(path=path, ignore_errors=True)

    if not os.path.exists(path):
        os.makedirs(name=path)

    return path
    


def write_single_chain_pdb(
    st: gemmi.Structure, chain_name: str, filename: str, out_dir: str, ow: bool = False
) -> str:
    """
    Given a target structure, deep copy the struct, and eliminate all chains
    other than chain name.  Write a minimal PDB file and return the filepath.

    Inputs:
        st (gemmi.Structure): structure to deep copy
        chain_name (str): chain name target
        filename (str): filename path
        out_dir (str): where to write the file
        ow (bool): simple "if file exists" check to prevent overwrites

    Returns:
        string path of target
    """
    chain_file = os.path.join(
        out_dir, os.path.basename(filename)[0:-4] + "_" + chain_name + ".pdb"
    )
    if os.path.isfile(chain_file) and not ow:
        return chain_file
    st_copy = st.clone()
    for chain in st_copy[0]:
        if chain.name != chain_name:
            del chain[:]
    st_copy.write_minimal_pdb(chain_file)
    return chain_file


def struct_handler(file: str) -> gemmi.Structure:
    """
    Abstraction to load a structure safely regardless of filetype.
    Only necessary to perform Gemmi operations on some .PDB files.

    Inputs:
        structure file

    Returns:
        safely opened gemmi.Structure
    """
    st = gemmi.read_structure(file)
    if file.endswith((".pdb", ".ent")):
        st.setup_entities()
        st.assign_label_seq_id()
    st.remove_ligands_and_waters()
    return st


def handle_cif_with_tmp(
    file: str, tmp_dir: str, res_check: bool = True
) -> tuple:
    """
    Handler function to take a structure file of indeterminate type and
    either pass, or attempt to write a minimal PDB.  Extant string for
    success.

    Inputs:
        file (str): full path to target
        tmp_dir (str): temp directory

    Returns:
        tuple success value, typically resfile address
    """
    file_path, filename = os.path.split(file)
    filename = "".join(filename.split(".")[0:-1])

    resfile = os.path.join(file_path, filename + "_interface_residues_dict.json")

    if not os.path.isfile(resfile) and res_check:
        # no interfaces
        raise FileNotFoundError(f"Resfile {resfile} isn't there.")
    if file.endswith(".cif"):
        st = gemmi.read_structure(file)
        file = os.path.join(tmp_dir, filename + ".pdb")
        try:
            st.write_minimal_pdb(file)
        except RuntimeError as e:
            raise RuntimeError(
                f"{file} could not be parsed" + f"into minature PDB: {e}"
            )

    return file, filename, resfile


def cleaned(func: callable, tmp_dir: str, wipe: bool = False) -> any:
    """
    Wrapper function to place a callable in that needs to make writes to
    a temporary directory.  Cleans the directory thereafter.

    Example usage:
        Y = cleaned(lambda: target(param1=x, param2=y), tmp_dir=...)

    Inputs:
        func (callable): lambda wrapped
        tmp_dir (str): your write directory for temporary files
        wipe (bool): prewipes directory if need be

    Returns:
        func output
    """
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
    else:
        if wipe:
            __reset_dir(tmp_dir)

    output = func()
    __reset_dir(tmp_dir)
    return output


def __valid_file(file: str, exclude: str = "") -> bool:
    """
    Internal method to provide an easier bool call from re.match, and to
    simply return True should an exclude param not be given (empty string).
    Used mainly to simplify my insatiable desire for list comprehensions.

    Inputs:
        file (str): target file
        exclude (str): string for use within re.match - match returns false

    Returns:
        bool
    """
    if not exclude:
        return True

    if match(pattern=exclude, string=file):
        return False
    else:
        return True


def get_files_in_dir(
    root: str, 
    file_end: Union[str, Tuple[str]],
    exclude=""
) -> list:
    """
    Returns an array of all the files in the root dir with the given file
    ending.

    Inputs:
        root (str): root directory
        file_end(str) | tuple[str, ...]: file ending, i.e. 'h5' or tuple
            of file endings
        exclude (str): optional argument to perform regex elimination

    Returns:
        List of the various files
    """
    files = [
            f
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
            and f.endswith(file_end)
            and __valid_file(f, exclude)
    ]

    return files


def get_structure_file_dict(
    roots: Tuple[str, ...], 
    file_lists: Tuple[str, ...]
) -> dict:
    """
    Gets a dictionary of the assembly file paths keyed against the PDB code.
    i.e., "{'1AFW': ['data/raw/xray/1AFW.cif', 'data/raw/dock_pos/1AFW.1.cif']}

    Inputs:
        roots (tuple[str, ...]): paths of the individual files
        file_lists: (tuple[str, ...]): list of files in path directory

    Returns:
        dict in above format
    """
    if len(roots) != len(file_lists):
        raise IndexError(
            f"Inputs are not the same length {len(roots)}, {len(file_lists)}"
        )

    mappings = zip(roots, file_lists)
    keys = np.unique([f[0:4].upper() for f in np.concatenate(file_lists)])
    st_dict = {k: [] for k in keys}
    for root, file_list in mappings:
        for file in file_list:
            st_dict[file[0:4].upper()].append(os.path.join(root, file))
    return st_dict


def get_structure_file_dict_casp(
    preds_path: str = r"data/casp/predictions/oligo",
    targets_path: str = r"data/casp/targets",
) -> dict:
    """
    Gets a dictionary of the assembly file paths keyed against the target CASP
    code.  Hard-to-adapt function based on a specific layout of the casp directory
    which has server_predicted, targets and results as seperate dirs, and server_preds
    subdirectoried into individual targets.

    Inputs:
        targets_path (str): the path to the native targets (flat dir of PDBs)
        preds_path (str): server predicteds, with subdirs SOD THE SERVER PREDS

    Returns:
        dict in above format
    """
    targets_list = get_files_in_dir(root=targets_path, file_end=".pdb")
    print(f"{len(targets_list)} targets structures in {targets_path}")
    
    # no longer has all the right answers since we're working with iAlign now
    keys = np.unique([f[0:-4] for f in targets_list])
    st_dict = {k: [] for k in keys}

    for dir in os.listdir(preds_path):
        # here we add the targets
        # we can also remove all targets without a native file
        native = os.path.join(targets_path, dir + ".pdb")
        # print(dir)
        if os.path.isfile(native):
            # print(native)
            st_dict[dir].append(native)
        else:
            continue

        rename_list = os.listdir(os.path.join(preds_path, dir))
        for file in rename_list:
            if not file.endswith(".pdb"):
                os.rename(
                    os.path.join(preds_path, dir, file),
                    os.path.join(preds_path, dir, file + ".pdb"),
                )
        files = get_files_in_dir(root=os.path.join(preds_path, dir), file_end=".pdb")

        for file in files:
            try:
                st_dict[dir].append(os.path.join(preds_path, dir, file))
            except KeyError:
                continue

    return st_dict


def unzip_files_in_dir(root: str) -> None:
    """
    Unzips all .gz files within a directory.

    Inputs:
        root (str): root directory

    Returns:
        None
    """
    print("Unzipping files in directory", root)
    target = get_files_in_dir(root=root, file_end=".gz")
    if len(target) == 0:
        print("No files to unzip in", root)
        pass
    for archive in target:
        file = archive[:-3]
        with gzip.open(os.path.join(root, archive), "rb") as file_in:
            with open(os.path.join(root, file), "wb") as file_out:
                shutil.copyfileobj(file_in, file_out)
        os.remove(os.path.join(root, archive))

def get_h5_stats(h5_dir: str) -> dict | None:
    """
    Hunts through h5 stats within the current directory to gather rundata
    stats.  Returns a dict to be printed or logged, accordingly.

    Inputs:
        h5_dir (str): string of h5 directory path

    Returns:
        Dict with individual file counts per directory/dataset
    """
    h5s = utils.get_files_in_dir(root=h5_dir, file_end=".h5")
    if h5s.size == 0:
        return None
    individual = {}
    for h5 in h5s:
        with h5py.File(os.path.join(h5_dir, h5), "r") as f:
            targets = list(f.keys())
            for target in targets:
                key = target.upper().split(h5[0:4])[0]
                try:
                    individual[key] += 1
                except KeyError:
                    individual[key] = 1
    out = {"total": h5s.size, "individual": individual}
    return out



def get_pd2_nd(
    roots: tuple[str, ...],
    zdict: dict,
    pd2_params: tuple[float, float] = (0.7, 3.0),
    nd_params: tuple[float, float] = (0.3, 4.0),
) -> tuple[list[str], list[str]]:
    """
    Given directories of dockfiles and the computed zdict, find a valid PD2
    and ND.  Takes search parameters in the form of [is_scr, rmsd].

    Inputs:
        roots (tuple[str, ...]): list of target directories
        zdict (dict): computed iAlign native-dockfile comparisons
        pd2_params (tuple[float, float]): is_scr, rmsd for positive set
        nd_params (tuple[float, float]): is_scr, rmsd for negative set

    Returns:
        tuple[list[str], list[str]] in format pd2, nd
    """
    pd2 = []
    nd = []
    for root in roots:
        target = utils.get_files_in_dir(root=root, file_end=".pdb")
        for file in target:
            ret = sim.get_zdock_dict_data(zdict, os.path.join(root, file))
            if ret:  # no empty dict
                try:
                    if (
                        float(ret["is_scr"]) >= pd2_params[0]
                        and float(ret["rmsd"]) <= pd2_params[1]
                    ):
                        pd2.append(os.path.join(root, file))
                    elif (
                        float(ret["is_scr"]) < nd_params[0]
                        and float(ret["rmsd"]) > nd_params[1]
                    ):
                        nd.append(os.path.join(root, file))
                except ValueError as e:
                    print(file, e)

    pd2_ref = []
    nd_ref = []
    for elem in pd2:
        if pd2_ref.count(elem) <= 2:
            pd2_ref.append(elem)

    for elem in nd:
        if nd_ref.count(elem) <= 2:
            nd_ref.append(elem)

    return (pd2_ref, nd_ref)


def sep_pd2_nd(
    pd2_list: list[str], nd_list: list[str], pos_dir: str, neg_dir: str
) -> None:
    """
    Given our calculated sets, just shove them into another directory.
    Okay, there's duplication but it makes EVERYTHING easier.

    Inputs:
        pd2_list (list[str]):
        nd_list (list[str]):
        pos_dir (str):
        neg_dir (str):

    Returns:
        None
    """
    for elem in pd2_list:
        shutil.copy(elem, pos_dir)
    for elem in nd_list:
        shutil.copy(elem, neg_dir)



def lookup_casp_old(file: str, gname: str, lookup_dir: str = r"./data/casp/results") -> int:
    """
    Relabels CASP data based on criteria from the pi-score paper
    file: File path of the PDB
    gname: name of the PDB structure
    """
    file = os.path.basename(file).split(".")[0]
    print(gname, file)
    
    lookup_file = os.path.join(lookup_dir, gname + ".txt")
    # print(lookup_file)
    
    if not os.path.isfile(lookup_file):
        raise FileNotFoundError(f"Lookup file {lookup_file} not found !!!")
        
    with open(lookup_file, "r") as f:
        data = f.readlines()
        
    gname = gname.removeprefix(file + "_")
    
    f1_scale_hundreds = False
 
    for line in data[4:]:  # skip empty lines:
        line_array = line.split()
        if len(line_array) == 0:
            # print("trigger")
            continue
        # print(line_array)
        if int(line_array[0]) == 1 and float(line_array[13]) > 1.0:
            f1_scale_hundreds = True
        if gname in line_array:
            try:
                f1, jaccard, lDDTo, TMscore = (
                    float(line_array[13]),
                    float(line_array[18]),
                    float(line_array[16]),
                    float(line_array[27]),
                )
                if f1_scale_hundreds:
                    f1 /= 100
                    f1 = np.round(f1, 3)

            except ValueError:  # occurs on NaN value due to failed float conv
                return -1
            
            if f1 >= 0.5 or jaccard >= 0.5 or lDDTo >= 0.5 or TMscore >= 0.5:
                return 1
            else:
                return 0
    
    return -1