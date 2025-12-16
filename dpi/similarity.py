"""
    @author: Nicholas Whyatt UKRI-STFC on behalf of CCP-EM
"""

from io import StringIO
from itertools import combinations
from subprocess import check_output

import os
import json

# import gemmi
import multiprocessing as mp
from . import helper

# TODO: Allow to edit and add/replace elements within a given interface
# similarity json with a flag
# TODO: Parallelise using multiprocessing
# exclude: str = ".{8}_.*"
def get_ialign_intra_output(
    target_dirs: tuple[str, ...],
    perl_path: str,
    ialign_path: str,
    out_dir: str,
    update_mode: str = "w",
    exclude: str = "^[a-zA-Z0-9_]+_",
    tmp_dir: str = r"/tmp/dip-score/ialign",
) -> dict:
    """
    Given a target directory(ies) of mixed pdb/cif files, parse iAlign
    similarity outputs into a centrally collated format.  Return a
    dictionary of directories -> files -> ialign outputs

    Inputs:
        target_dirs (tuple[str, ...]): target directories
        perl_path (str): your perl executable
        ialign_path (str): path to ialign.pl file
        out_dir (str): output directory
        update_mode (str): depending on mode will either overwrite
            or add.  WARNING, does not account for duplicate keys
        exclude (str): regex files to not include in search
        tmp_dir (str): directory to write temporary files to

    Returns:
        dict of iAlign output within interfaces
    """
    acc = {}
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    for target_dir in target_dirs:
        print("Evaluating iAlign output of dir", target_dir)
        basename = os.path.basename(target_dir)
        acc[basename] = {}
        files = helper.get_files_in_dir(
            root=target_dir, file_end=(".cif", ".pdb", ".ent"), exclude=exclude
        )
        for file in files:
            # must be getting a runtime error
            try:
                file_path, filename, resfile = helper.handle_cif_with_tmp(
                    file=os.path.join(target_dir, file), tmp_dir=tmp_dir
                )
            except (FileNotFoundError, RuntimeError) as e:
                # TODO: add logging
                print(e)
                continue

            acc[basename][filename] = {}

            with open(file=resfile, mode="rb") as handle:
                dict_intf_prop = json.load(handle)
            for ipair in combinations(dict_intf_prop.keys(), 2):
                intf_0 = ipair[0].replace("_", "")
                intf_1 = ipair[1].replace("_", "")
                cmd = [
                    perl_path,
                    ialign_path,
                    "-a",
                    "1",
                    "-w",
                    tmp_dir,
                    file_path,
                    intf_0,
                    file_path,
                    intf_1,
                ]
                ialign = check_output(cmd, shell=False)
                acc = parse_ialign_output(
                    ialign=ialign,
                    basename=basename,
                    filename=filename,
                    intf_0=intf_0,
                    intf_1=intf_1,
                    acc=acc,
                )

    with open(os.path.join(out_dir, "ialign_intra.json"), update_mode) as outfle:
        json.dump(acc, outfle)

    return acc


# perl_path, ialign_path, file_path, filename, basename,
# tmp_dir, mid_dir
def PD2_ND_ialign_call(
    perl_path: str,
    ialign_path: str,
    file_path: str,
    filename: str,
    basename: str,
    mid_dir: str,
    tmp_dir: str,
) -> dict:
    """
    Parallelisable call to ialign to generate a native file's
    """
    # structure
    # basename: em
    # filename: complex name without numbers (1C9S)
    # key: A_B.10
    acc = {}
    acc[basename] = {}
    acc[basename][filename] = {}
    # We need to find all the files in the mid_dir that regex
    # with the filename.  Can be done with a negative lookup.
    exclude = "^(?!" + filename + "[^_])"
    docked_targets = helper.get_files_in_dir(
        root=mid_dir, file_end=".pdb", exclude=exclude
    )
    print(docked_targets)
    for target in docked_targets:
        key = target.removeprefix(filename).removesuffix(".pdb")
        intf = "".join(key.split(".")[0].split("_"))
        full_target = os.path.join(mid_dir, target)
        cmd = [
            perl_path,
            ialign_path,
            "-a",
            "1",
            "-w",
            tmp_dir,
            file_path,
            intf,
            full_target,
            intf,
        ]
        ialign = check_output(cmd, shell=False)
        acc = parse_ialign_output_alt(
            ialign=ialign, basename=basename, filename=filename, key=key, acc=acc
        )
        # comes out with the right result.  Acc is not folding correctly with
        # .update
    return acc

def casp_ialign_call(
    perl_path: str,
    ialign_path: str,
    file_path: str,
    filename: str,
    basename: str,
    mid_dir: str,
    tmp_dir: str,
) -> dict:
    """
    Parallelisable call to ialign to generate a native file's
    """
    # structure
    # basename: em
    # filename: complex name without numbers (1C9S)
    # key: A_B.10
    acc = {}
    acc[basename] = {}
    acc[basename][filename] = {}
    # We need to find all the files in the mid_dir that regex
    # with the filename.  Can be done with a negative lookup.
    exclude = r"^T\d+TS\d+_[a-z0-9]+_interface_residues_dict\.json\.pdb$"
    docked_targets = helper.get_files_in_dir(
        root=mid_dir, file_end=".pdb", exclude=exclude
    )
    for target in docked_targets:
        key = target.removeprefix(filename).removesuffix(".pdb")
        # print(key)
        intf = "".join(key.split(".")[0].split("_"))
        full_target = os.path.join(mid_dir, target)
        cmd = [
            perl_path,
            ialign_path,
            "-a",
            "1",
            "-w",
            tmp_dir,
            file_path,
            full_target,
        ]
        ialign = check_output(cmd, shell=False)
        acc = save_ialign_output_saveall(
            ialign=ialign, basename=basename, filename=filename, key=key, acc=acc
        )
        # comes out with the right result.  Acc is not folding correctly with
        # .update
    return acc


def __dict_accumulator(res: dict, acc: dict) -> dict:
    """
    Internal accumulator method to safely add residuals to the primary dump
    dictionary.  Check for existence of highest level key, then insert
    individual proteins as insertions not updates.

    This method assumes that each PDB is completely performed in one job.

    Inputs:
        res (dict): dictionary of iAlign outputs for a given PDB/docking pair
        acc (dict): accumulated dictionary

    Returns:
        acc (dict): acc, now with res added
    """
    target = list(res.keys())[0]
    if target not in acc.keys():
        acc.update(res)
    else:
        new_res = res[target]
        acc[target].update(new_res)
    return acc


def get_PD2_ND_dict(
    mid_dirs: tuple[str, ...],
    ialign_path: str,
    tmp_dir: str,
    perl_path: str,
    native_dirs: tuple[str, ...],
    out_dir: str,
) -> dict:
    """
    Given a set of docked complexes, generate the iAlign dictionary of the
    similarity between the same interface docked and native.  Should
    be performed in a clean directory.

    Inputs:
        mid_dirs (str): directories of docked files and .outs
        ialign_path (str): full ialign path
        perl_path (str): full perl lath
        native_dirs (str): dirs of native structures to compare
        out_dir (str): out directory for dictionary
        update_mode (str): overwrite files or add

    Returns:
        dict of docked/native interface similarities
    """
    if os.path.exists(os.path.join(out_dir, "ialign_nat_dock.json")):
        os.remove(os.path.join(out_dir, "ialign_nat_dock.json"))

    acc = {}
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    for target_dir, mid_dir in zip(native_dirs, mid_dirs):
        print("Evaluating iAlign output of dir", target_dir)
        basename = os.path.basename(target_dir)
        acc[basename] = {}
        files = helper.get_files_in_dir(
            root=target_dir,
            file_end=(".cif", ".pdb", ".ent"),
            exclude="^(?:[^_]*_){2}[^_]*$",
        )
        targ_files = []
        for file in files:
            # append file metadata to a list or something
            # forms part of our payload
            try:
                file_path, filename, _ = helper.handle_cif_with_tmp(
                    file=os.path.join(target_dir, file), tmp_dir=tmp_dir
                )
                targ_files.append((file_path, filename))
            except (FileNotFoundError, RuntimeError):
                # TODO: add logging
                # print(e)
                continue

        targets = [
            (perl_path, ialign_path, file_path, filename, basename, mid_dir, tmp_dir)
            for (file_path, filename) in targ_files
        ]
        print(f"Beginning assessment on {len(targets)} targets.")
        with mp.Pool(mp.cpu_count()) as pool:
            out = pool.starmap(PD2_ND_ialign_call, targets)
        for res in out:
            acc = __dict_accumulator(res=res, acc=acc)

    # print(acc)
    with open(os.path.join(out_dir, "ialign_nat_dock.json"), "w") as outfle:
        json.dump(acc, outfle)

    return acc


# def get_PD2_ND_array(
#         filename: str, target_dict: dict, pd2_set: dict,
#         nd_set: dict, write: bool = True) -> tuple[list, list]:
#     '''
#     Given a target native file and a computed dict from get_PD2_ND_dict,
#     lookup the intf similarity and return a dict of files that fit the
#     conditions.

#     Inputs:
#         filename (str): native file
#         target_dict (dict): ialign_nat_dock.json
#         pd2_set (dict):
#         nd_set (dict):

#     Returns:
#         tuple with two sets
#     '''
#     pass


def __query_targets(
    target_dict: dict, intf: tuple[str, ...], final_targets: list[str], ialign_cutoff: float, rmsd_th: float,
) -> tuple[str, ...] | bool:
    """
    Internal method to figure out the is_scr of a given target is below the
    cutoff and should be appended or not.

    Inputs:
        targets (tuple[str, ...]): holds permutations of target key
        target_dict (dict): dict to query
        intf (tuple[str, ...]): interface pair
        final_targets (list[str]): prior accepted targets
        ialign_cutoff (float): max is_scr cutoff
        rmsd_th (float): min rmsd_th threshold

    Returns:
        bool on validity and existence
    """
    existence = False
    for final in final_targets:
        targets = (intf[1] + "_" + final[1], final[1] + "_" + intf[1])
        for target in targets:
            if target in target_dict:
                existence = intf
                
                try:
                    is_scr = float(target_dict[target]["is_scr"])
                    rmsd = float(target_dict[target]['rmsd'])
                except Exception:
                    return False
                    
                if (is_scr > ialign_cutoff) & (rmsd < rmsd_th):
                    return False
    return existence
    # final_targets.append(intf) # probline


def get_dissimilar_interfaces(
    filename: str, 
    intfs: dict, 
    ialign_dict: dict, 
    ialign_cutoff: float = 0.7, 
    rmsd_th: float = 3,
) -> list[str]:
    """
    Takes a filename and searches interfaces, alphabetically, for
    similarity to other interfaces.

    Inputs:
        filename (str): File path of target native structure
        intfs (dict): Dictionray of lists {'B_A': [['MET275', ... ], ['LEU605', .. ]]}
        ialign_dict (dict): ialign dictionary
        ialign_cutoff (float): is_scr cutoff - anything higher fails
        rmsd_th (float): rmsd threshold  - anything lower fails
    Returns:
        list[str] of valid interfaces to try
    """
    dir_key = os.path.basename(os.path.dirname(filename))
    # key = "".join(os.path.basename(filename).split(".")[0:-1]) # Why join? doesn't find the key in ialign_dict
    key = os.path.basename(filename).split(".")[0][0:4] # added by niraj
    
    # resfile = os.path.join(
    #     os.path.split(filename)[0], key + "_interface_residues_dict.json"
    # )
    # if not os.path.isfile(resfile):
    #     raise FileNotFoundError(f"resfile {resfile} n'existe pas")
    # with open(file=resfile, mode="rb") as handle:
    #     intfs = json.load(handle)

    # without difficult lookups, the join is irreversible
    intf_list = [(intf, "".join(intf.split("_"))) for intf in intfs.keys()] # [('B_A', 'BA')]

    # print(f"{dir_key=}, {key=}, {intfs=}, {intf_list=}")

    final_targets = []
    try:
        target_dict = ialign_dict[dir_key][key] # {'AB_CD': {'c0_contacts': '93', 'c1_contacts': '95', ...'is_scr': ' 0.89802', 'rmsd': '  0.58', ..}}
    except KeyError:
        # PDB format can't handle long chains, so it didn't go through iAlign
        return [intf[0] for intf in intf_list]
    
    for intf in intf_list:
        if len(final_targets) == 0:
            final_targets.append(intf)
            continue
        else:
            out = __query_targets(
                target_dict=target_dict,
                intf=intf,
                final_targets=final_targets,
                ialign_cutoff=ialign_cutoff,
                rmsd_th=rmsd_th,
            )
            if out:
                final_targets.append(out)

    return [final[0] for final in final_targets]


# TODO replace with parse_ialign_output_alt by generating key earlier
def parse_ialign_output(
    ialign: bytes, basename: str, filename: str, intf_0: str, intf_1: str, acc: dict
) -> dict:
    """
    Given an ialign bytes buffer, parse it and dump it a given dictionary.
    Accumulator method.

    Inputs:
        ialign (bytes): undecoded bytestring output from ialign
        basename (str): basename of target directory, i.e. "dock_pos"
        filename (str | tuple[str, ...]): filename of target
        intf_0 (str): first interface
        intf_1 (str): second interface
        acc (dict): accumulator

    Returns:
        acc dict with added records
    """
    f = StringIO(ialign.decode())
    for i in f:
        i = i.rstrip()
        if i.startswith("Structure 1"):
            c0_residues = i.split(",")[1].replace("AAs", "").strip()
            c0_contacts = i.split(",")[2].replace("Contacts", "").strip()
        elif i.startswith("Structure 2"):
            c1_residues = i.split(",")[1].replace("AAs", "").strip()
            c1_contacts = i.split(",")[2].replace("Contacts", "").strip()
        elif i.startswith("IS-score"):
            is_scr = i.split(",")[0].split("=")[-1]
            p_val = i.split(",")[1].split("=")[-1]
            zscr = i.split(",")[2].split("=")[-1]
        elif i.startswith("RMSD"):
            rmsd = i.split(",")[0].split("=")[-1]
        elif i.startswith("Number of aligned residues"):
            align_res = i.split("=")[-1]
        elif i.startswith("Number of aligned contacts"):
            align_ctcts = i.split("=")[-1]

    key = intf_0 + "_" + intf_1

    acc[basename][filename][key] = {}
    try:
        acc[basename][filename][key]["c0_contacts"] = c0_contacts
        acc[basename][filename][key]["c1_contacts"] = c1_contacts
        acc[basename][filename][key]["c0_residues"] = c0_residues
        acc[basename][filename][key]["c1_residues"] = c1_residues
        acc[basename][filename][key]["zscr"] = zscr
        acc[basename][filename][key]["p_val"] = p_val
        acc[basename][filename][key]["is_scr"] = is_scr
        acc[basename][filename][key]["align_res"] = align_res
        acc[basename][filename][key]["rmsd"] = rmsd
        acc[basename][filename][key]["align_ctcts"] = align_ctcts
    except NameError:
        # TODO: log alignment failures
        acc[basename][filename][key]["is_scr"] = 0.0

    return acc


def get_zdock_dict_data(zdict: dict, filepath: str) -> dict:
    """
    Function to return the lowest level results dict of a given PDB dockfile.

    Inputs:
        zdict (dict): loaded dictionary of targets
        filepath (str): target file as an absolute path

    Returns:
        dict of lowest level
    """
    path, filename = os.path.split(filepath)
    targ_dir = os.path.basename(path)
    if targ_dir.startswith("process_"):
        targ_dir = targ_dir.removeprefix("process_")
    # gets rid of file endings
    pdb, dockfile = filename[0:4], filename[4:-4]
    try:
        return zdict[targ_dir][pdb][dockfile]
    except KeyError:
        return {}


def parse_ialign_output_alt(
    ialign: bytes, basename: str, filename: str, key: str, acc: dict
) -> dict:
    """
    Given an ialign bytes buffer, parse it and dump it a given dictionary.
    Accumulator method.  Takes a configurable key argument.

    Inputs:
        ialign (bytes): undecoded bytestring output from ialign
        basename (str): basename of target directory, i.e. "dock_pos"
        filename (str | tuple[str, ...]): filename of target
        intf_0 (str): first interface
        intf_1 (str): second interface
        acc (dict): accumulator

    Returns:
        acc dict with added records
    """
    f = StringIO(ialign.decode())
    for i in f:
        i = i.rstrip()
        if i.startswith("Structure 1"):
            c0_residues = i.split(",")[1].replace("AAs", "").strip()
            c0_contacts = i.split(",")[2].replace("Contacts", "").strip()
        elif i.startswith("Structure 2"):
            c1_residues = i.split(",")[1].replace("AAs", "").strip()
            c1_contacts = i.split(",")[2].replace("Contacts", "").strip()
        elif i.startswith("IS-score"):
            is_scr = i.split(",")[0].split("=")[-1]
            p_val = i.split(",")[1].split("=")[-1]
            zscr = i.split(",")[2].split("=")[-1]
        elif i.startswith("RMSD"):
            rmsd = i.split(",")[0].split("=")[-1]
        elif i.startswith("Number of aligned residues"):
            align_res = i.split("=")[-1]
        elif i.startswith("Number of aligned contacts"):
            align_ctcts = i.split("=")[-1]

    acc[basename][filename][key] = {}
    try:
        acc[basename][filename][key]["c0_contacts"] = c0_contacts
        acc[basename][filename][key]["c1_contacts"] = c1_contacts
        acc[basename][filename][key]["c0_residues"] = c0_residues
        acc[basename][filename][key]["c1_residues"] = c1_residues
        acc[basename][filename][key]["zscr"] = zscr
        acc[basename][filename][key]["p_val"] = p_val
        acc[basename][filename][key]["is_scr"] = is_scr
        acc[basename][filename][key]["align_res"] = align_res
        acc[basename][filename][key]["rmsd"] = rmsd
        acc[basename][filename][key]["align_ctcts"] = align_ctcts
    except NameError:
        # TODO: log alignment failures
        # print("Error on filename", basename, filename, key, e)
        pass
        # acc[basename][filename][key]['is_scr'] = 0.0

    return acc

def save_ialign_output_saveall(
    ialign: bytes, basename: str, filename: str, key: str, acc: dict
) -> dict:
    """
    Given an ialign bytes buffer, parse it and dump it a given dictionary.
    Accumulator method.  Takes a configurable key argument.

    Inputs:
        ialign (bytes): undecoded bytestring output from ialign
        basename (str): basename of target directory, i.e. "dock_pos"
        filename (str | tuple[str, ...]): filename of target
        intf_0 (str): first interface
        intf_1 (str): second interface
        acc (dict): accumulator

    Returns:
        acc dict with added records
    """
    acc[basename][filename][key] = {}
    f = StringIO(ialign.decode())
    for i in f:
        i = i.rstrip()
        if i.startswith('>>>'):
            try:
                acc[basename][filename][key][native_model_intf_pair]["c0_contacts"] = c0_contacts
                acc[basename][filename][key][native_model_intf_pair]["c1_contacts"] = c1_contacts
                acc[basename][filename][key][native_model_intf_pair]["c0_residues"] = c0_residues
                acc[basename][filename][key][native_model_intf_pair]["c1_residues"] = c1_residues
                acc[basename][filename][key][native_model_intf_pair]["zscr"] = zscr
                acc[basename][filename][key][native_model_intf_pair]["p_val"] = p_val
                acc[basename][filename][key][native_model_intf_pair]["is_scr"] = is_scr
                acc[basename][filename][key][native_model_intf_pair]["align_res"] = align_res
                acc[basename][filename][key][native_model_intf_pair]["rmsd"] = rmsd
                acc[basename][filename][key][native_model_intf_pair]["align_ctcts"] = align_ctcts
            except NameError:
                # TODO: log alignment failures
                # print("Error on filename", basename, filename, key, e)
                pass
                # acc[basename][filename][key]['is_scr'] = 0.0
            i_n = i.split()
            native_model_intf_pair = i_n[0][-2:] + '_' + i_n[-1][-2:]
            acc[basename][filename][key][native_model_intf_pair] = {}
        if i.startswith("Structure 1"):
            c0_residues = i.split(",")[1].replace("AAs", "").strip()
            c0_contacts = i.split(",")[2].replace("Contacts", "").strip()
        elif i.startswith("Structure 2"):
            c1_residues = i.split(",")[1].replace("AAs", "").strip()
            c1_contacts = i.split(",")[2].replace("Contacts", "").strip()
        elif i.startswith("IS-score"):
            is_scr = i.split(",")[0].split("=")[-1]
            p_val = i.split(",")[1].split("=")[-1]
            zscr = i.split(",")[2].split("=")[-1]
        elif i.startswith("RMSD"):
            rmsd = i.split(",")[0].split("=")[-1]
        elif i.startswith("Number of aligned residues"):
            align_res = i.split("=")[-1]
        elif i.startswith("Number of aligned contacts"):
            align_ctcts = i.split("=")[-1]
    try:
        acc[basename][filename][key][native_model_intf_pair]["c0_contacts"] = c0_contacts
        acc[basename][filename][key][native_model_intf_pair]["c1_contacts"] = c1_contacts
        acc[basename][filename][key][native_model_intf_pair]["c0_residues"] = c0_residues
        acc[basename][filename][key][native_model_intf_pair]["c1_residues"] = c1_residues
        acc[basename][filename][key][native_model_intf_pair]["zscr"] = zscr
        acc[basename][filename][key][native_model_intf_pair]["p_val"] = p_val
        acc[basename][filename][key][native_model_intf_pair]["is_scr"] = is_scr
        acc[basename][filename][key][native_model_intf_pair]["align_res"] = align_res
        acc[basename][filename][key][native_model_intf_pair]["rmsd"] = rmsd
        acc[basename][filename][key][native_model_intf_pair]["align_ctcts"] = align_ctcts
    except NameError:
        # TODO: log alignment failures
        # print("Error on filename", basename, filename, key, e)
        pass
        # acc[basename][filename][key]['is_scr'] = 0.0
    return acc

def select_similar_interface(dict_ialign):
    """
    Given an ialign output saved in a dictionary with struture defined in 
    save_ialign_output_saveall function

    Inputs:
        acc dict with added records

    Returns:
        mapped interfaces
    """
    dict_ial = {}
    for k in dict_ialign:
        for native in dict_ialign[k]:
            for models in dict_ialign[k][native]:
                dict_ial[models] = {}
                m,n = [],[]
                for intfs in dict_ialign[k][native][models]:
                    model_intf = intfs.split('_')[1]
                    native_intf = intfs.split('_')[0]
                    n.append(native_intf)
                    m.append(model_intf)
                total_mtf = list(set(m))
                tmp = []
                for mf in total_mtf:
                    for intfs in dict_ialign[k][native][models]:
                        model_intf = intfs.split('_')[1]
                        native_intf = intfs.split('_')[0]
                        if mf == model_intf: 
                            if native_intf not in tmp:
                                # print(mf,intfs,dict_ialign[k][native][models][intfs]['rmsd'],tmp)
                                if model_intf not in dict_ial[models]:
                                    dict_ial[models][model_intf] = {native_intf:[dict_ialign[k][native][models][intfs]['rmsd'],dict_ialign[k][native][models][intfs]['is_scr']]}
                                else:
                                    for i in dict_ial[models][model_intf]:
                                        if float(dict_ial[models][model_intf][i][0]) > float(dict_ialign[k][native][models][intfs]['rmsd']):
                                            dict_ial[models][model_intf] = {native_intf:[dict_ialign[k][native][models][intfs]['rmsd'],dict_ialign[k][native][models][intfs]['is_scr']]}
                    sel_native_intf = list(dict_ial[models][mf].keys())
                    tmp.extend(sel_native_intf)
    return dict_ial