from itertools import combinations
from subprocess import run, CalledProcessError

import os
import gemmi
import shutil

from . import helper
# create.pl
# c_code = run([create_pl, out_file])
# try:
#     c_code.check_returncode()
# except CalledProcessError:
#     raise RuntimeError


def gen_zdock(
    file: str,
    intf_list: list[str],
    st: str,
    n_structs: int,
    zdock_path: str,
    out_dir: str,
    ow: bool = False,
) -> list[str]:
    """
    Parallelisable method meant to generate a set of zdock models for a
    given file and interface.  Generates the zdock.out file.
    No longer uses tmp directory due to mounting issues.

    Inputs:
        file (str): target filepath
        intf_list (list[str]): target interfaces
        st (gemmi.Structure): target structure
        n_structs (int): how many structures to generate
        zdock_path (str): path to zdock executable
        out_dir (str): final structures directory
        ow (bool): overwrite files

    Returns:
        list of str paths to outfiles, to be processed with create.pl
    """
    out_files = []
    for intf in intf_list:
        if not ow and os.path.exists(
            os.path.join(out_dir, os.path.basename(file)[0:4] + intf + ".out")
        ):
            out_files.append(
                os.path.join(out_dir, os.path.basename(file)[0:4] + intf + ".out")
            )
            continue
        intf_list = intf.split("_")
        # Non-overwriting chain files
        receptor = helper.write_single_chain_pdb(
            st=st, chain_name=intf_list[0], filename=file, out_dir=out_dir, ow=False
        )
        print(receptor)
        ligand = helper.write_single_chain_pdb(
            st=st, chain_name=intf_list[1], filename=file, out_dir=out_dir, ow=False
        )
        print(ligand)
        zdock_root = os.path.dirname(zdock_path)
        mark_sur = os.path.join(zdock_root, "mark_sur")
        # create_pl = os.path.join(zdock_root, "create.pl")

        r_mark = os.path.join(out_dir, receptor[0:-4] + "_m.pdb")
        l_mark = os.path.join(out_dir, ligand[0:-4] + "_m.pdb")
        if not os.path.exists(r_mark):
            r_code = run([mark_sur, receptor, r_mark], shell=False)
            try:
                r_code.check_returncode()
            except CalledProcessError:
                raise RuntimeError
        if not os.path.exists(l_mark):
            l_code = run([mark_sur, ligand, l_mark], shell=False)
            try:
                l_code.check_returncode()
            except CalledProcessError:
                raise RuntimeError

        # zdock
        out_file = os.path.join(out_dir, os.path.basename(file)[0:4] + intf + ".out")
        z_code = run(
            [
                zdock_path,
                "-R",
                r_mark,
                "-L",
                l_mark,
                "-N",
                str(n_structs),
                "-o",
                out_file,
            ],
            shell=False,
        )
        try:
            z_code.check_returncode()
        except CalledProcessError:
            pass

        out_files.append(out_file)

    return out_files


def ad_hoc_zdock_reparser(file: str, rename: str = "") -> str:
    """
    Function to eliminate random output from the side of zdock
    that doesn't work in Chimera or Gemmi or iAlign.  Hacky.  I despise it.

    Inputs:
        file (str): target file full path
        rename (str): new FILENAME (not full path)

    Returns:
        path of final file
    """
    with open(file=file, mode="r+") as f:
        lines = f.readlines()
        newlines = []
        offset = 0
        for line_no, line in enumerate(lines):
            # add TERs ourselves, and properly
            # ignores TERs before ATOM records
            if line.startswith("ATOM") or line.startswith("HETATOM"):
                # newline = line[0:54]
                newline = list(line)[0:68]
                newline.append("\n")

                res = "".join(newline[17:20])
                chain_id = newline[21]
                res_id = "".join(newline[22:26])
                # slightly evil numbering hack
                atom_serial = f"{(line_no+offset):>5}"
                for idx in range(len(newline)):
                    if 6 <= idx < 11:
                        newline[idx] = atom_serial[idx - 6]
                newline = "".join(newline)
                newlines.append(newline)
                # Trigger if last line, or TER line, or chain_id change
                # must also check if TER is for atom
                # nd lines[line_no].startswith("ATOM or HETATOM")
                if (
                    len(lines) - 1 == line_no
                    or (
                        lines[line_no + 1].startswith("TER")
                        and lines[line_no].startswith(("HETATOM", "ATOM"))
                    )
                    or lines[line_no + 1][21] != chain_id
                ):
                    offset += 1
                    a_s = f"{(line_no+offset):>5}"
                    new_ter = f"TER   {a_s}      {res} {chain_id}{res_id}"
                    new_ter += " " * 56
                    new_ter += "\n"
                    newlines.append(new_ter)

        newlines.append("END\n")
        f.truncate(0)
        f.seek(0)
        f.writelines(newlines)

    st = gemmi.read_structure(file)
    st.write_minimal_pdb(file)

    newlines = []
    with open(file=file, mode="r+") as f:
        lines = f.readlines()
        for line in lines:
            newline = line.rstrip()
            newline += "\n"
            newlines.append(newline)

        f.truncate(0)
        f.seek(0)
        f.writelines(newlines)

    if rename != "":
        new_file = os.path.join(os.path.dirname(file), rename)
        os.rename(file, new_file)
        return new_file

    return file


def zdock_createpl(createpl: str, outfile: str) -> list[str]:
    """
    Calls create.pl, then reparses the outfiles to format them
    and rename them.  Not parallelisable.

    Inputs:
        createpl (str): path to create.pl
        outfile (str): zdock.out

    Returns:
        list of renamed, reformatted files
    """
    directory = os.path.dirname(outfile)
    pwd = os.getcwd()
    os.chdir(directory)
    if not os.path.exists(os.path.join(directory, "create_lig")):
        # raise RuntimeError("create_lig not found in the working dir")
        shutil.copy(os.path.join(os.path.dirname(createpl), "create_lig"), directory)
    c_code = run(args=[createpl, outfile], shell=False)
    try:
        c_code.check_returncode()
    except CalledProcessError:
        raise RuntimeError("zdock_create pl failed")
    count = 1
    ret = []
    while os.path.exists(os.path.join(directory, "complex." + str(count) + ".pdb")):
        file = os.path.join(directory, "complex." + str(count) + ".pdb")
        rename = outfile[0:-4] + "_zdock_" + str(count) + ".pdb"
        ret.append(ad_hoc_zdock_reparser(file=file, rename=rename))
        count += 1
    os.chdir(pwd)
    return ret


def extract_docked(roots: tuple[str, ...], createpl: str) -> None:
    """
    Method to handle all the .out files within a set of directories.

    Inputs:
        roots (tuple[str, ...]): root directories to search for .out files
        createpl (str): full path to createpl method

    Returns:
        None
    """
    for root in roots:
        target = helper.get_files_in_dir(root=root, file_end=".out")
        for file in target:
            zdock_createpl(createpl=createpl, outfile=os.path.join(root, file))


def docked_cleanup(roots: tuple[str, ...], mid_dir: str) -> None:
    """
    Handles intermediary docked files by removing the individual
    chains and receptor/ligand pairs and the .out files to an
    intermediary directory.  Only used in native directories
    (with .cif files)

    Inputs:
        roots (tuple[str, ...]): target directories
        mid_dir (str): where gen'd .out should magically go to

    Returns:
        None
    """
    for root in roots:
        helper.gather_files(target_dir=root, dest_dir=mid_dir, file_end=".out")
        to_remove = helper.get_files_in_dir(root=root, file_end=(".pdb", ".out"))
        for file in to_remove:
            if not os.path.exists(os.path.join(mid_dir, file)) and file.endswith(
                ".out"
            ):
                continue
            os.remove(os.path.join(root, file))


def test_diffdock_target_crude(file_path: str, num_runs: int, out_dir: str) -> None:
    """
    Given a target file, calculate docking interface using
    DB5_inference.sh for every chain within a target.

    Crude implementation: makes no accomodation to using only
    the actual interfaces.  Makes as many runs as you specify. Be careful
    with the bash script this runs.

    Inputs:
        file_path (str): string filepath of structure
        num_runs (int): how many runs to attempt
        out_dir (str): where to write the intermediary PDBs

    Returns:
        Naught
    """
    if not os.path.isfile(file_path):
        return -1

    st = gemmi.read_structure(file_path)
    print(file_path)
    if file_path.endswith((".pdb", ".ent")):
        st.setup_entities()
        st.assign_label_seq_id()
    st.remove_ligands_and_waters()

    chain_comb_list = list(combinations(range(len(st[0])), 2))

    for chain_comb in chain_comb_list:
        ca_0 = st[0][chain_comb[0]].name
        ca_1 = st[0][chain_comb[1]].name

        filename = os.path.basename(file_path)
        try:
            target_0 = helper.write_single_chain_pdb(
                st=st, chain_name=ca_0, filename=filename, out_dir=out_dir
            )
            target_1 = data.write_single_chain_pdb(
                st=st, chain_name=ca_1, filename=filename, out_dir=out_dir
            )
        except RuntimeError:
            return -1

        print(target_0)
        print(target_1)

# TODO: docstring
def get_docked_files(
    file: str,
    ialign_dict: dict,
    cutoff: float,
    n_structs: int,
    out_dir: str,
    method_path: str,
    method: str = "zdock",
) -> list[str]:
    """
    Generates docked structures for a given structure file.

    Inputs:
        file (str): native structure
        ialign_dict (dict): computed internal similarity dict
        cutoff (float): is_scr cutoff for considering an intf "similar"
        n_structs (int): number of structures to gen
        out_dir (str): where to shunt the output dockfiles
        method_path (str): path to method executable (merge with method?)
        method (str): method type, currently only zdock
    Returns:
        list of generated structures
    """
    try:
        dis_intfs = sim.get_dissimilar_interfaces(
            filename=file, ialign_dict=ialign_dict, cutoff=cutoff
        )
    except FileNotFoundError:
        # No interfaces as no resfile
        return []

    for dis_intf in dis_intfs:
        if len(dis_intf) > 5:
            return []

    st = helper.struct_handler(file=file)
    if method == "zdock":
        out_files = gen_zdock(
            file=file,
            intf_list=dis_intfs,
            st=st,
            n_structs=n_structs,
            zdock_path=method_path,
            out_dir=out_dir,
        )
    else:
        raise NotImplementedError(f"method {method} is not recognised")

    return out_files


# TODO: docstring
def get_docked(
    root: str,
    ialign_dict_path: str,
    cutoff: float,
    n_structs: int,
    out_dir: str,
    method_path: str,
    method: str = "zdock",
) -> list[str]:
    """
    Method to get the "noisy" dataset of zdock-calculated structures.
    """

    # Perform initial docking method specific setup before thread-pool
    pwd = os.getcwd()
    with open(ialign_dict_path, "rb") as handle:
        ialign_dict = json.load(handle)
    if method == "zdock":
        # TODO: Implement for single case (i.e. not called through
        # get_docked())
        os.chdir(out_dir)
        zdock_dir = os.path.dirname(method_path)
        if not os.path.exists(os.path.join(out_dir, "create_lig")):
            shutil.copyfile(
                os.path.join(zdock_dir, "create_lig"),
                os.path.join(out_dir, "create_lig"),
            )
        if not os.path.exists(os.path.join(out_dir, "uniCHARMM")):
            shutil.copyfile(
                os.path.join(zdock_dir, "uniCHARMM"), os.path.join(out_dir, "uniCHARMM")
            )
    else:
        raise NotImplementedError(f"method {method} is not recognised")

    # heavy computation inbound...
    files = [
        os.path.join(root, file)
        for file in helper.get_files_in_dir(
            root=root, file_end=(".cif", ".pdb", ".ent"), exclude="^[a-zA-Z0-9_]+_"
        )
    ]
    print(files)
    # docking method specific arguments
    if method == "zdock":
        targets = [
            (file, ialign_dict, cutoff, n_structs, out_dir, method_path, "zdock")
            for file in files
        ]
    else:
        raise NotImplementedError(f"method {method} is not recognised")

    with mp.Pool(mp.cpu_count()) as pool:
        out = pool.starmap(get_docked_files, tqdm.tqdm(targets, total=len(files)))

    os.chdir(pwd)
    final = []
    map(final.extend, out)
    return final

