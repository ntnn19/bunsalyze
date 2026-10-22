import os
import io
import json
import ast
from pathlib import Path
from pprint import pprint
from collections import defaultdict, namedtuple
from typing import Mapping, Sequence

_thread_vars = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for key, val in _thread_vars.items():
    if key not in os.environ:
        os.environ[key] = val

import freesasa
import prody as pr
import numpy as np
from scipy.spatial.distance import cdist
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

from bunsalyze.utils.constants import PolarAtom 
from bunsalyze.utils.calc_ligand_dons_accs import get_ligand_polar_atoms, compute_ligand_capacity
from bunsalyze.utils.calc_protein_dons_accs import get_protein_polar_atoms
from bunsalyze.utils.burial_calc import compute_fast_ligand_burial_mask
from bunsalyze.utils.graph import PolarAtomGraph

freesasa.setVerbosity(freesasa.silent)
RDLogger.DisableLog('rdApp.*') # Disables all RDKit logging

_BACKBONE_NAMES = frozenset({"N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3", "HA", "HA2", "HA3"})


def compute_shell_residues(
    lig_ag: pr.AtomGroup, prot_ag: pr.AtomGroup,
    shell_dist: float = 5.0, n_shells: int = 3
) -> dict:
    """
    Assign protein residues to sidechain-contact shells expanding outward from the ligand.

    Shell 1: residues with any sidechain heavy atom within shell_dist of any ligand heavy atom.
    Shell k: not-yet-assigned residues with a sidechain heavy atom within shell_dist of any
             shell-(k-1) sidechain heavy atom.

    Returns {(chain, resnum, icode): shell_index} for residues reached within n_shells expansions.
    """
    lig_heavy = lig_ag.select("not hydrogen")
    prev_coords = (lig_heavy if lig_heavy is not None else lig_ag).getCoords()

    res_sc = {}
    for atom in prot_ag.iterAtoms():
        if atom.getName() in _BACKBONE_NAMES or atom.getElement().upper() == 'H':
            continue
        key = (str(atom.getChid()), int(atom.getResnum()), str(atom.getIcode()).strip())
        res_sc.setdefault(key, []).append(atom.getCoords())
    res_sc = {k: np.asarray(v, dtype=np.float64) for k, v in res_sc.items()}

    assignment = {}
    for shell_idx in range(1, n_shells + 1):
        new_coords = []
        for key, coords in res_sc.items():
            if key in assignment:
                continue
            if cdist(prev_coords, coords).min() <= shell_dist:
                assignment[key] = shell_idx
                new_coords.append(coords)
        if not new_coords:
            break
        prev_coords = np.concatenate(new_coords, axis=0)

    return assignment


def compute_shell_buns(
    ligand_buns: list, protein_buns: list,
    protein_buried_fraction_unsat: dict, ligand_buried_per_atom_capacity: dict,
    shell_assignment: dict, n_shells: int,
) -> dict:
    """
    Build cumulative per-shell BUNs and capacity scores.

    The H-bond graph is solved globally; here we only restrict which protein BUNs are counted.
    Ligand BUNs are always included (the ligand is the centre of every shell).
    cumulative_*[k] covers shells 1..(k+1).
    """
    lig_buns_score = 2 * len(ligand_buns)
    lig_capacity = 2 * sum(ligand_buried_per_atom_capacity.values())

    cumulative_buns, cumulative_cap, per_shell_prot = [], [], []
    prev_prot_count = 0
    for depth in range(1, n_shells + 1):
        in_shell = {res for res, idx in shell_assignment.items() if idx <= depth}
        # protein_buns entries: (name, chain, resname, resnum, icode, is_weak_acceptor)
        n_prot = sum(
            1 for e in protein_buns
            if (str(e[1]), int(e[3]), str(e[4]).strip()) in in_shell
        )
        # protein_buried_fraction_unsat keys: (chain, resname, resnum, icode)
        prot_cap = sum(
            v for k, v in protein_buried_fraction_unsat.items()
            if (str(k[0]), int(k[2]), str(k[3]).strip()) in in_shell
        )
        cumulative_buns.append(lig_buns_score + n_prot)
        cumulative_cap.append(lig_capacity + prot_cap)
        per_shell_prot.append(n_prot - prev_prot_count)
        prev_prot_count = n_prot

    return {
        'cumulative_buns_score': cumulative_buns,
        'cumulative_capacity_score': cumulative_cap,
        'per_shell_protein_buns_count': per_shell_prot,
    }


# One ligand residue with its RDKit molecules: `res_ag` is the single-residue prody
# AtomGroup, `smi_mol` the SMILES template used for bond-order assignment, `lig_mol` the
# RDKit mol built from that residue's PDB coordinates, and `residue_id` a human-readable
# label used in error messages.
LigandEntry = namedtuple('LigandEntry', ['res_ag', 'smi_mol', 'lig_mol', 'residue_id'])


def _resolve_smiles_per_residue(smiles, residues) -> list:
    """Map user-supplied SMILES onto ligand residues.

    `smiles` may be:
      - a single SMILES string (or single-element sequence): broadcast to every ligand
        residue (e.g. multiple copies of the same ligand in a homomer);
      - a sequence with one SMILES per ligand residue, matched to ligand residues sorted
        by (chain, resnum, icode);
      - a mapping locating residues by resname, chain ID, or 'chain:resnum' (see below).

    `residues` is a list of (resname, chain, resnum, icode, residue_id) tuples in that
    same sorted order.

    Mapping keys:
      - a residue name ('FAD'): covers every ligand residue with that resname;
      - a chain ID ('B'): covers the ligand residue in that chain — rejected if the
        chain contains more than one ligand residue (use 'chain:resnum' then);
      - 'chain:resnum' ('B:501', optionally with icode, 'B:501A'): covers exactly one
        ligand residue.
    Every key must match at least one residue, every residue must be covered exactly
    once, and a key that is both a chain ID and a resname among the selected ligand
    residues is rejected as ambiguous.
    """
    if isinstance(smiles, str):
        return [smiles] * len(residues)
    if isinstance(smiles, Mapping):
        chains = {r[1] for r in residues}
        resnames = {r[0] for r in residues}
        resolved = [None] * len(residues)
        for key, smi in smiles.items():
            key = str(key)
            if ':' in key:
                kchain, kres = key.split(':', 1)
                matches = [
                    i for i, r in enumerate(residues)
                    if r[1] == kchain and (kres == str(r[2]) or (r[3] and kres == f"{r[2]}{r[3]}"))
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"SMILES mapping key {key!r} matches {len(matches)} ligand residues; "
                        f"expected exactly one. Ligand residues: "
                        f"{', '.join(r[4] for r in residues)}."
                    )
            elif key in chains and key in resnames:
                raise ValueError(
                    f"SMILES mapping key {key!r} is ambiguous: it is both a chain ID and a "
                    "residue name among the selected ligand residues. Use the "
                    "'chain:resnum' form instead."
                )
            elif key in chains:
                matches = [i for i, r in enumerate(residues) if r[1] == key]
                if len(matches) > 1:
                    raise ValueError(
                        f"SMILES mapping key {key!r} matches {len(matches)} ligand residues in "
                        f"chain {key} ({', '.join(residues[i][4] for i in matches)}). Use the "
                        "'chain:resnum' form to specify them individually."
                    )
            elif key in resnames:
                matches = [i for i, r in enumerate(residues) if r[0] == key]
            else:
                raise ValueError(
                    f"SMILES mapping key {key!r} matches no ligand residue. Available chain "
                    f"IDs: {sorted(chains)}; resnames: {sorted(resnames)}."
                )
            for i in matches:
                if resolved[i] is not None:
                    raise ValueError(
                        f"Ligand residue {residues[i][4]} is covered by more than one SMILES "
                        "mapping key."
                    )
                resolved[i] = smi
        uncovered = [r[4] for r, s in zip(residues, resolved) if s is None]
        if uncovered:
            raise ValueError(
                f"No SMILES provided for ligand residue(s): {', '.join(uncovered)}. "
                f"Mapping keys were: {sorted(str(k) for k in smiles)}."
            )
        return resolved
    smiles = list(smiles)
    if len(smiles) == 1:
        return smiles * len(residues)
    if len(smiles) != len(residues):
        raise ValueError(
            f"Got {len(smiles)} SMILES strings but the ligand selection matches "
            f"{len(residues)} residues ({', '.join(r[4] for r in residues)}). Provide "
            "either one SMILES (broadcast to all ligand residues) or one SMILES per ligand "
            "residue in (chain, resnum, icode) order."
        )
    return list(smiles)


def parse_complex_and_build_rdkit_ligand(pr_complex: pr.AtomGroup, smiles, ligand_selection_string: str):
    """Split the complex into protein and ligand atom groups and build one RDKit mol per
    ligand residue.

    Multiple ligand residues are supported; see `_resolve_smiles_per_residue` for how
    `smiles` is mapped onto residues. Returns (prot_ag, lig_ag, ligand_entries) where
    `lig_ag` is the combined ligand AtomGroup and `ligand_entries` is a list of
    LigandEntry, one per ligand residue, sorted by (chain, resnum, icode).
    """
    prot_ag = pr_complex.select(f'not {ligand_selection_string}').copy()
    lig_sel = pr_complex.select(ligand_selection_string)
    if lig_sel is None or lig_sel.numAtoms() == 0:
        raise ValueError(
            f"Ligand selection '{ligand_selection_string}' matched no atoms. "
            "Check --override_ligand_selection_string."
        )
    lig_ag = lig_sel.copy()

    # Enumerate ligand residues in deterministic (chain, resnum, icode) order.
    resindices, chids, resnums, icodes, resnames = (
        lig_ag.getResindices(), lig_ag.getChids(), lig_ag.getResnums(),
        lig_ag.getIcodes(), lig_ag.getResnames(),
    )
    seen, residues = set(), []
    for i, ridx in enumerate(resindices):
        if ridx not in seen:
            seen.add(ridx)
            residues.append((ridx, str(chids[i]), int(resnums[i]), str(icodes[i]), str(resnames[i])))
    residues.sort(key=lambda r: (r[1], r[2], r[3]))

    smiles_per_residue = _resolve_smiles_per_residue(
        smiles, [(r[4], r[1], r[2], r[3], f"{r[4]}:{r[1]}:{r[2]}{r[3]}") for r in residues]
    )

    # Precondition: every ligand atom must carry an element. bunsalyze relies on per-atom elements
    # for H-bond donor/acceptor and burial analysis (e.g. prody `element H within ...` in
    # get_ligand_polar_atoms). A blank PDB element column (cols 77-78) makes prody report empty
    # elements, which used to surface only as a cryptic `TypeError: object of type 'NoneType' has no
    # len()` deep in calc_ligand_dons_accs. Fail here with an actionable message instead. (This is
    # how the tmol MMFF writer regression silently NaN'd 100% of divlib01 buns scores.)
    lig_elements = [str(e).strip() for e in lig_ag.getElements()]
    n_blank = sum(1 for e in lig_elements if not e)
    if n_blank:
        raise ValueError(
            f"{n_blank}/{len(lig_elements)} ligand atoms have a blank element symbol (PDB element "
            f"column, cols 77-78, not populated) for selection '{ligand_selection_string}'. bunsalyze "
            f"requires per-atom elements; fix the upstream PDB writer to emit the element column."
        )

    ligand_entries = []
    for (ridx, chid, resnum, icode, resname), smi in zip(residues, smiles_per_residue):
        residue_id = f"{resname}:{chid}:{resnum}{icode}"
        smi_mol = Chem.MolFromSmiles(smi)
        if smi_mol is None:
            raise ValueError(f"RDKit could not parse the ligand SMILES: {smi!r}")

        res_ag = lig_ag.select(f'resindex {ridx}').toAtomGroup()
        buff = io.StringIO()
        pr.writePDBStream(buff, res_ag)
        lig_mol = Chem.MolFromPDBBlock(buff.getvalue())
        if lig_mol is None:
            raise ValueError(
                f"RDKit could not build the ligand molecule (MolFromPDBBlock returned None) for "
                f"ligand residue {residue_id}. Common cause: missing/ambiguous element symbols "
                f"(e.g. halogens) in the ligand PDB records."
            )
        try:
            lig_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, lig_mol)
        except Exception as e:
            raise ValueError(
                f"Could not assign bond orders for ligand residue {residue_id} from SMILES "
                f"{smi!r}: {e}. If the complex contains multiple different ligands, provide one "
                "SMILES per ligand residue (matched in chain/resnum order) or a "
                "{resname: SMILES} mapping."
            ) from e
        ligand_entries.append(LigandEntry(res_ag=res_ag, smi_mol=smi_mol, lig_mol=lig_mol, residue_id=residue_id))

    return prot_ag, lig_ag, ligand_entries


def _ligand_atom_label(atom: PolarAtom, multi_ligand: bool) -> str:
    """Key for ligand-atom annotation dicts. Bare atom names collide across ligand copies,
    so qualify with the residue identity when the complex has multiple ligand residues."""
    if not multi_ligand:
        return atom.name
    chain, resname, resnum, icode, *_ = atom.parent_group_identifier
    return f"{atom.name}:{chain}:{resname}:{resnum}:{icode}"


def set_burial_annotations(
    ligand_polar_atoms: Sequence[PolarAtom], protein_polar_atoms: Sequence[PolarAtom], 
    ca_coords: np.ndarray, input_path: os.PathLike, sasa_threshold: float, silent: bool,
    alpha_hull_alpha: float, ignore_sasa_threshold: bool, ignore_all_burial_criteria: bool,
    ignore_ligand_sasa_threshold: bool, multi_ligand: bool = False
):
    ligand_burial_mask, protein_burial_mask = None, None
    if not ignore_all_burial_criteria:
        ligand_burial_mask = compute_fast_ligand_burial_mask(ca_coords, np.array([x.coord for x in ligand_polar_atoms]), alpha=alpha_hull_alpha)
        protein_burial_mask = compute_fast_ligand_burial_mask(ca_coords, np.array([x.coord for x in protein_polar_atoms]), alpha=alpha_hull_alpha)

    freesasa_struct = freesasa.Structure(str(input_path), options={'hetatm': True, 'hydrogen': False})
    sasa_data = freesasa.calc(freesasa_struct)

    ligand_burial_annotations = {'ligand_atoms_in_hull': [], 'ligand_atoms_buried_sasa': [], 'ligand_atoms_sasa': {}, 'ligand_atoms_considered_buried': []}
    if not silent: print(f'Ligand Atom SASA: (< {sasa_threshold} A^2 == buried)')
    for i, atom in enumerate(ligand_polar_atoms):
        atom_name = atom.name
        label = _ligand_atom_label(atom, multi_ligand)
        chain, resname, resnum, *_ = atom.parent_group_identifier

        sasa = freesasa.selectArea([f's1, name {atom_name} and resn {resname} and chain {chain} and resi {resnum}'], freesasa_struct, sasa_data)['s1']

        ligand_burial_annotations['ligand_atoms_sasa'][label] = sasa
        if ignore_all_burial_criteria or ligand_burial_mask[i].item():
            ligand_burial_annotations['ligand_atoms_in_hull'].append(label)
        if sasa < sasa_threshold:
            ligand_burial_annotations['ligand_atoms_buried_sasa'].append(label)

        if ignore_all_burial_criteria:
            atom.is_buried = True
        else:
            atom.is_buried = ligand_burial_mask[i].item() and ((sasa < sasa_threshold) or ignore_sasa_threshold or ignore_ligand_sasa_threshold)
        
        if atom.is_buried:
            ligand_burial_annotations['ligand_atoms_considered_buried'].append(label)

        if not silent: print(f'\t{label} {sasa}')

    for i, atom in enumerate(protein_polar_atoms):
        atom_name = atom.name
        chain, resname, resnum, *_ = atom.parent_group_identifier
        sasa = freesasa.selectArea([f's1, name {atom_name} and resn {resname} and chain {chain} and resi {resnum}'], freesasa_struct, sasa_data)['s1']
        if ignore_all_burial_criteria:
            atom.is_buried = True
        else:
            atom.is_buried = protein_burial_mask[i].item() and ((sasa < sasa_threshold) or ignore_sasa_threshold )
    
    return ligand_burial_annotations


def compute_capacity_score(
    ligand_polar_atoms: Sequence[PolarAtom], protein_polar_atoms: Sequence[PolarAtom],
    multi_ligand: bool = False
) -> dict:
    ligand_per_atom_capacity = {}
    buried_ligand_per_atom_capacity = {}
    output_data = []
    for idx, polar_atoms in enumerate((ligand_polar_atoms, protein_polar_atoms)):
        residue_to_buried_atoms = defaultdict(list)
        residue_fraction_unsatisfied = {}
        residue_fraction_buried_unsatisfied = {}

        # Map from residue to polar atoms
        for atom in polar_atoms:
            residue_to_buried_atoms[atom.parent_group_identifier].append(atom)
        
        # Loop over each residue and compute the fraction of capacity not satisfied
        for residue, atoms in residue_to_buried_atoms.items():
            remaining_capacity, max_capacity, buried_remaining_capacity, max_buried_capacity = 0, 0, 0, 0
            for atom in atoms:
                if (atom.name not in ('N', 'O', 'OXT', 'SD', 'CA')) or (idx == 0):
                    remaining_capacity += atom.donor_count + atom.acceptor_count
                    max_capacity += atom.max_donor_count + atom.max_acceptor_count
                    if atom.is_buried:
                        buried_remaining_capacity += atom.donor_count + atom.acceptor_count
                        max_buried_capacity += atom.max_donor_count + atom.max_acceptor_count
                    
                    if idx == 0:
                        label = _ligand_atom_label(atom, multi_ligand)
                        if atom.is_buried:
                            buried_ligand_per_atom_capacity[label] = (atom.donor_count + atom.acceptor_count) / (atom.max_donor_count + atom.max_acceptor_count) if (atom.max_donor_count + atom.max_acceptor_count) > 0 else 1.0
                        ligand_per_atom_capacity[label] = (atom.donor_count + atom.acceptor_count) / (atom.max_donor_count + atom.max_acceptor_count) if (atom.max_donor_count + atom.max_acceptor_count) > 0 else 1.0

            # Calculate the fraction of capacity satisfied for buried and non-buried atoms.
            fraction_buried_capacity_satisfied = (max_buried_capacity - buried_remaining_capacity) / max_buried_capacity if max_buried_capacity else 1.0
            fraction_capacity_satisfied = (max_capacity - remaining_capacity) / max_capacity if max_capacity else 1.0

            # If not fully satisfied, add to the unsatisfied lists.
            if fraction_capacity_satisfied < 1.0:
                residue_fraction_unsatisfied[residue] = 1 - fraction_capacity_satisfied
            if fraction_buried_capacity_satisfied < 1.0:
                residue_fraction_buried_unsatisfied[residue] = 1 - fraction_buried_capacity_satisfied
        
        output_data.append((residue_fraction_unsatisfied, residue_fraction_buried_unsatisfied))
    
    return {
        'ligand_buried_fraction_unsat': output_data[0][1], 
        'protein_buried_fraction_unsat': output_data[1][1],
        'ligand_fraction_unsat': output_data[0][0], 
        'protein_fraction_unsat': output_data[1][0],
        'ligand_per_atom_capacity': ligand_per_atom_capacity,
        'ligand_buried_per_atom_capacity': buried_ligand_per_atom_capacity
    }


def main(
    input_path: os.PathLike, smiles,
    sasa_threshold: float = 2.5, silent: bool = True, disable_hydrogen_clash_check: bool = False,
    alpha_hull_alpha: float = 14.0, override_ligand_selection_string: str = 'not protein',
    ncaa_dict: dict = {}, ignore_sulfur_acceptors: bool = False, ignore_sasa_threshold: bool = False,
    use_ca_donors: bool = False, ignore_all_burial_criteria: bool = False,
    covalent_hydrogen_max_distance: float = 1.2, ignore_ligand_intramolecular_hbonds: bool = False,
    report_weak_acceptor_buns: bool = False, ignore_ligand_sasa_threshold: bool = False,
    shells: int = 3, shell_dist: float = 5.0,
) -> dict:

    protein_complex = pr.parsePDB(str(input_path))
    assert protein_complex is not None, f"Failed to parse PDB file at {input_path}"

    # Load the relevant protein and ligand data. One RDKit mol is built per ligand residue;
    # a single SMILES is broadcast to all residues (e.g. multiple copies of the same ligand).
    prot_ag, lig_ag, ligand_entries = parse_complex_and_build_rdkit_ligand(protein_complex, smiles, override_ligand_selection_string)
    multi_ligand = len(ligand_entries) > 1
    ca_coords = prot_ag.select('name CA').getCoords()

    # Get the hbond-able polar atoms. Atom names are unique within a single residue, so the
    # per-residue AtomGroup keeps the name-based lookups in get_ligand_polar_atoms unambiguous.
    ligand_polar_atoms = []
    for entry in ligand_entries:
        lig_cap = compute_ligand_capacity(entry.lig_mol)
        ligand_polar_atoms.extend(get_ligand_polar_atoms(
            lig_cap, entry.res_ag, entry.lig_mol,
            covalent_hydrogen_max_distance=covalent_hydrogen_max_distance,
        ))
    protein_polar_atoms = get_protein_polar_atoms(prot_ag, ncaa_dict=ncaa_dict, use_sulfur_acceptors=not ignore_sulfur_acceptors, use_ca_donors=use_ca_donors, silent=silent)
    ligand_burial_annotations = set_burial_annotations(ligand_polar_atoms, protein_polar_atoms, ca_coords, input_path, sasa_threshold=sasa_threshold, silent=silent, alpha_hull_alpha=alpha_hull_alpha, ignore_sasa_threshold=ignore_sasa_threshold, ignore_all_burial_criteria=ignore_all_burial_criteria, ignore_ligand_sasa_threshold=ignore_ligand_sasa_threshold, multi_ligand=multi_ligand)

    # Build a radius graph of the polar atoms and compute the buns for the ligand and protein.
    g = PolarAtomGraph(
        ligand_polar_atoms, protein_polar_atoms, run_hydrogen_atom_clash_check=not disable_hydrogen_clash_check,
        ignore_ligand_intramolecular_hbonds=ignore_ligand_intramolecular_hbonds, debug=not silent
    )
    ligand_buns = g.compute_ligand_buns(report_weak_acceptor_buns=report_weak_acceptor_buns)
    protein_buns = g.compute_protein_buns()

    n_buried_lig = sum(
        1 for a in ligand_polar_atoms
        if a.is_buried and (report_weak_acceptor_buns or not a.is_weak_acceptor)
    )
    ligand_buried_binary_unsat = len(ligand_buns) / n_buried_lig if n_buried_lig > 0 else 0.0

    fraction_unsat_dicts = compute_capacity_score(ligand_polar_atoms, protein_polar_atoms, multi_ligand=multi_ligand)

    protein_buns_tuples = [(str(i.name), *i.parent_group_identifier, i.is_weak_acceptor) for i in protein_buns]
    shell_assignment = compute_shell_residues(lig_ag, prot_ag, shell_dist=shell_dist, n_shells=shells)
    shell_buns = compute_shell_buns(
        ligand_buns, protein_buns_tuples,
        fraction_unsat_dicts['protein_buried_fraction_unsat'],
        fraction_unsat_dicts['ligand_buried_per_atom_capacity'],
        shell_assignment, shells,
    )

    # Per-ligand scores. Burial and the H-bond graph are solved on the whole complex (so
    # inter-ligand H-bonds and mutual burial are fully accounted); only attribution is per
    # residue: ligand BUNs belong to their residue, and each ligand's site score counts the
    # protein BUNs within its own sidechain-contact shells. A protein BUN within shell range
    # of two ligands is attributed to both sites; protein BUNs outside every ligand's shells
    # appear only in the global scores.
    per_ligand = []
    for entry in ligand_entries:
        res_key = (
            str(entry.res_ag.getChids()[0]), str(entry.res_ag.getResnames()[0]),
            int(entry.res_ag.getResnums()[0]), str(entry.res_ag.getIcodes()[0]),
        )
        atoms_i = [a for a in ligand_polar_atoms if a.parent_group_identifier == res_key]
        buns_i = [a for a in ligand_buns if a.parent_group_identifier == res_key]
        n_buried_i = sum(
            1 for a in atoms_i
            if a.is_buried and (report_weak_acceptor_buns or not a.is_weak_acceptor)
        )
        labels_i = {_ligand_atom_label(a, multi_ligand) for a in atoms_i}
        cap_buried_i = {
            k: v for k, v in fraction_unsat_dicts['ligand_buried_per_atom_capacity'].items()
            if k in labels_i
        }
        shell_assignment_i = compute_shell_residues(entry.res_ag, prot_ag, shell_dist=shell_dist, n_shells=shells)
        shell_buns_i = compute_shell_buns(
            buns_i, protein_buns_tuples,
            fraction_unsat_dicts['protein_buried_fraction_unsat'],
            cap_buried_i, shell_assignment_i, shells,
        )
        per_ligand.append({
            'residue_id': entry.residue_id,
            'ligand_buns': [(str(a.name), *a.parent_group_identifier, a.is_weak_acceptor) for a in buns_i],
            'n_ligand_buns': len(buns_i),
            'n_buried_ligand_polar_atoms': n_buried_i,
            'ligand_buried_binary_unsat': len(buns_i) / n_buried_i if n_buried_i > 0 else 0.0,
            'ligand_buried_per_atom_capacity': cap_buried_i,
            'shell_buns': {'shell_dist': shell_dist, 'n_shells': shells, **shell_buns_i},
        })

    output = {
        'input_path': str(input_path),
        'n_ligand_residues': len(ligand_entries),
        'per_ligand': per_ligand,
        'ligand_buns': [(str(i.name), *i.parent_group_identifier, i.is_weak_acceptor) for i in ligand_buns],
        'protein_buns': [(str(i.name), *i.parent_group_identifier, i.is_weak_acceptor) for i in protein_buns],
        'buns_score': (2 * len(ligand_buns)) + len(protein_buns),
        'buns_capacity_score': 2 * sum(fraction_unsat_dicts['ligand_buried_per_atom_capacity'].values()) + sum(fraction_unsat_dicts['protein_buried_fraction_unsat'].values()),
        'ligand_buried_binary_unsat': ligand_buried_binary_unsat,
        'shell_buns': {'shell_dist': shell_dist, 'n_shells': shells, **shell_buns},
        **fraction_unsat_dicts,
        **ligand_burial_annotations,
    }

    return output



def _jsonable(o):
    """Recursively convert tuple dict keys (e.g. residue identifiers in the fraction_unsat
    dicts) and tuples to JSON-serializable forms."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def cli():
    """Command-line interface entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Analyze protein-ligand complexes for buried unsatisfied polar atoms (BUNs).")
    parser.add_argument("input_path", type=str, help="Path to the PDB file containing the protein-ligand complex")
    parser.add_argument("smiles", type=str, nargs='*', help="SMILES string(s) for the ligand(s). A single SMILES is broadcast to every ligand residue (e.g. multiple copies of the same ligand); with multiple different ligands, give one SMILES per ligand residue in (chain, resnum) order, or use --smiles_map.")
    parser.add_argument('--smiles_map', type=str, default='', help='Dictionary string mapping ligand residues to SMILES, keyed by resname, chain ID, or chain:resnum — e.g. \'{"FAD": "..."}\' or \'{"B": "...", "C": "..."}\'. Alternative to positional SMILES; the fail-proof option for multiple different ligands.')
    parser.add_argument("--sasa_threshold", type=float, default=2.5, help="SASA threshold for burial (default: 2.5 Å²)")
    parser.add_argument("--alpha_hull_alpha", type=float, default=14.0, help="Alpha-shape alpha parameter (larger = looser, closer to the convex hull; default is 14.0)")
    parser.add_argument("--output", type=str, help="Output file path (default: print to stdout)")
    parser.add_argument("--disable_hydrogen_clash_check", action='store_true', help="Default behavior doesn't count hbonds made at the expense of a hydrogen vdW clash. Set this flag to disable that check.")
    parser.add_argument('--override_ligand_selection_string', type=str, default='not protein', help='How to select the ligand from the PDB file, default is "not protein" but this fails with noncanonical amino acids.')
    parser.add_argument('--ignore_ligand_intramolecular_hbonds', action='store_true', help='If set, ignores intramolecular hbonds within the ligand when determining if a hbond is satisfied. Default behavior counts intramolecular ligand hbonds as satisfying capacity.')

    ncaa_dict_example_str = r'{"DJD": {"N": (0, ["H"]), "O": (2, []), "N03": (1, []), "N04": (1, []), "N05": (1, []), "N06": (1, [])}}'
    parser.add_argument('--ncaa_dict', type=str, default='', help=f'Dictionary mapping ncaa 3-letter code to polar atoms which map to tuples of (# hbonds atom can accept, list of atom names of attached donor hydrogens). Format: \'{ncaa_dict_example_str}\'')
    parser.add_argument('--ignore_sulfur_acceptors', action='store_true', help='If set, ignores sulfur atoms as potential acceptors. Default behavior includes sulfur atoms as acceptors.')
    parser.add_argument('--ignore_sasa_threshold', action='store_true', help='If set, does not use a SASA threshold to determine burial, only uses convex hull. Default behavior uses both SASA and convex hull.')
    parser.add_argument('--ignore_ligand_sasa_threshold', action='store_true', help='If set, does not use a SASA threshold to determine ligand burial, only uses convex hull. Default behavior uses both SASA and convex hull for the ligand.')
    parser.add_argument('--use_ca_donors', action='store_true', help='If set, uses CA atoms as potential donors. Default behavior does not use CA atoms as hbond donors.')
    parser.add_argument('--verbose', '-v', action='store_true', help='If set, prints additional information about the analysis to the console.')
    parser.add_argument('--report_weak_acceptor_buns', action='store_true', help='If set, reports weak acceptors (such as ligand aromatic nitrogen atoms without hydrogens) as BUNs. Default is False.')
    parser.add_argument('--shells', type=int, default=3, help='Number of sidechain-contact shells to expand from the ligand for the shell_buns breakdown (default: 3). Shell 1: residues with a sidechain atom within --shell_dist of any ligand heavy atom; shell k: residues within --shell_dist of any shell-(k-1) sidechain atom.')
    parser.add_argument('--shell_dist', type=float, default=5.0, help='Heavy-atom contact distance (Å) used to define shell membership (default: 5.0).')
    args = parser.parse_args()

    ncaa_dict = {}
    if args.ncaa_dict is not None and args.ncaa_dict != '':
        try:
            ncaa_dict = ast.literal_eval(args.ncaa_dict)
        except Exception as e:
            raise ValueError(f'Error parsing ncaa_dict argument: {e}. Please provide a valid dictionary string, e.g. \'{ncaa_dict_example_str}\'')

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {args.input_path} does not exist.")

    if args.smiles and args.smiles_map:
        parser.error("Provide either positional SMILES or --smiles_map, not both.")
    if not args.smiles and not args.smiles_map:
        parser.error("Provide ligand SMILES: positional argument(s) or --smiles_map.")
    smiles = args.smiles
    if args.smiles_map:
        try:
            smiles = ast.literal_eval(args.smiles_map)
        except Exception as e:
            raise ValueError(f"Error parsing --smiles_map argument: {e}. Provide a dictionary string, e.g. {{'B': 'CCO'}}")
        if not isinstance(smiles, dict):
            raise ValueError(f"--smiles_map must be a dictionary string, e.g. {{'B': 'CCO'}}; got {type(smiles).__name__}.")

    results = main(
        args.input_path, smiles,
        sasa_threshold=args.sasa_threshold, silent=not args.verbose, disable_hydrogen_clash_check=args.disable_hydrogen_clash_check,
        alpha_hull_alpha=args.alpha_hull_alpha, override_ligand_selection_string=args.override_ligand_selection_string,
        ncaa_dict=ncaa_dict, ignore_sulfur_acceptors=args.ignore_sulfur_acceptors,
        ignore_sasa_threshold=args.ignore_sasa_threshold, use_ca_donors=args.use_ca_donors,
        ignore_ligand_intramolecular_hbonds=args.ignore_ligand_intramolecular_hbonds,
        report_weak_acceptor_buns=args.report_weak_acceptor_buns,
        ignore_ligand_sasa_threshold=args.ignore_ligand_sasa_threshold,
        shells=args.shells, shell_dist=args.shell_dist,
    )
    
    if args.output:
        pprint(results)
        with open(args.output, 'w') as f:
            json.dump(_jsonable(results), f, indent=2, default=str)
    else:
        pprint(results)


if __name__ == '__main__':
    cli()
