"""Per-ligand chemistry resolution: build the RDKit template mol that defines a
ligand's bond orders, protonation state, and H-bond capacity — from explicit
SMILES, the input mmCIF's own bond table, a CCD lookup, or geometry perception.

Resolution hierarchy (per ligand residue, first hit wins):
  1. explicit user SMILES (positional / --smiles_map)   -> source 'user'
  2. the input mmCIF's own chemistry                    -> source 'mmcif_bond_table'
     or 'mmcif_pdbx_smiles': the _chem_comp_bond/_chem_comp_atom tables (RCSB
     entries), or _chem_comp.pdbx_smiles (what AlphaFold 3 writes for its
     ligands — preserving the SMILES AF3 was given, protonation included)
  3. CCD lookup by resname (RCSB chemcomp API)          -> source 'ccd:<RESNAME>'
  4. geometry perception from coordinates               -> source 'geometry_perception'
Auto-derived sources (2-4) are best-effort: if the resulting template does not
match the residue's heavy-atom graph, resolution falls through to the next
source. A user-supplied SMILES that fails is a hard error instead. Residues for
which every source fails are excluded from scoring (recorded, loudly).
"""
import logging
import os
import urllib.request

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger("bunsalyze.chemistry")

# Resnames that carry no chemistry information; never attempted for CCD lookup.
GENERIC_RESNAMES = {"", "LIG", "UNL", "UNK", "DUM", "MOL", "XXX", "L", "X"}

_CCD_CACHE: dict = {}  # resname -> SMILES or None (negative-cached)

_BOND_ORDER_MAP = {
    "sing": Chem.BondType.SINGLE,
    "doub": Chem.BondType.DOUBLE,
    "trip": Chem.BondType.TRIPLE,
    "arom": Chem.BondType.AROMATIC,
}


def template_from_smiles(smiles: str):
    """RDKit mol from a SMILES string, or None if unparseable."""
    return Chem.MolFromSmiles(smiles)


def _template_from_bond_table(block, resname: str):
    """Template mol from the entry mmCIF's _chem_comp_atom/_chem_comp_bond
    tables, or None if absent/unsanitizable. Formal charges are not stored in
    entry mmCIFs, so charged reference states are approximated by bond orders
    alone (same limitation as a neutral CCD SMILES)."""
    atoms = block.get_mmcif_category("_chem_comp_atom.")
    bonds = block.get_mmcif_category("_chem_comp_bond.")
    if not atoms or not bonds:
        return None
    try:
        atom_rows = [
            (a, t, f)
            for c, a, t, f in zip(
                atoms["comp_id"], atoms["atom_id"], atoms["type_symbol"],
                atoms.get("pdbx_aromatic_flag", ["N"] * len(atoms["comp_id"])))
            if c == resname
        ]
        if not atom_rows:
            return None
        rw = Chem.RWMol()
        idx_of = {}
        for atom_id, element, arom in atom_rows:
            rd_atom = Chem.Atom(element.capitalize())
            rd_atom.SetIsAromatic(arom == "Y")
            idx_of[atom_id] = rw.AddAtom(rd_atom)
        for c, a1, a2, order, arom in zip(
                bonds["comp_id"], bonds["atom_id_1"], bonds["atom_id_2"],
                bonds["value_order"],
                bonds.get("pdbx_aromatic_flag", ["N"] * len(bonds["comp_id"]))):
            if c != resname or a1 not in idx_of or a2 not in idx_of:
                continue
            bt = _BOND_ORDER_MAP.get(order.lower())
            if bt is None:
                return None
            rw.AddBond(idx_of[a1], idx_of[a2], bt)
            if bt == Chem.BondType.AROMATIC or arom == "Y":
                rw.GetBondBetweenAtoms(idx_of[a1], idx_of[a2]).SetIsAromatic(True)
        mol = rw.GetMol()
        # The table is the fully protonated reference compound; the template
        # must be the heavy-atom framework (implicit Hs carry protonation).
        mol = Chem.RemoveHs(mol, sanitize=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        return mol
    except Exception as e:
        logger.debug("mmCIF bond-table template failed for %s: %s", resname, e)
        return None


def _smiles_from_chem_comp(block, resname: str):
    """The entry mmCIF's own SMILES for `resname` from _chem_comp.pdbx_smiles
    (written by AlphaFold 3 for its ligands; preserves the SMILES AF3 was
    given, protonation state included). None if the category/column is absent."""
    cc = block.get_mmcif_category("_chem_comp.")
    if not cc or "pdbx_smiles" not in cc:
        return None
    for comp_id, smi in zip(cc["id"], cc["pdbx_smiles"]):
        if comp_id == resname and smi and smi not in ("?", "."):
            return smi
    return None


def template_from_mmcif(cif_path: str, resname: str):
    """Build a template mol for `resname` from the input mmCIF's own chemistry
    records. Returns (mol, source) — source 'mmcif_bond_table' from the
    _chem_comp_bond table (RCSB entries) or 'mmcif_pdbx_smiles' from
    _chem_comp.pdbx_smiles (AlphaFold 3 outputs) — or None if the file carries
    no usable chemistry for this resname."""
    try:
        import gemmi
        block = gemmi.cif.read_file(str(cif_path)).sole_block()
        mol = _template_from_bond_table(block, resname)
        if mol is not None:
            return mol, "mmcif_bond_table"
        smi = _smiles_from_chem_comp(block, resname)
        if smi is not None:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                return mol, "mmcif_pdbx_smiles"
        return None
    except Exception as e:
        logger.debug("mmCIF chemistry extraction failed for %s: %s", resname, e)
        return None


def smiles_from_ccd(resname: str, timeout: float = 10.0):
    """Canonical SMILES for a CCD resname via the RCSB chemcomp API.
    Returns None for generic resnames, unknown codes, or network failure.
    Results are cached per process (including failures)."""
    resname = str(resname).strip().upper()
    if resname in GENERIC_RESNAMES:
        return None
    if resname in _CCD_CACHE:
        return _CCD_CACHE[resname]
    smi = None
    try:
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{resname}"
        with urllib.request.urlopen(url, timeout=timeout) as fh:
            import json
            smi = json.load(fh)["rcsb_chem_comp_descriptor"]["SMILES"]
        if not smi or Chem.MolFromSmiles(smi) is None:
            smi = None
    except Exception as e:
        logger.debug("CCD lookup failed for %s: %s", resname, e)
        smi = None
    _CCD_CACHE[resname] = smi
    return smi


def template_from_geometry(pdb_block: str):
    """Perceive bond orders from 3D coordinates alone. Least reliable source —
    geometry-dependent and blind to tautomers/protonation. Returns None on
    failure."""
    mol = Chem.MolFromPDBBlock(pdb_block)
    if mol is not None:
        return mol
    try:
        from rdkit.Chem import rdDetermineBonds
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
        if mol is None:
            return None
        rdDetermineBonds.DetermineBonds(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        logger.debug("geometry perception failed: %s", e)
        return None


def assign_bond_orders(template_mol, lig_mol):
    """AssignBondOrdersFromTemplate with a Kekulized-template retry for the
    aromatic-lactam valence failure mode (see bunsalyze.py)."""
    try:
        return AllChem.AssignBondOrdersFromTemplate(template_mol, lig_mol)
    except Exception:
        kek = Chem.Mol(template_mol)
        Chem.Kekulize(kek, clearAromaticFlags=True)
        return AllChem.AssignBondOrdersFromTemplate(kek, lig_mol)


def candidate_templates(resname: str, pdb_block: str, input_path: str = "",
                        allow_ccd_lookup: bool = True) -> list:
    """Auto-resolution candidates for one ligand residue, in priority order:
    input mmCIF bond table, CCD lookup, geometry perception. Each entry is
    (template_mol, source, canonical_smiles). Quiet: warnings for actually
    used fallbacks are emitted by the caller at selection time."""
    resname = str(resname)
    candidates = []
    if input_path and os.path.splitext(str(input_path))[1].lower() in (".cif", ".mmcif", ".bcif"):
        hit = template_from_mmcif(str(input_path), resname)
        if hit is not None:
            mol, source = hit
            candidates.append((mol, source, Chem.MolToSmiles(mol)))
    if allow_ccd_lookup:
        smi = smiles_from_ccd(resname)
        if smi is not None:
            mol = template_from_smiles(smi)
            if mol is not None:
                candidates.append((mol, f"ccd:{resname}", Chem.MolToSmiles(mol)))
    mol = template_from_geometry(pdb_block)
    if mol is not None:
        candidates.append((mol, "geometry_perception", Chem.MolToSmiles(mol)))
    return candidates
