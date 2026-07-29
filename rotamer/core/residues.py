"""Side-chain chi (chi) dihedral definitions for the standard amino acids.

A *rotamer* is a particular set of side-chain torsion (chi) angles. To place a
rotamer we need to know, for each residue type, which four atoms define each chi
angle. The quartets below follow the standard IUPAC / PDB convention and use the
PDB atom names that RDKit assigns when a peptide is built with
``Chem.MolFromSequence``.

Notes
-----
* ``ALA`` and ``GLY`` have no rotatable side chain, so they define no chi angles.
* ``PRO`` chi angles are part of a five-membered ring; rotating them would break
  the ring, so PRO is treated as rigid here (no chi entries).
* Only heavy-atom chi angles are defined (hydrogen torsions are not rotamers).
"""

from __future__ import annotations

# Mapping: 3-letter residue name -> list of chi angles, each a 4-tuple of PDB
# atom names (chi1, chi2, ... in order).
CHI_DEFINITIONS: dict[str, list[tuple[str, str, str, str]]] = {
    "ARG": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "NE"),
        ("CG", "CD", "NE", "CZ"),
    ],
    "ASN": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "OD1"),
    ],
    "ASP": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "OD1"),
    ],
    "CYS": [
        ("N", "CA", "CB", "SG"),
    ],
    "GLN": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "GLU": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "HIS": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "ND1"),
    ],
    "ILE": [
        ("N", "CA", "CB", "CG1"),
        ("CA", "CB", "CG1", "CD1"),
    ],
    "LEU": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "LYS": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "CE"),
        ("CG", "CD", "CE", "NZ"),
    ],
    "MET": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "SD"),
        ("CB", "CG", "SD", "CE"),
    ],
    "PHE": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "SER": [
        ("N", "CA", "CB", "OG"),
    ],
    "THR": [
        ("N", "CA", "CB", "OG1"),
    ],
    "TRP": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "TYR": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "VAL": [
        ("N", "CA", "CB", "CG1"),
    ],
}

# Residues with no rotatable side chain.
RIGID_RESIDUES = {"ALA", "GLY", "PRO"}


def n_chi(resname: str) -> int:
    """Number of chi angles for a residue (0 if rigid / unknown)."""
    return len(CHI_DEFINITIONS.get(resname.upper(), ()))


def chi_atom_names(resname: str) -> list[tuple[str, str, str, str]]:
    """Return the chi atom-name quartets for a residue (empty if none)."""
    return list(CHI_DEFINITIONS.get(resname.upper(), ()))
