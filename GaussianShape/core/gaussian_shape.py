"""Gaussian molecular shape representation and similarity comparison.

This module represents a molecule's 3D shape as a sum of isotropic Gaussian
functions centered on each atom::

    rho(r) = sum_i exp(-|r - r_i|^2 / (2 * sigma_i^2))

The key advantage of this representation is that the *overlap* between two
molecular shapes has a closed-form (analytical) solution, so we can measure
shape similarity without any atom-to-atom mapping and without a numerical grid.

For two Gaussians centered at ``a`` and ``b`` with widths ``sa`` and ``sb``:

    integral(g_a * g_b) dr = (2*pi*sa^2*sb^2 / (sa^2 + sb^2))^(3/2)
                             * exp(-|a - b|^2 / (2 * (sa^2 + sb^2)))

The total shape overlap between molecules A and B is the double sum over all
atom pairs, and shape similarity is reported as a Tanimoto-like coefficient.

References
----------
Grant, J. A.; Pickett, S. D. "A Gaussian Description of Molecular Shape."
J. Phys. Chem. 1995, 99, 3503-3510. (ROCS-style shape comparison.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Van der Waals radii in Angstrom for common organic elements.
# Source: Bondi, A. J. Phys. Chem. 1964, 68, 441.
VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
}
DEFAULT_RADIUS = 1.70  # fallback (carbon) for unknown elements


@dataclass
class GaussianShape:
    """A molecular shape modeled as a sum of isotropic atomic Gaussians.

    Parameters
    ----------
    centers : (N, 3) array
        Atom coordinates in Angstrom.
    sigmas : (N,) array
        Gaussian width (standard deviation) for each atom in Angstrom.
    elements : list of str, optional
        Element symbols, kept only for bookkeeping / plotting.
    """

    centers: np.ndarray
    sigmas: np.ndarray
    elements: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=float).reshape(-1, 3)
        self.sigmas = np.asarray(self.sigmas, dtype=float).reshape(-1)
        if self.sigmas.shape[0] != self.centers.shape[0]:
            raise ValueError(
                f"centers ({self.centers.shape[0]}) and sigmas "
                f"({self.sigmas.shape[0]}) must have the same length"
            )
        if np.any(self.sigmas <= 0):
            raise ValueError("all sigmas must be positive")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_coordinates(
        cls,
        coords,
        elements=None,
        radius_scale: float = 0.8,
    ) -> "GaussianShape":
        """Build a shape from raw coordinates and element symbols.

        The Gaussian width for each atom is derived from its van der Waals
        radius: ``sigma = radius_scale * R_vdw``. A scale near 0.7-0.8 makes
        each Gaussian roughly approximate the atom's hard-sphere volume.
        """
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        if elements is None:
            elements = ["C"] * coords.shape[0]
        elements = list(elements)
        radii = np.array(
            [VDW_RADII.get(e, DEFAULT_RADIUS) for e in elements], dtype=float
        )
        sigmas = radius_scale * radii
        return cls(centers=coords, sigmas=sigmas, elements=elements)

    @classmethod
    def from_rdkit_mol(
        cls,
        mol,
        conf_id: int = -1,
        radius_scale: float = 0.8,
        include_hydrogens: bool = True,
    ) -> "GaussianShape":
        """Build a shape from an RDKit molecule with a 3D conformer.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            Molecule that already has at least one 3D conformer embedded.
        conf_id : int
            Conformer id to use (default -1 = the molecule's default conformer).
        radius_scale : float
            Multiplier applied to van der Waals radii to obtain Gaussian widths.
        include_hydrogens : bool
            If False, hydrogen atoms are skipped (heavy-atom shape only).
        """
        conf = mol.GetConformer(conf_id)
        coords = []
        elements = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if not include_hydrogens and symbol == "H":
                continue
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append((pos.x, pos.y, pos.z))
            elements.append(symbol)
        if not coords:
            raise ValueError("no atoms selected to build the shape")
        return cls.from_coordinates(coords, elements, radius_scale=radius_scale)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return self.centers.shape[0]

    def centroid(self) -> np.ndarray:
        """Unweighted centroid of the atom centers."""
        return self.centers.mean(axis=0)

    def copy(self) -> "GaussianShape":
        return GaussianShape(
            centers=self.centers.copy(),
            sigmas=self.sigmas.copy(),
            elements=list(self.elements),
        )

    def translated(self, vector) -> "GaussianShape":
        """Return a copy translated by ``vector``."""
        new = self.copy()
        new.centers = new.centers + np.asarray(vector, dtype=float).reshape(3)
        return new

    def transformed(self, rotation, translation=None) -> "GaussianShape":
        """Return a copy with a rigid-body transform applied.

        Points are transformed as ``r' = R @ r + t``.
        """
        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        new = self.copy()
        new.centers = new.centers @ rotation.T
        if translation is not None:
            new.centers = new.centers + np.asarray(translation, dtype=float).reshape(3)
        return new

    # ------------------------------------------------------------------
    # Density evaluation (for visualization)
    # ------------------------------------------------------------------
    def density(self, points) -> np.ndarray:
        """Evaluate the shape density rho(r) at arbitrary points.

        Parameters
        ----------
        points : (M, 3) array
            Query points.

        Returns
        -------
        (M,) array of density values.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        # Squared distances between every query point and every atom center:
        # shape (M, N).
        diff = points[:, None, :] - self.centers[None, :, :]
        sq_dist = np.einsum("mnk,mnk->mn", diff, diff)
        gauss = np.exp(-sq_dist / (2.0 * self.sigmas[None, :] ** 2))
        return gauss.sum(axis=1)

    # ------------------------------------------------------------------
    # Overlap and similarity
    # ------------------------------------------------------------------
    def overlap(self, other: "GaussianShape") -> float:
        """Analytical Gaussian overlap integral with another shape.

        Computes ``sum_i sum_j integral(g_i * g_j) dr`` in closed form.
        """
        # Pairwise squared distances between the two atom sets: shape (Na, Nb).
        diff = self.centers[:, None, :] - other.centers[None, :, :]
        sq_dist = np.einsum("abk,abk->ab", diff, diff)

        sa2 = self.sigmas[:, None] ** 2  # (Na, 1)
        sb2 = other.sigmas[None, :] ** 2  # (1, Nb)
        sum_s2 = sa2 + sb2  # (Na, Nb)

        prefactor = (2.0 * np.pi * sa2 * sb2 / sum_s2) ** 1.5
        pair_overlap = prefactor * np.exp(-sq_dist / (2.0 * sum_s2))
        return float(pair_overlap.sum())

    def self_overlap(self) -> float:
        """Overlap of the shape with itself, ``S_AA``."""
        return self.overlap(self)

    def tanimoto(self, other: "GaussianShape") -> float:
        """Tanimoto-like shape similarity coefficient in [0, 1].

        ``T = S_AB / (S_AA + S_BB - S_AB)``. A value of 1 means the two shapes
        overlap perfectly; 0 means no overlap.
        """
        s_ab = self.overlap(other)
        s_aa = self.self_overlap()
        s_bb = other.self_overlap()
        denom = s_aa + s_bb - s_ab
        if denom <= 0:
            return 0.0
        return float(s_ab / denom)


# ----------------------------------------------------------------------
# Rigid-body alignment (maximize shape overlap)
# ----------------------------------------------------------------------
def _euler_to_matrix(angles) -> np.ndarray:
    """Build a rotation matrix from ZYX Euler angles (radians)."""
    rz, ry, rx = angles
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rot_z @ rot_y @ rot_x


def _euler_rotation_and_derivatives(angles):
    """Return ``R`` and its derivatives ``dR/drz, dR/dry, dR/drx``.

    Uses the same ZYX convention as :func:`_euler_to_matrix`. The three
    derivative matrices are obtained by replacing the relevant factor with its
    own derivative (chain rule on a matrix product).
    """
    rz, ry, rx = angles
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)

    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])

    # Element-wise derivatives of each single-axis rotation.
    d_rot_z = np.array([[-sz, -cz, 0.0], [cz, -sz, 0.0], [0.0, 0.0, 0.0]])
    d_rot_y = np.array([[-sy, 0.0, cy], [0.0, 0.0, 0.0], [-cy, 0.0, -sy]])
    d_rot_x = np.array([[0.0, 0.0, 0.0], [0.0, -sx, -cx], [0.0, cx, -sx]])

    rot = rot_z @ rot_y @ rot_x
    d_rz = d_rot_z @ rot_y @ rot_x
    d_ry = rot_z @ d_rot_y @ rot_x
    d_rx = rot_z @ rot_y @ d_rot_x
    return rot, (d_rz, d_ry, d_rx)


def _overlap_and_gradient(mobile_centers, mobile_sigmas, ref_centers, ref_sigmas,
                          params):
    """Overlap ``S_AB`` and its gradient w.r.t. ``params = [rz, ry, rx, tx, ty, tz]``.

    The mobile atoms are transformed as ``a' = R @ a + t`` and the raw Gaussian
    overlap with the (fixed) reference is returned together with the analytical
    6-vector gradient. All pairs are handled in a vectorized fashion.
    """
    angles = params[:3]
    translation = params[3:6]
    rot, (d_rz, d_ry, d_rx) = _euler_rotation_and_derivatives(angles)

    a_prime = mobile_centers @ rot.T + translation  # (Na, 3)

    # Pairwise differences a'_i - b_j and squared distances: (Na, Nb, 3)/(Na, Nb).
    diff = a_prime[:, None, :] - ref_centers[None, :, :]
    d2 = np.einsum("abk,abk->ab", diff, diff)

    sa2 = mobile_sigmas[:, None] ** 2  # (Na, 1)
    sb2 = ref_sigmas[None, :] ** 2     # (1, Nb)
    s2 = sa2 + sb2                     # (Na, Nb)

    prefactor = (2.0 * np.pi * sa2 * sb2 / s2) ** 1.5
    kernel = prefactor * np.exp(-d2 / (2.0 * s2))  # (Na, Nb)
    overlap = float(kernel.sum())

    # dS/d(diff_ij) = kernel_ij * (-diff_ij / s2_ij); accumulate per mobile atom.
    coeff = (-kernel / s2)[:, :, None]              # (Na, Nb, 1)
    grad_per_atom = np.einsum("abk->ak", coeff * diff)  # (Na, 3)

    grad_t = grad_per_atom.sum(axis=0)              # translation gradient
    grad_angles = np.array([
        np.sum(grad_per_atom * (mobile_centers @ d_rz.T)),
        np.sum(grad_per_atom * (mobile_centers @ d_ry.T)),
        np.sum(grad_per_atom * (mobile_centers @ d_rx.T)),
    ])
    grad = np.concatenate([grad_angles, grad_t])
    return overlap, grad


@dataclass
class AlignmentResult:
    """Result of aligning a mobile shape onto a reference shape.

    Attributes
    ----------
    aligned : GaussianShape
        The mobile shape after the optimal transform has been applied.
    tanimoto : float
        Tanimoto shape similarity achieved after alignment.
    rotation : (3, 3) array
        Rotation ``R`` that maps the *original* mobile coordinates.
    translation : (3,) array
        Translation ``t`` that maps the *original* mobile coordinates.

    The transform acts on original mobile coordinates as ``r' = R @ r + t``,
    so it can be applied directly to the source molecule for visualization.
    """

    aligned: GaussianShape
    tanimoto: float
    rotation: np.ndarray
    translation: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        """4x4 homogeneous transform matrix combining rotation and translation."""
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.translation
        return m

    def apply_to_coords(self, coords) -> np.ndarray:
        """Apply the transform to an (N, 3) array of coordinates."""
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        return coords @ self.rotation.T + self.translation


def apply_transform_to_mol(mol, rotation, translation, conf_id: int = -1):
    """Apply a rigid transform ``r' = R @ r + t`` to an RDKit conformer in place.

    Rewrites the atom positions of the given conformer so the molecule sits in
    the aligned frame. Returns the same ``mol`` for convenience.
    """
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    translation = np.asarray(translation, dtype=float).reshape(3)
    conf = mol.GetConformer(conf_id)
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        r = np.array([pos.x, pos.y, pos.z])
        r_new = rotation @ r + translation
        conf.SetAtomPosition(i, tuple(float(v) for v in r_new))
    return mol


def align_shapes_bfgs(
    mobile: "GaussianShape",
    reference: "GaussianShape",
    n_starts: int = 8,
    seed: int | None = 0,
) -> "AlignmentResult":
    """Align ``mobile`` onto ``reference`` with gradient-based BFGS optimization.

    Maximizes the analytical Gaussian overlap ``S_AB`` over a full rigid-body
    transform (3 rotation angles + 3 translations), using an exact gradient.
    Maximizing ``S_AB`` also maximizes the Tanimoto coefficient, because the
    self-overlaps are invariant under rigid motion.

    Both shapes are pre-centered on their centroids so that the initial
    translation guess is zero. Several random rotational starts are tried to
    escape local optima on the non-convex overlap surface.

    Returns
    -------
    AlignmentResult
        Contains the aligned shape, the Tanimoto score, and the rotation /
        translation that map the *original* mobile coordinates into place.
    """
    from scipy.optimize import minimize

    ref_centroid = reference.centroid()
    mob_centroid = mobile.centroid()
    ref_centered = reference.translated(-ref_centroid)
    mob_centered = mobile.translated(-mob_centroid)

    mob_centers = mob_centered.centers
    mob_sigmas = mob_centered.sigmas
    ref_centers = ref_centered.centers
    ref_sigmas = ref_centered.sigmas

    def neg_overlap_and_grad(params):
        overlap, grad = _overlap_and_gradient(
            mob_centers, mob_sigmas, ref_centers, ref_sigmas, params
        )
        return -overlap, -grad

    rng = np.random.default_rng(seed)
    best_result = None
    best_score = np.inf
    for i in range(max(1, n_starts)):
        x0 = np.zeros(6)
        if i > 0:
            x0[:3] = rng.uniform(-np.pi, np.pi, size=3)
        result = minimize(neg_overlap_and_grad, x0, method="BFGS", jac=True)
        if result.fun < best_score:
            best_score = result.fun
            best_result = result

    best_rot = _euler_to_matrix(best_result.x[:3])
    best_t = best_result.x[3:6]

    # Compose the transform that acts on the *original* mobile coordinates:
    #   r' = R (r - c_mob) + (t_opt + c_ref) = R r + (-R c_mob + t_opt + c_ref)
    rotation = best_rot
    translation = -best_rot @ mob_centroid + best_t + ref_centroid

    aligned = mobile.transformed(rotation, translation=translation)
    tanimoto = aligned.tanimoto(reference)
    return AlignmentResult(aligned, float(tanimoto), rotation, translation)


def align_shapes(
    mobile: "GaussianShape",
    reference: "GaussianShape",
    n_starts: int = 8,
    seed: int | None = 0,
) -> "AlignmentResult":
    """Find the rigid-body transform of ``mobile`` that best overlaps ``reference``.

    Rotation-only variant using the derivative-free Powell method (kept as a
    simple baseline against :func:`align_shapes_bfgs`). Both shapes are centered
    on their centroids, then a rotation is optimized to maximize Tanimoto.

    Returns
    -------
    AlignmentResult
        Contains the aligned shape, the Tanimoto score, and the rotation /
        translation that map the *original* mobile coordinates into place.
    """
    from scipy.optimize import minimize

    ref_centroid = reference.centroid()
    mob_centroid = mobile.centroid()
    ref_centered = reference.translated(-ref_centroid)
    mob_centered = mobile.translated(-mob_centroid)

    def neg_tanimoto(angles) -> float:
        rot = _euler_to_matrix(angles)
        candidate = mob_centered.transformed(rot)
        return -candidate.tanimoto(ref_centered)

    rng = np.random.default_rng(seed)
    best_result = None
    best_score = np.inf
    for i in range(max(1, n_starts)):
        x0 = np.zeros(3) if i == 0 else rng.uniform(-np.pi, np.pi, size=3)
        result = minimize(neg_tanimoto, x0, method="Powell")
        if result.fun < best_score:
            best_score = result.fun
            best_result = result

    best_rot = _euler_to_matrix(best_result.x)
    # Compose transform on original mobile coordinates (translation-only shift):
    #   r' = R (r - c_mob) + c_ref = R r + (-R c_mob + c_ref)
    rotation = best_rot
    translation = -best_rot @ mob_centroid + ref_centroid
    aligned = mobile.transformed(rotation, translation=translation)
    return AlignmentResult(aligned, float(-best_score), rotation, translation)
