"""Generate pedagogical figures for GaussianShape_tutorial.md.

Run from the project root::

    python docs/make_tutorial_figs.py

Figures are written to docs/images/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(IMG_DIR, exist_ok=True)


def fig_gaussian_1d() -> None:
    """A single atomic Gaussian for three different sigma (width) values."""
    x = np.linspace(-5, 5, 400)
    fig, ax = plt.subplots(figsize=(6, 4))
    for sigma in (0.7, 1.2, 2.0):
        y = np.exp(-(x ** 2) / (2 * sigma ** 2))
        ax.plot(x, y, label=f"sigma = {sigma}")
    ax.set_title("Single-atom Gaussian g(r) = exp(-|r - r_i|^2 / 2 sigma^2)")
    ax.set_xlabel("distance from atom center r - r_i (Angstrom)")
    ax.set_ylabel("g(r)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "gaussian_1d.png"), dpi=130)
    plt.close(fig)


def fig_two_atom_density() -> None:
    """Sum of two atomic Gaussians -> a smooth 1D 'molecular' density."""
    x = np.linspace(-6, 6, 500)
    centers = [-1.5, 1.5]
    sigma = 1.0
    fig, ax = plt.subplots(figsize=(6, 4))
    total = np.zeros_like(x)
    for c in centers:
        g = np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
        total += g
        ax.plot(x, g, "--", alpha=0.6, label=f"atom at {c}")
    ax.plot(x, total, "k", lw=2, label="sum rho(r)")
    ax.set_title("Molecular density = sum of atomic Gaussians")
    ax.set_xlabel("position (Angstrom)")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "two_atom_density.png"), dpi=130)
    plt.close(fig)


def fig_overlap_vs_distance() -> None:
    """Analytical overlap of two 3D Gaussians as a function of separation."""
    d = np.linspace(0, 6, 300)
    s1 = s2 = 1.0
    sum_s2 = s1 ** 2 + s2 ** 2
    prefactor = (2 * np.pi * s1 ** 2 * s2 ** 2 / sum_s2) ** 1.5
    overlap = prefactor * np.exp(-(d ** 2) / (2 * sum_s2))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d, overlap, "b", lw=2)
    ax.fill_between(d, overlap, alpha=0.15)
    ax.set_title("Pairwise Gaussian overlap vs. center-center distance")
    ax.set_xlabel("distance |a - b| (Angstrom)")
    ax.set_ylabel("overlap integral")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "overlap_vs_distance.png"), dpi=130)
    plt.close(fig)


def fig_pipeline() -> None:
    """A simple block diagram of the shape-comparison pipeline."""
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.axis("off")
    steps = [
        "SMILES /\nMol",
        "RDKit 3D\nembed",
        "Gaussian\nshape",
        "Overlap /\nTanimoto",
        "BFGS\nalign",
        "SDF for\nPyMOL",
    ]
    n = len(steps)
    for i, text in enumerate(steps):
        x = i / n
        box = plt.Rectangle((x + 0.01, 0.3), 0.9 / n - 0.02, 0.4,
                            facecolor="#dce6f2", edgecolor="#2b5aa0")
        ax.add_patch(box)
        ax.text(x + 0.45 / n, 0.5, text, ha="center", va="center", fontsize=9)
        if i < n - 1:
            ax.annotate("", xy=(x + 0.9 / n + 0.005, 0.5),
                        xytext=(x + 0.9 / n - 0.015, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#2b5aa0"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "pipeline.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    fig_gaussian_1d()
    fig_two_atom_density()
    fig_overlap_vs_distance()
    fig_pipeline()
    print(f"Figures written to {IMG_DIR}")


if __name__ == "__main__":
    main()
