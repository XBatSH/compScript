"""
make_figures.py -- Generate all figures for TUTORIAL.md.

Run from the project root:
    python docs/make_figures.py

Figures land in docs/figures/. Several of them are *computed with the
core code itself* (H2 dissociation curve, H2O MO diagram), so this
script doubles as an integration test.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# make "core" importable when running from the docs/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 130})


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ===================================================================== #
# Figure 1: Slater 1s orbital vs STO-3G contraction (hydrogen)
# ===================================================================== #
def fig_sto3g():
    r = np.linspace(0, 5, 400)
    zeta = 1.0
    slater = (zeta ** 3 / np.pi) ** 0.5 * np.exp(-zeta * r)

    # STO-3G hydrogen 1s: exponents & contraction coefficients
    exps = np.array([3.425250914, 0.6239137298, 0.1688554040])
    coefs = np.array([0.1543289673, 0.5353281423, 0.4446345422])
    norms = (2 * exps / np.pi) ** 0.75          # s-primitive normalization

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    total = np.zeros_like(r)
    for a, c, n in zip(exps, coefs, norms):
        g = c * n * np.exp(-a * r ** 2)
        total += g
        ax.plot(r, g, "--", lw=1.2,
                label=rf"primitive $\alpha$ = {a:.3f}")
    ax.plot(r, total, "b-", lw=2.2, label="STO-3G contraction (sum)")
    ax.plot(r, slater, "k-", lw=2.2, alpha=0.6,
            label=r"exact Slater $e^{-r}$")
    ax.set_xlabel("distance from nucleus  $r$ / Bohr")
    ax.set_ylabel(r"radial amplitude  $\phi(r)$")
    ax.set_title("STO-3G: three Gaussians imitate one Slater orbital")
    ax.annotate("Gaussians are flat at r = 0,\nSlater has a cusp",
                xy=(0.05, 0.55), xytext=(1.0, 0.48),
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.legend(fontsize=9)
    ax.set_xlim(0, 5)
    save(fig, "sto3g_contraction.png")


# ===================================================================== #
# Figure 2: Gaussian product theorem
# ===================================================================== #
def fig_gaussian_product():
    x = np.linspace(-3, 5, 500)
    a, A = 0.8, 0.0            # Gaussian 1: exponent, center
    b, B = 1.2, 2.0            # Gaussian 2
    gA = np.exp(-a * (x - A) ** 2)
    gB = np.exp(-b * (x - B) ** 2)
    p = a + b
    P = (a * A + b * B) / p    # new center: exponent-weighted average
    K = np.exp(-a * b / p * (A - B) ** 2)
    product = gA * gB

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, gA, "C0-", lw=2, label=rf"$g_A$: $\alpha$={a}, center A={A}")
    ax.plot(x, gB, "C1-", lw=2, label=rf"$g_B$: $\beta$={b}, center B={B}")
    ax.plot(x, product, "C3-", lw=2.5,
            label=r"product $g_A \cdot g_B$ (one Gaussian!)")
    ax.plot(x, K * np.exp(-p * (x - P) ** 2), "k:", lw=2,
            label=rf"$K\,e^{{-p(x-P)^2}}$, P={P:.2f}")
    for pos, txt in ((A, "A"), (B, "B"), (P, "P")):
        ax.axvline(pos, color="gray", lw=0.7, ls=":")
        ax.text(pos, 1.04, txt, ha="center", color="gray")
    ax.set_xlabel("x")
    ax.set_ylabel("amplitude")
    ax.set_title("Gaussian product theorem: the key to all integrals")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_ylim(0, 1.15)
    save(fig, "gaussian_product.png")


# ===================================================================== #
# Figure 3: Boys function
# ===================================================================== #
def fig_boys():
    from core.integrals import boys
    x = np.linspace(0, 20, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for n in range(4):
        ax.plot(x, [boys(n, xi) for xi in x], lw=2, label=rf"$F_{n}(x)$")
    ax.plot(x[1:], 0.5 * np.sqrt(np.pi / x[1:]), "k:", lw=1.5,
            label=r"asymptote $\frac{1}{2}\sqrt{\pi/x}$ of $F_0$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$F_n(x)$")
    ax.set_title("Boys function: every Coulomb integral ends here")
    ax.legend()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.05)
    save(fig, "boys_function.png")


# ===================================================================== #
# Figure 4: SCF flowchart
# ===================================================================== #
def fig_flowchart():
    fig, ax = plt.subplots(figsize=(6.8, 8.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)

    def box(x, y, w, h, text, fc="#e8f0fe", ec="#1a56db"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.12",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5)

    def arrow(x1, y1, x2, y2, text="", color="#333333"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=color, lw=1.4))
        if text:
            ax.text((x1 + x2) / 2 + 0.25, (y1 + y2) / 2, text,
                    fontsize=9, color=color, ha="left")

    box(5, 14.2, 6.6, 1.0, "molecule + basis set\n(geometry, charge; STO-3G)",
        fc="#fef3e2", ec="#b45309")
    box(5, 12.4, 6.6, 1.1,
        "compute integrals once:\n$S,\\ T,\\ V,\\ (\\mu\\nu|\\lambda\\sigma)$"
        "  and  $H^{core}=T+V$")
    box(5, 10.7, 6.6, 0.9, "orthogonalizer  $X = S^{-1/2}$")
    box(5, 9.2, 6.6, 0.9,
        "initial guess: diagonalize $H^{core}$  $\\rightarrow$  $C$")
    box(5, 7.6, 6.6, 1.0,
        "density  $D_{\\mu\\nu} = 2\\sum_i^{occ} C_{\\mu i}C_{\\nu i}$",
        fc="#e7f6ec", ec="#15803d")
    box(5, 5.9, 6.6, 1.1,
        "Fock matrix  $F = H^{core} + J(D) - \\frac{1}{2}K(D)$",
        fc="#e7f6ec", ec="#15803d")
    box(5, 4.2, 6.6, 1.1,
        "solve $FC = SC\\varepsilon$ :\n"
        "$F' = X^T F X$, diag., $C = XC'$",
        fc="#e7f6ec", ec="#15803d")
    box(5, 2.5, 6.6, 0.9,
        "converged?  $|\\Delta E| < 10^{-8}$ and $|\\Delta D| < 10^{-6}$",
        fc="#fdeaea", ec="#b91c1c")
    box(5, 0.8, 6.6, 0.9,
        "$E_{tot} = \\frac{1}{2}\\sum D(H^{core}+F) + E_{nn}$\n done!",
        fc="#fef3e2", ec="#b45309")

    arrow(5, 13.7, 5, 12.95)
    arrow(5, 11.85, 5, 11.15)
    arrow(5, 10.25, 5, 9.65)
    arrow(5, 8.75, 5, 8.1)
    arrow(5, 7.1, 5, 6.45)
    arrow(5, 5.35, 5, 4.75)
    arrow(5, 3.65, 5, 2.95)
    arrow(5, 2.05, 5, 1.25, "yes", "#15803d")
    # loop back: no
    arrow(8.3, 2.5, 8.3, 7.6, "", "#b91c1c")
    ax.plot([6.55, 8.3], [2.5, 2.5], color="#b91c1c", lw=1.4)
    ax.plot([8.3, 6.55], [7.6, 7.6], color="#b91c1c", lw=1.4)
    ax.text(8.55, 5.0, "no: new density,\niterate", fontsize=9,
            color="#b91c1c", ha="left")
    ax.set_title("The self-consistent field (SCF) loop", fontsize=13)
    save(fig, "scf_flowchart.png")


# ===================================================================== #
# Figure 5: H2 dissociation curve computed WITH core
# ===================================================================== #
def fig_h2_curve():
    from core import Molecule, rhf
    distances = np.linspace(0.8, 5.0, 25)          # Bohr
    energies = []
    for d in distances:
        mol = Molecule.from_atoms([("H", (0, 0, 0)), ("H", (0, 0, d))],
                                  unit="bohr", name=f"H2 d={d:.2f}")
        energies.append(rhf(mol, verbose=False)["energy"])
    energies = np.array(energies)
    i_min = energies.argmin()

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(distances, energies, "o-", color="C0", lw=2, ms=4)
    ax.axhline(-1.0, color="gray", ls=":", lw=1,
               label="2 isolated H atoms (exact, $-1$ Ha)")
    ax.plot(distances[i_min], energies[i_min], "r*", ms=15,
            label=f"minimum: R = {distances[i_min]:.2f} Bohr, "
                  f"E = {energies[i_min]:.4f} Ha")
    ax.set_xlabel("H-H distance / Bohr")
    ax.set_ylabel("RHF/STO-3G total energy / Hartree")
    ax.set_title("H$_2$ potential energy curve (computed with core)")
    ax.annotate("RHF dissociates incorrectly:\n"
                "curve goes far above $-1$ Ha\n(the famous RHF failure)",
                xy=(4.6, energies[-1] - 0.005), xytext=(2.2, -0.78),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "h2_dissociation.png")


# ===================================================================== #
# Figure 6: H2O molecular-orbital diagram computed WITH core
# ===================================================================== #
def fig_h2o_mo():
    from core import Molecule, rhf
    mol = Molecule.from_atoms(
        [("O", (0.0, 0.0, 0.1173470)),
         ("H", (0.0, 0.7572260, -0.4693879)),
         ("H", (0.0, -0.7572260, -0.4693879))],
        unit="angstrom", name="H2O")
    res = rhf(mol, verbose=False)
    eps = res["mo_energies"]
    n_occ = mol.n_electrons // 2
    labels = ["1a$_1$ (O 1s core)", "2a$_1$", "1b$_2$", "3a$_1$",
              "1b$_1$ (lone pair)", "4a$_1^*$", "2b$_2^*$"]

    fig, ax = plt.subplots(figsize=(5.6, 6.0))
    ax.set_ylim(-2.0, 1.1)
    # stagger labels of near-degenerate levels so text does not overlap
    text_y = list(eps)
    for i in range(1, len(text_y)):
        if 0 <= text_y[i] - text_y[i - 1] < 0.13:
            text_y[i] = text_y[i - 1] + 0.13
    for i, e in enumerate(eps):
        if e < -2.0:                    # deep core level: off scale, note only
            continue
        occ = i < n_occ
        color = "C0" if occ else "C3"
        ax.hlines(e, 0.28, 0.62, color=color, lw=3)
        ax.text(0.65, text_y[i], f"{labels[i]}   {e:.3f} Ha",
                va="center", fontsize=9.5, color=color)
        if occ:  # draw the two electrons as up/down arrows
            ax.annotate("", xy=(0.41, e + 0.06), xytext=(0.41, e - 0.06),
                        arrowprops=dict(arrowstyle="-|>", color="k", lw=1))
            ax.annotate("", xy=(0.49, e - 0.06), xytext=(0.49, e + 0.06),
                        arrowprops=dict(arrowstyle="-|>", color="k", lw=1))
    gap = (eps[n_occ] - eps[n_occ - 1]) * 27.211386
    ax.annotate("", xy=(0.2, eps[n_occ]), xytext=(0.2, eps[n_occ - 1]),
                arrowprops=dict(arrowstyle="<->", color="green"))
    ax.text(0.03, (eps[n_occ] + eps[n_occ - 1]) / 2,
            f"gap\n{gap:.1f} eV", fontsize=9, color="green", va="center")
    # note about the deep O 1s level that is far below the plot range
    ax.text(0.45, -1.9, rf"(1a$_1$ at {eps[0]:.2f} Ha, off scale)",
            ha="center", fontsize=9, color="gray")
    ax.set_xlim(0, 1.6)
    ax.set_xticks([])
    ax.set_ylabel("orbital energy / Hartree")
    ax.set_title("H$_2$O molecular orbitals, RHF/STO-3G\n"
                 "(blue = occupied, red = virtual)")
    save(fig, "h2o_mo_diagram.png")


if __name__ == "__main__":
    fig_sto3g()
    fig_gaussian_product()
    fig_boys()
    fig_flowchart()
    fig_h2_curve()
    fig_h2o_mo()
    print("all figures done.")
