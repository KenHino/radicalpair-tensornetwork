"""C-D hopping model by vMPDO"""
# To understand code, see vmpdo.py at first, which is a single radical pair solved by vMPDO
"""
```
>>> uv run crypyo-aiso-twopairs.py 1e-01 0 
```
"""
import json
import sys

import numpy as np
import radicalpy as rp
from pympo import (
    AssignManager,
    OpSite,
    SumOfProducts,
)
from sympy import Symbol

import pytdscf

from pytdscf import BasInfo, Exciton, Model, Simulator, units
from pytdscf.dvr_operator_cls import TensorOperator
from pytdscf.hamiltonian_cls import TensorHamiltonian
from pytdscf.util import read_nc


cutoff = float(sys.argv[1]) # in mT
angle = int(sys.argv[2])    # k/8 pi
assert angle in [0, 1, 2, 3, 4, 5, 6, 7]
m = 128 # 1024  # Bond dimension
B0 = 0.050 # or 5.0 in mT

print(f"{pytdscf.__version__=}")

def get_mol_from_json(file, name):
    data = json.load(open(file, "r"))["data"]
    nuclei = [
        rp.data.Nucleus.fromisotope(val["element"], val["hfc"])
        for key, val in data.items()
    ]
    molecule = rp.simulation.Molecule(name, nuclei)
    return molecule


flavin = get_mol_from_json("./coordinate/flavin-wb97xd-lab.json", "flavin")
trpC = get_mol_from_json("./coordinate/TrpC-wb97xd-lab.json", "trpC")
trpD = get_mol_from_json("./coordinate/TrpD-wb97xd-lab.json", "trpD")

r_12 = np.array([9.480, -13.675, 5.388]) * 1e-10
r_13 = np.array([8.980, -18.684, 4.159]) * 1e-10
d_12 = np.linalg.norm(r_12)
d_13 = np.linalg.norm(r_13)

# p25 in SI of Nature 2021
kr_dict = {(0, 1): 1.7e07, (0, 2): 0.0}  # recombination
kf_dict = {(0, 1): 5.7e06, (0, 2): 1.0e07}  # fuluorecence

# table S3
# BC 5.0e+10
# CB 9.0e+05
kDC = 1.3e10  # 1.3e+10
kCD = 1.5e10  # 1.5e+10


def sort_nuclei(mol, reverse=False, central=False):
    # Sort nuclei
    hf_abs = []
    nucs = mol.nuclei
    for nuc in nucs:
        eigvals = np.linalg.eigvals(nuc.hfc.anisotropic)
        hf_abs.append(np.mean(np.abs(eigvals)))
    if reverse:
        mol.nuclei = [nucs[i] for i in np.argsort(hf_abs)[::-1]]
    elif central:
        """
        Central sort indices
        Example: n=6, [2, 3, 0, 1, 4, 5]
        This will make the strength [1, 3, 2, 4, 5, 6] to [2, 3, 6, 5, 4, 1]
        """
        left = 0
        right = len(hf_abs) - 1
        sorted_indices = np.argsort(hf_abs)
        # ascending order of strength
        nucs_sorted = [nucs[i] for i in sorted_indices]
        new_sorted_indices = []
        left_update, right_update = 2, 1
        while left <= right:
            print(f"{left=}, {right=}, {left_update=}, {right_update=}")
            match (left_update, right_update):
                case (2, 0):
                    # insert right
                    new_sorted_indices.append(right)
                    right -= 1
                    left_update, right_update = 2, 1
                case (2, 1):
                    # insert right
                    new_sorted_indices.append(right)
                    right -= 1
                    left_update, right_update = 0, 2
                case (0, 2):
                    # insert left
                    new_sorted_indices.append(left)
                    left += 1
                    left_update, right_update = 1, 2
                case (1, 2):
                    # insert left
                    new_sorted_indices.append(left)
                    left += 1
                    left_update, right_update = 2, 0
                case _:
                    raise ValueError
        print(new_sorted_indices)
        assert len(new_sorted_indices) == len(nucs_sorted)
        mol.nuclei = [nucs_sorted[i] for i in np.argsort(new_sorted_indices)]
    else:
        mol.nuclei = [nucs[i] for i in np.argsort(hf_abs)]


def truncate_nuclei(mol, cutoff=cutoff):
    nucs = []
    for nuc in mol.nuclei:
        eigvals = np.linalg.eigvals(nuc.hfc.anisotropic)
        if np.mean(np.abs(eigvals)) > cutoff:
            nucs.append(nuc)
    mol.nuclei = nucs

def fuse_nuclei(mol):
    nucs = []
    sort_nuclei(mol, reverse=False)
    nuc_pend = []
    for nuc in mol.nuclei:
        if nuc_pend:
            if (nuc_pend[-1].multiplicity == nuc.multiplicity) and (
                np.allclose(nuc_pend[-1].hfc.anisotropic, nuc.hfc.anisotropic)
            ):
                nuc_pend.append(nuc)
            else:
                if len(nuc_pend) == 1:
                    nucs.append(nuc_pend[0])
                else:
                    print(f"Fusing {nuc_pend}")
                    nucs.append(rp.data.FuseNucleus.from_nuclei(nuc_pend))
                nuc_pend = [nuc]
        else:
            nuc_pend.append(nuc)
    if nuc_pend:
        if len(nuc_pend) == 1:
            nucs.append(nuc_pend[0])
        else:
            print(f"Fusing {nuc_pend}")
            nucs.append(rp.data.FuseNucleus.from_nuclei(nuc_pend))
    mol.nuclei = nucs

fuse_nuclei(flavin) # flavin has methyl hydrogens

truncate_nuclei(flavin)
truncate_nuclei(trpC)
truncate_nuclei(trpD)

sort_nuclei(trpC, reverse=False)
sort_nuclei(flavin, central=True)
sort_nuclei(trpD, reverse=True)

print(trpC)
print(flavin)
print(trpD)

# MPS is aligned to
# (TrpC) - (flavin/2) - (electron) - (flavin/2) - (TrpD)

sim_C = rp.simulation.LiouvilleSimulation([flavin, trpC])
sim_D = rp.simulation.LiouvilleSimulation([flavin, trpD])

ele_site = len(sim_C.molecules[1].nuclei) + len(sim_C.molecules[0].nuclei) // 2
bgn_C_site = 0
bgn_D_site = len(sim_C.molecules[0].nuclei) + len(sim_C.molecules[1].nuclei) + 1


B = np.array((np.sin(angle * np.pi / 8), 0.0, np.cos(angle * np.pi / 8))) * B0

A = {}
for i in range(3):
    if i <= 1:
        mol = sim_C.molecules[i]
    elif i == 2:
        mol = sim_D.molecules[1]
    for j, nuc in enumerate(mol.nuclei):
        A[(i, j)] = nuc.hfc.anisotropic

print(f"{ele_site=}")
print(f"{A=}")

# taken from table S4 of https://pubs.acs.org/doi/10.1021/jacs.4c14037
J_dict = {
    (0, 1): 0.011,
    (0, 2): 0.001,
}
print(rp.estimations.dipolar_interaction_isotropic(d_12))
D_scalar_dict = {
    (0, 1): rp.estimations.dipolar_interaction_isotropic(d_12),  # 0.43
    (0, 2): rp.estimations.dipolar_interaction_isotropic(d_13),  # 0.28
}

D_dict = {}
for key, D in D_scalar_dict.items():
    assert D <= 0
    match key:
        case (0, 1):
            rx, ry, rz = r_12 / d_12
        case (0, 2):
            rx, ry, rz = r_13 / d_13
        case _:
            raise ValueError

    Dtensor = np.eye(3) - 3.0 * np.array(
        [
            [rx * rx, rx * ry, rx * rz],
            [ry * rx, ry * ry, ry * rz],
            [rz * rx, rz * ry, rz * rz],
        ]
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(Dtensor), np.array([-2, 1, 1])
    )

    D_dict[key] = 2 / 3 * (-D) * Dtensor
    print(
        f"{key=}: D-2J={D_dict[key]  - 2 * J_dict[(0, 1)] * np.eye(3)}"
    )


# Clear nuclei temporally
_nuclei_tmp0 = sim_C.molecules[0].nuclei
_nuclei_tmp1 = sim_C.molecules[1].nuclei
_nuclei_tmp2 = sim_D.molecules[1].nuclei

sim_C.molecules[0].nuclei = []
sim_C.molecules[1].nuclei = []
sim_D.molecules[0].nuclei = []
sim_D.molecules[1].nuclei = []


# for Singlet-Triplet basis
def fill_block(block, slice1, slice2, ref):
    ref = ref.copy()
    ref[slice1, slice2] = block
    return ref

"""
In Haberkorn formalism, electronic spin sites are defined as follows:
0: C: T+
1: C: T0
2: C: S
3: C: T-
4: D: T+
5: D: T0
6: D: S
7: D: T-
While Lindblad formalism, electronic spin sites are defined as follows:
0: C: T+
1: C: T0
2: C: S
3: C: T-
4: C: Singlet product
5: C: Triplet product
6: D: T+
7: D: T0
8: D: S
9: D: T-
10: D: Singlet product
11: D: Triplet product
"""

# Haberkorn formalism
# zeros = np.zeros((8, 8), dtype=np.complex128)
# Lindblad formalism
zeros = np.zeros((12, 12), dtype=np.complex128)

sx_1 = fill_block(sim_C.spin_operator(0, "x"), slice(0, 4), slice(0, 4), zeros)
sy_1 = fill_block(sim_C.spin_operator(0, "y"), slice(0, 4), slice(0, 4), zeros)
sz_1 = fill_block(sim_C.spin_operator(0, "z"), slice(0, 4), slice(0, 4), zeros)

sx_2 = fill_block(sim_C.spin_operator(1, "x"), slice(0, 4), slice(0, 4), zeros)
sy_2 = fill_block(sim_C.spin_operator(1, "y"), slice(0, 4), slice(0, 4), zeros)
sz_2 = fill_block(sim_C.spin_operator(1, "z"), slice(0, 4), slice(0, 4), zeros)

# Haberkorn formalism
# sx_1 = fill_block(sim_D.spin_operator(0, "x"), slice(4, 8), slice(4, 8), sx_1)
# sy_1 = fill_block(sim_D.spin_operator(0, "y"), slice(4, 8), slice(4, 8), sy_1)
# sz_1 = fill_block(sim_D.spin_operator(0, "z"), slice(4, 8), slice(4, 8), sz_1)
# Lindblad formalism
sx_1 = fill_block(sim_D.spin_operator(0, "x"), slice(6, 10), slice(6, 10), sx_1)
sy_1 = fill_block(sim_D.spin_operator(0, "y"), slice(6, 10), slice(6, 10), sy_1)
sz_1 = fill_block(sim_D.spin_operator(0, "z"), slice(6, 10), slice(6, 10), sz_1)

# Haberkorn formalism
# sx_3 = fill_block(sim_D.spin_operator(1, "x"), slice(4, 8), slice(4, 8), zeros)
# sy_3 = fill_block(sim_D.spin_operator(1, "y"), slice(4, 8), slice(4, 8), zeros)
# sz_3 = fill_block(sim_D.spin_operator(1, "z"), slice(4, 8), slice(4, 8), zeros)
# Lindblad formalism
sx_3 = fill_block(sim_D.spin_operator(1, "x"), slice(6, 10), slice(6, 10), zeros)
sy_3 = fill_block(sim_D.spin_operator(1, "y"), slice(6, 10), slice(6, 10), zeros)
sz_3 = fill_block(sim_D.spin_operator(1, "z"), slice(6, 10), slice(6, 10), zeros)


_Qs = sim_C.projection_operator(rp.simulation.State.SINGLET)
_Qt = sim_C.projection_operator(rp.simulation.State.TRIPLET)
_E = np.eye(4)
Qs_C = fill_block(_Qs, slice(0, 4), slice(0, 4), zeros)
Qt_C = fill_block(_Qt, slice(0, 4), slice(0, 4), zeros)
E_C = fill_block(_E, slice(0, 4), slice(0, 4), zeros)

# Haberkorn formalism
# Qs_D = fill_block(_Qs, slice(4, 8), slice(4, 8), zeros)
# Qt_D = fill_block(_Qt, slice(4, 8), slice(4, 8), zeros)
# E_D = fill_block(_E, slice(4, 8), slice(4, 8), zeros)
# Lindblad formalism
Qs_D = fill_block(_Qs, slice(6, 10), slice(6, 10), zeros)
Qt_D = fill_block(_Qt, slice(6, 10), slice(6, 10), zeros)
E_D = fill_block(_E, slice(6, 10), slice(6, 10), zeros)

Qs = Qs_C + Qs_D
Qt = Qt_C + Qt_D
E = E_C + E_D
# Revert nuclei
sim_C.molecules[0].nuclei = _nuclei_tmp0
sim_C.molecules[1].nuclei = _nuclei_tmp1
sim_D.molecules[0].nuclei = _nuclei_tmp0
sim_D.molecules[1].nuclei = _nuclei_tmp2

print(sim_C, sim_D)

SCALE = 1.0e-09

g_ele_sym = [Symbol(r"\gamma_e^{(" + f"{i + 1}" + ")}") for i in range(3)]
g_nuc_sym = {}
for i in range(3):
    if i <= 1:
        for j in range(len(sim_C.molecules[i].nuclei)):
            g_nuc_sym[(i, j)] = Symbol(
                r"\gamma_n^{" + f"{(i + 1, j + 1)}" + "}"
            )
    elif i == 2:
        for j in range(len(sim_D.molecules[1].nuclei)):
            g_nuc_sym[(i, j)] = Symbol(
                r"\gamma_n^{" + f"{(i + 1, j + 1)}" + "}"
            )


subs = {}
for i, ge in enumerate(g_ele_sym):
    if i <= 1:
        subs[ge] = sim_C.radicals[i].gamma_mT
    elif i == 2:
        subs[ge] = sim_D.radicals[1].gamma_mT
for (i, j), gn in g_nuc_sym.items():
    if i <= 1:
        subs[gn] = sim_C.molecules[i].nuclei[j].gamma_mT
    elif i == 2:
        subs[gn] = sim_D.molecules[1].nuclei[j].gamma_mT


def get_OE(op):
    """
    Ot ⊗ 𝟙
    """
    return np.kron(op.T, np.eye(op.shape[0], dtype=op.dtype))


def get_EO(op):
    """
    𝟙 ⊗ O
    """
    return np.kron(np.eye(op.shape[0], dtype=op.dtype), op)


def get_LL(op):
    """
    L* ⊗ L
    """
    return np.kron(op.conjugate(), op)


SxE_ops = []
SyE_ops = []
SzE_ops = []
ESx_ops = []
ESy_ops = []
ESz_ops = []

S1S2E_op = OpSite(
    r"(\hat{S}_1\cdot\hat{S}_2)^\ast ⊗ 𝟙",
    ele_site,
    value=get_OE(sx_1 @ sx_2 + sy_1 @ sy_2 + sz_1 @ sz_2),
)
ES1S2_op = OpSite(
    r"𝟙 ⊗ (\hat{S}_1\cdot\hat{S}_2)",
    ele_site,
    value=get_EO(sx_1 @ sx_2 + sy_1 @ sy_2 + sz_1 @ sz_2),
)
S1S3E_op = OpSite(
    r"(\hat{S}_1\cdot\hat{S}_3)^\ast ⊗ 𝟙",
    ele_site,
    value=get_OE(sx_1 @ sx_3 + sy_1 @ sy_3 + sz_1 @ sz_3),
)
ES1S3_op = OpSite(
    r"𝟙 ⊗ (\hat{S}_1\cdot\hat{S}_3)",
    ele_site,
    value=get_EO(sx_1 @ sx_3 + sy_1 @ sy_3 + sz_1 @ sz_3),
)
EE_op = OpSite(
    r"\hat{E} ⊗ \hat{E}", ele_site, value=get_OE(np.eye(sx_1.shape[0]))
)

QsE_op = OpSite(r"\hat{Q}_S ⊗ 𝟙", ele_site, value=get_OE(Qs))
EQs_op = OpSite(r"𝟙 ⊗ \hat{Q}_S", ele_site, value=get_EO(Qs))
QtE_op = OpSite(r"\hat{Q}_T ⊗ 𝟙", ele_site, value=get_OE(Qt))
EQt_op = OpSite(r"𝟙 ⊗ \hat{Q}_T", ele_site, value=get_EO(Qt))

Qs_CE_op = OpSite(r"\hat{Q}_S^{(0,1)} ⊗ 𝟙", ele_site, value=get_OE(Qs_C))
EQs_C_op = OpSite(r"𝟙 ⊗ \hat{Q}_S^{(0,1)}", ele_site, value=get_EO(Qs_C))
Qs_DE_op = OpSite(r"\hat{Q}_S^{(0,2)} ⊗ 𝟙", ele_site, value=get_OE(Qs_D))
EQs_D_op = OpSite(r"𝟙 ⊗ \hat{Q}_S^{(0,2)}", ele_site, value=get_EO(Qs_D))

EE_C_op = OpSite(r"E^{(0,1)} ⊗ 𝟙", ele_site, value=get_OE(E_C))
E_CE_op = OpSite(r"𝟙 ⊗ E^{(0,1)}", ele_site, value=get_EO(E_C))
EE_D_op = OpSite(r"E^{(0,2)} ⊗ 𝟙", ele_site, value=get_OE(E_D))
E_DE_op = OpSite(r"𝟙 ⊗ E^{(0,2)}", ele_site, value=get_EO(E_D))


SxE_ops.append(OpSite(r"\hat{S}_x^{(1)\ast} ⊗ 𝟙", ele_site, value=get_OE(sx_1)))
SxE_ops.append(OpSite(r"\hat{S}_x^{(2)\ast} ⊗ 𝟙", ele_site, value=get_OE(sx_2)))
SxE_ops.append(OpSite(r"\hat{S}_x^{(3)\ast} ⊗ 𝟙", ele_site, value=get_OE(sx_3)))

SyE_ops.append(OpSite(r"\hat{S}_y^{(1)\ast} ⊗ 𝟙", ele_site, value=get_OE(sy_1)))
SyE_ops.append(OpSite(r"\hat{S}_y^{(2)\ast} ⊗ 𝟙", ele_site, value=get_OE(sy_2)))
SyE_ops.append(OpSite(r"\hat{S}_y^{(3)\ast} ⊗ 𝟙", ele_site, value=get_OE(sy_3)))

SzE_ops.append(OpSite(r"\hat{S}_z^{(1)\ast} ⊗ 𝟙", ele_site, value=get_OE(sz_1)))
SzE_ops.append(OpSite(r"\hat{S}_z^{(2)\ast} ⊗ 𝟙", ele_site, value=get_OE(sz_2)))
SzE_ops.append(OpSite(r"\hat{S}_z^{(3)\ast} ⊗ 𝟙", ele_site, value=get_OE(sz_3)))

ESx_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_x^{(1)}", ele_site, value=get_EO(sx_1)))
ESx_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_x^{(2)}", ele_site, value=get_EO(sx_2)))
ESx_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_x^{(3)}", ele_site, value=get_EO(sx_3)))

ESy_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_y^{(1)}", ele_site, value=get_EO(sy_1)))
ESy_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_y^{(2)}", ele_site, value=get_EO(sy_2)))
ESy_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_y^{(3)}", ele_site, value=get_EO(sy_3)))

ESz_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_z^{(1)}", ele_site, value=get_EO(sz_1)))
ESz_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_z^{(2)}", ele_site, value=get_EO(sz_2)))
ESz_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_z^{(3)}", ele_site, value=get_EO(sz_3)))

SrE_ops = [SxE_ops, SyE_ops, SzE_ops]
ESr_ops = [ESx_ops, ESy_ops, ESz_ops]


# Define nuclear spin operators
IxE_ops = {}
IyE_ops = {}
IzE_ops = {}
EIx_ops = {}
EIy_ops = {}
EIz_ops = {}

for i in range(3):
    if i == 0:
        bgn = len(sim_C.molecules[1].nuclei)
        mol = sim_C.molecules[0]
    elif i == 1:
        bgn = bgn_C_site
        mol = sim_C.molecules[1]
    elif i == 2:
        bgn = bgn_D_site
        mol = sim_D.molecules[1]

    for j, nuc in enumerate(mol.nuclei):
        if i == 0 and j == len(mol.nuclei) // 2:
            bgn += 1
        val = nuc.pauli["x"]
        op_name_x1 = r"\hat{I}_x^{" + f"{(i + 1, j + 1)}" + r"\ast} ⊗ 𝟙"
        op_name_1x = r"𝟙 ⊗ \hat{I}_x^{" + f"{(i + 1, j + 1)}" + "}"
        IxE_ops[(i, j)] = OpSite(op_name_x1, bgn + j, value=get_OE(val))
        EIx_ops[(i, j)] = OpSite(op_name_1x, bgn + j, value=get_EO(val))
        val = nuc.pauli["y"]
        op_name_y1 = op_name_x1.replace("_x", "_y")
        op_name_1y = op_name_1x.replace("_x", "_y")
        IyE_ops[(i, j)] = OpSite(op_name_y1, bgn + j, value=get_OE(val))
        EIy_ops[(i, j)] = OpSite(op_name_1y, bgn + j, value=get_EO(val))
        val = nuc.pauli["z"]
        op_name_z1 = op_name_y1.replace("_y", "_z")
        op_name_1z = op_name_1y.replace("_y", "_z")
        IzE_ops[(i, j)] = OpSite(op_name_z1, bgn + j, value=get_OE(val))
        EIz_ops[(i, j)] = OpSite(op_name_1z, bgn + j, value=get_EO(val))
        print(f"{bgn+j=}, {ele_site=}, {i=}, {j=}")

IrE_ops = [IxE_ops, IyE_ops, IzE_ops]
EIr_ops = [EIx_ops, EIy_ops, EIz_ops]


# ## Hyperfine coupling Hamiltonian
hyperfine = SumOfProducts()
xyz = "xyz"
for i in range(3):
    if i == 0:
        mol = sim_C.molecules[0]
    elif i == 1:
        mol = sim_C.molecules[1]
    elif i == 2:
        mol = sim_D.molecules[1]
    for j in range(len(mol.nuclei)):
        for k, (SrE_op, ESr_op) in enumerate(
            zip(SrE_ops, ESr_ops, strict=True)
        ):
            for l, (IrE_op, EIr_op) in enumerate(
                zip(IrE_ops, EIr_ops, strict=True)
            ):
                if A[(i, j)][k, l] == 0.0:
                    continue
                Asym = Symbol(
                    "A^{"
                    + f"{(i + 1, j + 1)}"
                    + "}_{"
                    + f"{xyz[k]}"
                    + f"{xyz[l]}"
                    + "}"
                )
                subs[Asym] = A[(i, j)][k, l].item() * SCALE
                hyperfine += Asym * abs(g_ele_sym[i]) * ESr_op[i] * EIr_op[(i, j)]
                hyperfine -= Asym * abs(g_ele_sym[i]) * SrE_op[i] * IrE_op[(i, j)]
hyperfine = hyperfine.simplify()


#  Zeeman Hamiltonian
zeeman = SumOfProducts()
xyz = "xyz"
for k, (SrE_op, ESr_op, IrE_op, EIr_op) in enumerate(
    zip(SrE_ops, ESr_ops, IrE_ops, EIr_ops, strict=True)
):
    if B[k] == 0.0:
        continue
    r = xyz[k]
    Br = Symbol(f"B_{r}")
    subs[Br] = B[k] * SCALE
    for i in range(3):
        zeeman += -Br * g_ele_sym[i] * ESr_op[i]
        zeeman -= -Br * g_ele_sym[i] * SrE_op[i]
        if i == 0:
            mol = sim_C.molecules[0]
        elif i == 1:
            mol = sim_C.molecules[1]
        elif i == 2:
            mol = sim_D.molecules[1]
        for j in range(len(mol.nuclei)):
            zeeman += -Br * g_nuc_sym[(i, j)] * EIr_op[(i, j)]
            zeeman -= -Br * g_nuc_sym[(i, j)] * IrE_op[(i, j)]

zeeman = zeeman.simplify()


# Exchange Hamiltonian
exchange = SumOfProducts()
for key, J in J_dict.items():
    if J != 0.0:
        Jsym = Symbol(r"J_{" + f"{key}" + "}")
        subs[Jsym] = J * SCALE
        if key == (0, 1):
            exchange += -Jsym * abs(g_ele_sym[1]) * (2 * ES1S2_op + 0.5 * EE_op)
            exchange -= -Jsym * abs(g_ele_sym[1]) * (2 * S1S2E_op + 0.5 * EE_op)
        elif key == (0, 2):
            exchange += -Jsym * abs(g_ele_sym[2]) * (2 * ES1S3_op + 0.5 * EE_op)
            exchange -= -Jsym * abs(g_ele_sym[2]) * (2 * S1S3E_op + 0.5 * EE_op)
exchange = exchange.simplify()


# Define Dipolar Hamiltonian
dipolar = SumOfProducts()
for key, D in D_dict.items():
    for k in range(3):
        for l in range(3):
            if D[k, l] == 0.0:
                continue
            else:
                Dsym = Symbol(
                    "D_{" + f"{xyz[k]}" + f"{xyz[l]}" + "}^{" + f"{key}" + "}"
                )
                subs[Dsym] = D[k, l] * SCALE
                dipolar += Dsym * abs(g_ele_sym[2]) * ESr_ops[k][key[0]] * ESr_ops[l][key[1]]
                dipolar -= Dsym * abs(g_ele_sym[2]) * SrE_ops[k][key[0]] * SrE_ops[l][key[1]]
dipolar = dipolar.simplify()

# Define Haberkorn relaxation
# haberkorn = SumOfProducts()
# for key, kr in kr_dict.items():
#     if kr == 0.0:
#         continue
#     krsym = Symbol("k_r^{" + f"{key}" + "}")
#     subs[krsym] = kr * SCALE
#     if key == (0, 1):
#         haberkorn -= 1.0j * krsym / 2 * (Qs_CE_op + EQs_C_op)
#     elif key == (0, 2):
#         haberkorn -= 1.0j * krsym / 2 * (Qs_DE_op + EQs_D_op)
# 
# for key, kf in kf_dict.items():
#     if kf == 0.0:
#         continue
#     kfsym = Symbol("k_f^{" + f"{key}" + "}")
#     subs[kfsym] = kf * SCALE
#     if key == (0, 1):
#         haberkorn -= 1.0j * kfsym / 2 * (EE_C_op + E_CE_op)
#     elif key == (0, 2):
#         haberkorn -= 1.0j * kfsym / 2 * (EE_D_op + E_DE_op)
# haberkorn = haberkorn.simplify()


# Define Lindblad Jump
#
# $$
# \mathcal{D}[\rho]=L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho\}
# $$
#
# is linearised as
#
# $$
# L^\ast \otimes L - \frac{1}{2}(I\otimes L^\dagger L + (L^\dagger L)^\top \otimes I)
# $$
#
# we shall consider jump such as
#
# $$
# L = \sqrt{k_{C \leftarrow D}} \ket{C}\bra{D}
# $$


lindblad = SumOfProducts()
if kDC != 0.0:
    # Haberkorn formalism
    # L = np.zeros((8, 8))
    # L[0:4, 4:8] = np.eye(4)
    # Lindblad formalism
    L = np.zeros((12, 12))
    # Haberkorn formalism
    # L[0:4, 4:8] = np.eye(4)
    # Lindblad formalism
    L[0:4, 6:10] = np.eye(4)
    # 0 0 0 0 0 0 1 0 0 0 0 0
    # 0 0 0 0 0 0 0 1 0 0 0 0
    # 0 0 0 0 0 0 0 0 1 0 0 0
    # 0 0 0 0 0 0 0 0 0 1 0 0
    # 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0 0 0
    LdagL = L.conjugate().T @ L
    eye = np.eye(L.shape[0])
    val_DC = np.kron(L.conjugate(), L) - 0.5 * (
        np.kron(eye, LdagL) + np.kron(LdagL.T, eye)
    )
    name = r"[L^\ast_{DC} \otimes L_{DC} - \frac{1}{2}(I\otimes L^\dagger_{DC} L_{DC} + (L^\dagger_{DC} L_{DC})^\top \otimes I)]"
    op = OpSite(name, ele_site, value=val_DC)
    kDCsym = Symbol("k_{DC}")
    subs[kDCsym] = kDC * SCALE
    lindblad += 1.0j * kDCsym * op
    A = np.kron(L.conjugate(), L)
    nonzero = np.nonzero(A)
    nonzero_row = nonzero[0]
    nonzero_col = nonzero[1]

if kCD != 0.0:
    # Haberkorn formalism
    # L = np.zeros((8, 8))
    # Lindblad formalism
    L = np.zeros((12, 12))
    # Haberkorn formalism
    # L[4:8, 0:4] = np.eye(4)
    # Lindblad formalism
    L[6:10, 0:4] = np.eye(4)
    LdagL = L.conjugate().T @ L
    eye = np.eye(L.shape[0])
    val_CD = np.kron(L.conjugate(), L) - 0.5 * (
        np.kron(eye, LdagL) + np.kron(LdagL.T, eye)
    )
    name = r"[L^\ast_{CD} \otimes L_{CD} - \frac{1}{2}(I\otimes L^\dagger_{CD} L_{CD} + (L^\dagger_{CD} L_{CD})^\top \otimes I)]"
    op = OpSite(name, ele_site, value=val_CD)
    kCDsym = Symbol("k_{CD}")
    subs[kCDsym] = kCD * SCALE
    lindblad += 1.0j * kCDsym * op

# For Lindblad formalism rather than Haberkorn formalism
kSC = (kr_dict[(0, 1)] + kf_dict[(0, 1)])
kTC = (kf_dict[(0, 1)])
kSD = (kr_dict[(0, 2)] + kf_dict[(0, 2)])
kTD = (kf_dict[(0, 2)])

if kSC != 0.0:
    L = np.zeros((12, 12))
    L[4, 2] = 1.0 # S -> singlet product
    LdagL = L.conjugate().T @ L
    eye = np.eye(L.shape[0])
    val_SC = np.kron(L.conjugate(), L) - 0.5 * (
        np.kron(eye, LdagL) + np.kron(LdagL.T, eye)
    )
    name = r"[L^\ast_{SC} \otimes L_{SC} - \frac{1}{2}(I\otimes L^\dagger_{SC} L_{SC} + (L^\dagger_{SC} L_{SC})^\top \otimes I)]"
    op = OpSite(name, ele_site, value=val_SC)
    kSCsym = Symbol("k_{SC}")
    subs[kSCsym] = kSC * SCALE
    lindblad += 1.0j * kSCsym * op

if kTC != 0.0:
    L = np.zeros((12, 12))
    L[5, 0] = 1.0 # T+ -> triplet product
    L[5, 1] = 1.0 # T0 -> triplet product
    L[5, 3] = 1.0 # T- -> triplet product
    LdagL = L.conjugate().T @ L
    eye = np.eye(L.shape[0])
    val_TC = np.kron(L.conjugate(), L) - 0.5 * (
        np.kron(eye, LdagL) + np.kron(LdagL.T, eye)
    )
    name = r"[L^\ast_{TC} \otimes L_{TC} - \frac{1}{2}(I\otimes L^\dagger_{TC} L_{TC} + (L^\dagger_{TC} L_{TC})^\top \otimes I)]"
    op = OpSite(name, ele_site, value=val_TC)
    kTCsym = Symbol("k_{TC}")
    subs[kTCsym] = kTC * SCALE
    lindblad += 1.0j * kTCsym * op

if kSD != 0.0:
    L = np.zeros((12, 12))
    L[10, 8] = 1.0 # S -> singlet product
    LdagL = L.conjugate().T @ L
    eye = np.eye(L.shape[0])
    val_SD = np.kron(L.conjugate(), L) - 0.5 * (
        np.kron(eye, LdagL) + np.kron(LdagL.T, eye)
    )
    name = r"[L^\ast_{SD} \otimes L_{SD} - \frac{1}{2}(I\otimes L^\dagger_{SD} L_{SD} + (L^\dagger_{SD} L_{SD})^\top \otimes I)]"
    op = OpSite(name, ele_site, value=val_SD)
    kSDsym = Symbol("k_{SD}")
    subs[kSDsym] = kSD * SCALE
    lindblad += 1.0j * kSDsym * op

if kTD != 0.0:
    L = np.zeros((12, 12))
    L[11, 6] = 1.0 # T+ -> triplet product
    L[11, 7] = 1.0 # T0 -> triplet product
    L[11, 9] = 1.0 # T- -> triplet product
    LdagL = L.conjugate().T @ L
    eye = np.eye(L.shape[0])
    val_TD = np.kron(L.conjugate(), L) - 0.5 * (
        np.kron(eye, LdagL) + np.kron(LdagL.T, eye)
    )
    name = r"[L^\ast_{TD} \otimes L_{TD} - \frac{1}{2}(I\otimes L^\dagger_{TD} L_{TD} + (L^\dagger_{TD} L_{TD})^\top \otimes I)]"
    op = OpSite(name, ele_site, value=val_TD)
    kTDsym = Symbol("k_{TD}")
    subs[kTDsym] = kTD * SCALE
    lindblad += 1.0j * kTDsym * op

lindblad = lindblad.simplify()

# Construct MPO
# L = hyperfine + zeeman + exchange + dipolar + haberkorn + lindblad
L = hyperfine + zeeman + exchange + dipolar + lindblad
L = L.simplify()
am = AssignManager(L)
_ = am.assign()
mpo = am.numerical_mpo(subs=subs)


ndt = 4
dt = 1e-09 / ndt


# Liouville simulation
# "jax" is suitable for large bond dimension and GPU
backend = "jax"
Δt = dt / SCALE * units.au_in_fs  # dt in ns


# Dfine basis and initial singlet state
basis = []
hp = []
fuse_sites = []
for nuc in sim_C.molecules[1].nuclei:
    basis.append(Exciton(nstate=nuc.multiplicity**2))
    if isinstance(nuc, rp.data.FuseNucleus):
        fuse_sites.append(len(hp))
        hp.append(nuc.initial_density_matrix)
    else:
        hp.append(np.eye(nuc.multiplicity) / nuc.multiplicity)

if len(sim_C.molecules[0].nuclei) > 0:
    for i, nuc in enumerate(sim_C.molecules[0].nuclei):
        if i == len(sim_C.molecules[0].nuclei) // 2:
            # Haberkorn formalism
            # basis.append(Exciton(nstate=8**2))
            # Lindblad formalism
            basis.append(Exciton(nstate=12**2))
            hp.append(Qs_C)
        basis.append(Exciton(nstate=nuc.multiplicity**2))
        if isinstance(nuc, rp.data.FuseNucleus):
            fuse_sites.append(len(hp))
            hp.append(nuc.initial_density_matrix)
        else:
            hp.append(np.eye(nuc.multiplicity) / nuc.multiplicity)
else:
    # Haberkorn formalism
    # basis.append(Exciton(nstate=8**2))
    # Lindblad formalism
    basis.append(Exciton(nstate=12**2))
    hp.append(Qs_C)
for nuc in sim_D.molecules[1].nuclei:
    basis.append(Exciton(nstate=nuc.multiplicity**2))
    if isinstance(nuc, rp.data.FuseNucleus):
        hp.append(nuc.initial_density_matrix)
    else:
        hp.append(np.eye(nuc.multiplicity) / nuc.multiplicity)
basinfo = BasInfo([basis], spf_info=None)
hp = [item.reshape(-1).tolist() for item in hp]

nsite = len(basis)
print(f"{nsite=}")


# Define Liouvillian
op_dict = {
    tuple([(isite, isite) for isite in range(nsite)]): TensorOperator(mpo=mpo)
}
H = TensorHamiltonian(
    nsite, potential=[[op_dict]], kinetic=None, backend=backend
)

# eLt = scipy.linalg.expm((kCD * val_CD + kDC * val_DC) * dt)
# op_dict = {
#     ((ele_site, ele_site),): TensorOperator(
#         mpo=[eLt[None, :, :, None]], legs=(ele_site, ele_site)
#     )
# }
# expLt = TensorHamiltonian(
#     nsite, potential=[[op_dict]], kinetic=None, backend=backend
# )

# Haberkorn formalism
# subspace_inds={
#     ele_site: (
#          0,  1,  2,  3,
#          8,  9, 10, 11,
#         16, 17, 18, 19,
#         24, 25, 26, 27,
#                         36, 37, 38, 39,
#                         44, 45, 46, 47,
#                         52, 53, 54, 55,
#                         60, 61, 62, 63,
#     ) # ruff: noqa
# }
# Lindblad formalism
subspace_inds={
    ele_site: (
         0,  1,  2,  3, 
        12, 13, 14, 15, 
        24, 25, 26, 27, 
        36, 37, 38, 39, 
                        52,
                            65,
                                78, 79, 80, 81,
                                90, 91, 92, 93,
                                102, 103, 104, 105,
                                114, 115, 116, 117,
                                                  130,
                                                       143,
    ) # ruff: noqa
}

for fuse_site in fuse_sites:
    subspace_inds[fuse_site] = (
        0, 1, 2, 3,
        6, 7, 8, 9,
       12,13,14,15,
       18,19,20,21,
                   28,29,
                   34,35,
    ) # ruff: noqa


def process(H):
    operators = {"hamiltonian": H}
    model = Model(
        basinfo=basinfo,
        operators=operators,
        space="liouville",
        # one_gate_to_apply=expLt,  # <- changed!
        subspace_inds=subspace_inds,
    )
    model.m_aux_max = m
    model.init_HartreeProduct = [hp]

    integrator = "arnoldi"
    jobname = f"dnp_chi{m}_cutoff{cutoff}_eq_B{B0}_angle{angle}pi_over_8_n{ndt}_{integrator}"
    simulator = Simulator(
        jobname=jobname,
        model=model,
        backend=backend,
        verbose=0,
    )
    # Initiate the propagation setting with maxstep=0
    nstep = 151 * ndt
    ener, wf = simulator.propagate(
        reduced_density=(
            [(site, site) for site in range(nsite)],
            1,
        ),
        maxstep=nstep,
        stepsize=Δt,
        autocorr=False,
        energy=False,
        norm=False,
        populations=False,
        observables=False,
        integrator=integrator,
    )
    data = read_nc(f"{jobname}_prop/reduced_density.nc", [(ele_site, ele_site)])
    time_data = data["time"]
    density_data = data[(ele_site, ele_site)]
    return density_data, time_data


dm, time_data = process(H)
time_data_μs = time_data * SCALE * 1e06 / units.au_in_fs
