"""Vectorised MPDO"""

from math import isqrt
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

print(f"{pytdscf.__version__=}")

B0 = float(sys.argv[1])  # in mT
m = int(sys.argv[2])  # Bond dimension

if B0 >= 5.0:
    dn = 2 # In high magnetic field, use twice finer time step
else:
    dn = 1

def sort_nuclei(sim, isotropic):
    # Sort nuclei such that strongly coupled nuclear spin is allocated close to electronic site
    hf_magnitude_0 = []
    hf_magnitude_1 = []
    nuc_0 = sim.molecules[0].nuclei
    nuc_1 = sim.molecules[1].nuclei
    for nuc in nuc_0:
        if isotropic:
            hf_magnitude_0.append(abs(nuc.hfc.isotropic))
        else:
            hf_magnitude_0.append(np.linalg.norm(nuc.hfc.anisotropic))
    for nuc in nuc_1:
        if isotropic:
            hf_magnitude_1.append(abs(nuc.hfc.isotropic))
        else:
            hf_magnitude_1.append(np.linalg.norm(nuc.hfc.anisotropic))

    sim.molecules[0].nuclei = [nuc_0[i] for i in np.argsort(hf_magnitude_0)]
    sim.molecules[1].nuclei = [nuc_1[i] for i in np.argsort(hf_magnitude_1)[::-1]]


flavin = rp.simulation.Molecule.all_nuclei("flavin_anion")
trp = rp.simulation.Molecule.all_nuclei("tryptophan_cation")
for mol in [flavin, trp]:
    nucs = []
    for nuc in mol.nuclei:
        # Truncate nuclei with hyperfine coupling less than 0.1 mT
        if abs(nuc.hfc.isotropic) > 0.1:
            nucs.append(nuc)
    mol.nuclei = nucs
sim = rp.simulation.LiouvilleSimulation([flavin, trp])
isotropic = True
sort_nuclei(sim, isotropic)
A = {}
for i in range(len(sim.radicals)):
    for j, nuc in enumerate(sim.molecules[i].nuclei):
        if isotropic:
            A[(i, j)] = np.eye(3) * nuc.hfc.isotropic
        else:
            A[(i, j)] = nuc.hfc.anisotropic

B = np.array((0.0, 0.0, 1.0)) * B0
J = 0.224  # Fay et al 2020
D = -0.38  # Fay et al 2020
kS = 1.0e06  # in s-1
kT = 1.0e06  # in s-1
if isinstance(D, float):
    assert D <= 0
    D = 2 / 3 * np.diag((1.0, 1.0, -2.0)) * (-D)

# Clear nuclei temporally to get electronic spin operators
_nuclei_tmp0 = sim.molecules[0].nuclei
_nuclei_tmp1 = sim.molecules[1].nuclei
sim.molecules[0].nuclei = []
sim.molecules[1].nuclei = []

# for Singlet-Triplet basis
sx_1 = sim.spin_operator(0, "x")
sy_1 = sim.spin_operator(0, "y")
sz_1 = sim.spin_operator(0, "z")
sx_2 = sim.spin_operator(1, "x")
sy_2 = sim.spin_operator(1, "y")
sz_2 = sim.spin_operator(1, "z")

Qs = sim.projection_operator(rp.simulation.State.SINGLET)
Qt = sim.projection_operator(rp.simulation.State.TRIPLET)

# Revert nuclei
sim.molecules[0].nuclei = _nuclei_tmp0
sim.molecules[1].nuclei = _nuclei_tmp1


# Rescale energy and time for numerical stability
SCALE = 1.0e-09

# Save gyromagnetic ratio
gamma = [p.gamma_mT for p in sim.particles]
g_ele_sym = [
    Symbol(r"\gamma_e^{(" + f"{i + 1}" + ")}") for i in range(len(sim.radicals))
]
g_nuc_sym = {}
for i in range(len(sim.radicals)):
    for j in range(len(sim.molecules[i].nuclei)):
        g_nuc_sym[(i, j)] = Symbol(r"\gamma_n^{" + f"{(i + 1, j + 1)}" + "}")

subs = {}
for i, ge in enumerate(g_ele_sym):
    subs[ge] = sim.radicals[i].gamma_mT
for (i, j), gn in g_nuc_sym.items():
    subs[gn] = sim.molecules[i].nuclei[j].gamma_mT


# Define linear operation
def get_OE(op):
    """
    OT ⊗ 𝟙
    """
    return np.kron(op.T, np.eye(op.shape[0], dtype=op.dtype))


def get_EO(op):
    """
    𝟙 ⊗ O
    """
    return np.kron(np.eye(op.shape[0], dtype=op.dtype), op)


# Define electronic spin operators
ele_site = len(sim.molecules[0].nuclei)

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
EE_op = OpSite(r"\hat{E} ⊗ \hat{E}", ele_site, value=get_OE(np.eye(sx_1.shape[0])))

QsE_op = OpSite(r"\hat{Q}_S ⊗ 𝟙", ele_site, value=get_OE(Qs))
EQs_op = OpSite(r"𝟙 ⊗ \hat{Q}_S", ele_site, value=get_EO(Qs))
QtE_op = OpSite(r"\hat{Q}_T ⊗ 𝟙", ele_site, value=get_OE(Qt))
EQt_op = OpSite(r"𝟙 ⊗ \hat{Q}_T", ele_site, value=get_EO(Qt))


SxE_ops.append(OpSite(r"\hat{S}_x^{(1)\ast} ⊗ 𝟙", ele_site, value=get_OE(sx_1)))
SxE_ops.append(OpSite(r"\hat{S}_x^{(2)\ast} ⊗ 𝟙", ele_site, value=get_OE(sx_2)))
SyE_ops.append(OpSite(r"\hat{S}_y^{(1)\ast} ⊗ 𝟙", ele_site, value=get_OE(sy_1)))
SyE_ops.append(OpSite(r"\hat{S}_y^{(2)\ast} ⊗ 𝟙", ele_site, value=get_OE(sy_2)))
SzE_ops.append(OpSite(r"\hat{S}_z^{(1)\ast} ⊗ 𝟙", ele_site, value=get_OE(sz_1)))
SzE_ops.append(OpSite(r"\hat{S}_z^{(2)\ast} ⊗ 𝟙", ele_site, value=get_OE(sz_2)))

ESx_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_x^{(1)}", ele_site, value=get_EO(sx_1)))
ESx_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_x^{(2)}", ele_site, value=get_EO(sx_2)))
ESy_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_y^{(1)}", ele_site, value=get_EO(sy_1)))
ESy_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_y^{(2)}", ele_site, value=get_EO(sy_2)))
ESz_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_z^{(1)}", ele_site, value=get_EO(sz_1)))
ESz_ops.append(OpSite(r"𝟙 ⊗ \hat{S}_z^{(2)}", ele_site, value=get_EO(sz_2)))

SrE_ops = [SxE_ops, SyE_ops, SzE_ops]
ESr_ops = [ESx_ops, ESy_ops, ESz_ops]


# Define nuclear spin operators

IxE_ops = {}
IyE_ops = {}
IzE_ops = {}
EIx_ops = {}
EIy_ops = {}
EIz_ops = {}

for j, nuc in enumerate(sim.molecules[0].nuclei):
    val = nuc.pauli["x"]
    IxE_ops[(0, j)] = OpSite(
        r"\hat{I}_x^{" + f"{(1, j + 1)}" + r"\ast} ⊗ 𝟙",
        j,
        value=get_OE(val),
    )
    EIx_ops[(0, j)] = OpSite(
        r"𝟙 ⊗ \hat{I}_x^{" + f"{(1, j + 1)}" + "}",
        j,
        value=get_EO(val),
    )
    val = nuc.pauli["y"]
    IyE_ops[(0, j)] = OpSite(
        r"\hat{I}_y^{" + f"{(1, j + 1)}" + r"\ast} ⊗ 𝟙",
        j,
        value=get_OE(val),
    )
    EIy_ops[(0, j)] = OpSite(
        r"𝟙 ⊗ \hat{I}_y^{" + f"{(1, j + 1)}" + "}",
        j,
        value=get_EO(val),
    )
    val = nuc.pauli["z"]
    IzE_ops[(0, j)] = OpSite(
        r"\hat{I}_z^{" + f"{(1, j + 1)}" + r"\ast} ⊗ 𝟙",
        j,
        value=get_OE(val),
    )
    EIz_ops[(0, j)] = OpSite(
        r"𝟙 ⊗ \hat{I}_z^{" + f"{(1, j + 1)}" + "}",
        j,
        value=get_EO(val),
    )

for j, nuc in enumerate(sim.molecules[1].nuclei):
    site = ele_site + 1 + j
    val = nuc.pauli["x"]
    IxE_ops[(1, j)] = OpSite(
        r"\hat{I}_x^{" + f"{(2, j + 1)}" + r"\ast} ⊗ 𝟙",
        site,
        value=get_OE(val),
    )
    EIx_ops[(1, j)] = OpSite(
        r"𝟙 ⊗ \hat{I}_x^{" + f"{(2, j + 1)}" + "}",
        site,
        value=get_EO(val),
    )
    val = nuc.pauli["y"]
    IyE_ops[(1, j)] = OpSite(
        r"\hat{I}_y^{" + f"{(2, j + 1)}" + r"\ast} ⊗ 𝟙",
        site,
        value=get_OE(val),
    )
    EIy_ops[(1, j)] = OpSite(
        r"𝟙 ⊗ \hat{I}_y^{" + f"{(2, j + 1)}" + "}",
        ele_site + 1 + j,
        value=get_EO(val),
    )
    val = nuc.pauli["z"]
    IzE_ops[(1, j)] = OpSite(
        r"\hat{I}_z^{" + f"{(2, j + 1)}" + r"\ast} ⊗ 𝟙",
        site,
        value=get_OE(val),
    )
    EIz_ops[(1, j)] = OpSite(
        r"𝟙 ⊗ \hat{I}_z^{" + f"{(2, j + 1)}" + "}",
        site,
        value=get_EO(val),
    )

IrE_ops = [IxE_ops, IyE_ops, IzE_ops]
EIr_ops = [EIx_ops, EIy_ops, EIz_ops]


# Define Hyperfine coupling Hamiltonian

hyperfine = SumOfProducts()
xyz = "xyz"
for i in range(len(sim.radicals)):
    for j in range(len(sim.molecules[i].nuclei)):
        for k, (SrE_op, ESr_op) in enumerate(zip(SrE_ops, ESr_ops, strict=True)):
            for l, (IrE_op, EIr_op) in enumerate(zip(IrE_ops, EIr_ops, strict=True)):
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
hyperfine.symbol


# Define Zeeman Hamiltonian
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
    for i in range(len(sim.radicals)):
        zeeman += -Br * g_ele_sym[i] * ESr_op[i]
        zeeman += Br * g_ele_sym[i] * SrE_op[i]
        for j in range(len(sim.molecules[i].nuclei)):
            zeeman += -Br * g_nuc_sym[(i, j)] * EIr_op[(i, j)]
            zeeman -= -Br * g_nuc_sym[(i, j)] * IrE_op[(i, j)]

zeeman = zeeman.simplify()
zeeman.symbol


# Define Exchange Hamiltonian
exchange = SumOfProducts()
if J != 0.0:
    Jsym = Symbol("J")
    subs[Jsym] = J * SCALE
    exchange += -Jsym * abs(g_ele_sym[0]) * (2 * ES1S2_op + 0.5 * EE_op)
    exchange -= -Jsym * abs(g_ele_sym[0]) * (2 * S1S2E_op + 0.5 * EE_op)
    exchange = exchange.simplify()
exchange.symbol


# Define Dipolar Hamiltonian
dipolar = SumOfProducts()
for k in range(3):
    for l in range(3):
        if D[k, l] == 0.0:
            continue
        else:
            Dsym = Symbol("D_{" + f"{xyz[k]}" + f"{xyz[l]}" + "}")
            subs[Dsym] = D[k, l] * SCALE
            dipolar += (
                Dsym * abs(g_ele_sym[0]) * ESr_ops[k][0] * ESr_ops[l][1]
            )
            dipolar -= (
                Dsym * abs(g_ele_sym[0]) * SrE_ops[k][0] * SrE_ops[l][1]
            )
dipolar = dipolar.simplify()
dipolar.symbol


# Define Haberkorn relaxation
haberkorn = SumOfProducts()
if kS != 0.0:
    kSsym = Symbol("k_S")
    subs[kSsym] = kS * SCALE
    haberkorn -= 1.0j * kSsym / 2 * (QsE_op + EQs_op)
if kT != 0.0:
    kTsym = Symbol("k_T")
    subs[kTsym] = kT * SCALE
    haberkorn -= 1.0j * kTsym / 2 * (QtE_op + EQt_op)
haberkorn = haberkorn.simplify()
haberkorn.symbol


# Construct MPO for Livouvillian
L = hyperfine + zeeman + exchange + dipolar + haberkorn
L = L.simplify()
am = AssignManager(L)
_ = am.assign()
mpo = am.numerical_mpo(subs=subs)
L.symbol


# Liouville simulation
backend = "jax"  # "JAX" is suitable for large bond dimension and GPU
# dt = 1.0 ns, units.au_in_fs is magic word, because of time unit in PyTDSCF
Δt = 1.0e-09 / SCALE * units.au_in_fs / dn

# Define basis function for vMPDO
basis = []
for nuc in sim.molecules[0].nuclei:
    basis.append(Exciton(nstate=nuc.multiplicity**2))
basis.append(Exciton(nstate=4**2))
for nuc in sim.molecules[1].nuclei:
    basis.append(Exciton(nstate=nuc.multiplicity**2))
basinfo = BasInfo([basis], spf_info=None)

nsite = len(basis)


# Define Liouvillian (although it is named Hamiltonian)
op_dict = {tuple([(isite, isite) for isite in range(nsite)]): TensorOperator(mpo=mpo)}
H = TensorHamiltonian(nsite, potential=[[op_dict]], kinetic=None, backend=backend)


# Define initial singlet state in 1-rank form
def singlet_state():
    hp = []
    for isite in range(nsite):
        if isite == ele_site:
            op = Qs
        else:
            # Mixed states !
            op = np.eye(isqrt(basis[isite].nstate))
        # Automatically nomarized so that trace=1 in internal code
        hp.append(op.reshape(-1).tolist())
    return hp


def process(H):
    operators = {"hamiltonian": H}
    # space is "Liouville" rather than "Hilbert"
    model = Model(basinfo=basinfo, operators=operators, space="Liouville")
    model.m_aux_max = m
    hp = singlet_state()
    model.init_HartreeProduct = [hp]

    jobname = f"radicalpair_liouville_chi{m}_B{B0}"
    simulator = Simulator(
        jobname=jobname,
        model=model,
        backend=backend,
        verbose=0,
    )
    # Save whole reduced density matrix every 1 steps
    nstep = 200 * dn
    ener, wf = simulator.propagate(
        reduced_density=(
            [(ele_site, ele_site)],
            dn,
        ),
        maxstep=nstep,
        stepsize=Δt,
        autocorr=False,
        energy=False,
        norm=False,
        populations=False,
        observables=False,
        integrator="arnoldi",  # or Lanczos if linealised Liouvillian is (skew-) Hermitian
    )
    data = read_nc(f"{jobname}_prop/reduced_density.nc", [(ele_site, ele_site)])
    time_data = data["time"]
    density_data = data[(ele_site, ele_site)]
    return density_data, time_data


dm, time_data = process(H)
time_data_μs = time_data * SCALE * 1e06 / units.au_in_fs

# You can plot or save in your favorite.
