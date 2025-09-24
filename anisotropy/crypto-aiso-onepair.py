"""RP_C or RP_D by LPMPS"""

import json
import sys

import numpy as np
import radicalpy as rp
from pympo import (
    AssignManager,
    OpSite,
    SumOfProducts,
    get_eye_site,
)
from sympy import Symbol
from itertools import chain

import pytdscf
from pytdscf import BasInfo, Exciton, Model, Simulator, units
from pytdscf.dvr_operator_cls import TensorOperator
from pytdscf.hamiltonian_cls import TensorHamiltonian
from pytdscf.util import read_nc

cutoff = float(sys.argv[1])
angle = int(sys.argv[2])
assert angle in [0, 1, 2, 3, 4, 5, 6, 7]
CorD = str(sys.argv[3])
assert CorD in ["C", "D"]
B0 = float(sys.argv[4])
m = 1024  # Bond dimension

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

r_12 = np.array([9.480, -13.675, 5.388]) * 1e-10
r_13 = np.array([8.980, -18.684, 4.159]) * 1e-10
d_12 = np.linalg.norm(r_12)
d_13 = np.linalg.norm(r_13)

# p25 in SI of Nature 2021
kr_dict = {(0, 1): 1.7e07, (0, 2): 0.0}  # recombination
kf_dict = {(0, 1): 5.7e06, (0, 2): 1.0e07}  # fuluorecence

if CorD == "C":
    trp = get_mol_from_json("./coordinate/TrpC-wb97xd-lab.json", "trpC")
    r = r_12
    d = d_12
    kr = kr_dict[(0, 1)]
    kf = kf_dict[(0, 1)]
else:
    trp = get_mol_from_json("./coordinate/TrpD-wb97xd-lab.json", "trpD")
    r = r_13
    d = d_13
    kr = kr_dict[(0, 2)]
    kf = kf_dict[(0, 2)]

print(flavin)
print(trp)


def sort_nuclei(mol, reverse=False):
    # Sort nuclei
    hf_abs = []
    nucs = mol.nuclei
    for nuc in nucs:
        eigvals = np.linalg.eigvals(nuc.hfc.anisotropic)
        hf_abs.append(np.mean(np.abs(eigvals)))
    if reverse:
        mol.nuclei = [nucs[i] for i in np.argsort(hf_abs)[::-1]]
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


truncate_nuclei(flavin)
truncate_nuclei(trp)

fuse_nuclei(flavin)  # flavin has methyl hydrogens

sort_nuclei(flavin)
sort_nuclei(trp, reverse=True)

sim = rp.simulation.LiouvilleSimulation([flavin, trp])
B = np.array((np.sin(angle * np.pi / 8), 0.0, np.cos(angle * np.pi / 8))) * B0

A = {}
for i in range(2):
    mol = sim.molecules[i]
    for j, nuc in enumerate(mol.nuclei):
        A[(i, j)] = nuc.hfc.anisotropic

print(A)

kS = kf + kr  # Exponential model in s-1
kT = kf
# 1.35e-09 for B
J = rp.estimations.exchange_interaction_in_protein(d)
D_scalar = rp.estimations.dipolar_interaction_isotropic(d)  # 0.28

assert D_scalar <= 0
assert np.linalg.norm(r) == d, f"{np.linalg.norm(r)=} {d=}"
rx, ry, rz = r / d

Dtensor = np.eye(3) - 3.0 * np.array(
    [
        [rx * rx, rx * ry, rx * rz],
        [ry * rx, ry * ry, ry * rz],
        [rz * rx, rz * ry, rz * rz],
    ]
)
np.testing.assert_allclose(np.linalg.eigvalsh(Dtensor), np.array([-2, 1, 1]))

assert D_scalar <= 0
D = 2 / 3 * (-D_scalar) * Dtensor


sim

# Clear nuclei temporally
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


SCALE = 1.0e-09
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


# Define radical spin operators


ele_site = len(sim.molecules[0].nuclei) * 2

Sx_ops = []
Sy_ops = []
Sz_ops = []

S1S2_op = OpSite(
    r"\hat{S}_1\cdot\hat{S}_2",
    ele_site,
    value=(sx_1 @ sx_2 + sy_1 @ sy_2 + sz_1 @ sz_2),
)
E_op = OpSite(r"\hat{E}", ele_site, value=np.eye(*sx_1.shape))

Qs_op = OpSite(r"\hat{Q}_S", ele_site, value=Qs)
Qt_op = OpSite(r"\hat{Q}_T", ele_site, value=Qt)

Sx_ops.append(OpSite(r"\hat{S}_x^{(1)}", ele_site, value=sx_1))
Sy_ops.append(OpSite(r"\hat{S}_y^{(1)}", ele_site, value=sy_1))
Sz_ops.append(OpSite(r"\hat{S}_z^{(1)}", ele_site, value=sz_1))
Sx_ops.append(OpSite(r"\hat{S}_x^{(2)}", ele_site, value=sx_2))
Sy_ops.append(OpSite(r"\hat{S}_y^{(2)}", ele_site, value=sy_2))
Sz_ops.append(OpSite(r"\hat{S}_z^{(2)}", ele_site, value=sz_2))

Sr_ops = [Sx_ops, Sy_ops, Sz_ops]


Ix_ops = {}
Iy_ops = {}
Iz_ops = {}

for j, nuc in enumerate(sim.molecules[0].nuclei):
    site = 2 * j + 1
    Ix_ops[(0, j)] = OpSite(
        r"\hat{I}_x^{" + f"{(1, j + 1)}" + "}", site, value=nuc.pauli["x"]
    )
    Iy_ops[(0, j)] = OpSite(
        r"\hat{I}_y^{" + f"{(1, j + 1)}" + "}", site, value=nuc.pauli["y"]
    )
    Iz_ops[(0, j)] = OpSite(
        r"\hat{I}_z^{" + f"{(1, j + 1)}" + "}", site, value=nuc.pauli["z"]
    )

for j, nuc in enumerate(sim.molecules[1].nuclei):
    site = ele_site - 1 + (j + 1) * 2
    Ix_ops[(1, j)] = OpSite(
        r"\hat{I}_x^{" + f"{(2, j + 1)}" + "}",
        site,
        value=nuc.pauli["x"],
    )
    Iy_ops[(1, j)] = OpSite(
        r"\hat{I}_y^{" + f"{(2, j + 1)}" + "}",
        site,
        value=nuc.pauli["y"],
    )
    Iz_ops[(1, j)] = OpSite(
        r"\hat{I}_z^{" + f"{(2, j + 1)}" + "}",
        site,
        value=nuc.pauli["z"],
    )

Ir_ops = [Ix_ops, Iy_ops, Iz_ops]


hyperfine = SumOfProducts()

xyz = "xyz"

for i in range(len(sim.radicals)):
    for j in range(len(sim.molecules[i].nuclei)):
        for k, Sr_op in enumerate(Sr_ops):
            for l, Ir_op in enumerate(Ir_ops):
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
                hyperfine += Asym * abs(g_ele_sym[i]) * Sr_op[i] * Ir_op[(i, j)]


hyperfine = hyperfine.simplify()
hyperfine.symbol

zeeman = SumOfProducts()

xyz = "xyz"

for k, (Sr_op, Ir_op) in enumerate(zip(Sr_ops, Ir_ops, strict=True)):
    if B[k] == 0.0:
        continue
    r = xyz[k]
    Br = Symbol(f"B_{r}")
    subs[Br] = B[k] * SCALE
    for i in range(len(sim.radicals)):
        zeeman += -Br * g_ele_sym[i] * Sr_op[i]
        for j in range(len(sim.molecules[i].nuclei)):
            zeeman += -Br * g_nuc_sym[(i, j)] * Ir_op[(i, j)]

zeeman = zeeman.simplify()
zeeman.symbol

exchange = SumOfProducts()
Jsym = Symbol("J")
subs[Jsym] = J * SCALE
exchange += -Jsym * abs(g_ele_sym[0]) * (2 * S1S2_op + 0.5 * E_op)
exchange = exchange.simplify()
exchange.symbol


dipolar = SumOfProducts()
for k in range(3):
    for l in range(3):
        if D[k, l] == 0.0:
            continue
        else:
            Dsym = Symbol("D_{" + f"{xyz[k]}" + f"{xyz[l]}" + "}")
            subs[Dsym] = D[k, l] * SCALE
            dipolar += Dsym * abs(g_ele_sym[0]) * Sr_ops[k][0] * Sr_ops[l][1]
dipolar = dipolar.simplify()
dipolar.symbol


haberkorn = SumOfProducts()
if kS != 0.0:
    kSsym = Symbol("k_S")
    subs[kSsym] = kS * SCALE
    haberkorn -= 1.0j * kSsym / 2 * Qs_op
if kT != 0.0:
    kTsym = Symbol("k_T")
    subs[kTsym] = kT * SCALE
    haberkorn -= 1.0j * kTsym / 2 * Qt_op
haberkorn = haberkorn.simplify()
haberkorn.symbol


# Dammy eye site
eye_sites = []
for nuc in sim.molecules[0].nuclei:
    eye_sites.append(get_eye_site(i=len(eye_sites), n_basis=nuc.multiplicity))
    eye_sites.append(get_eye_site(i=len(eye_sites), n_basis=nuc.multiplicity))

eye_sites.append(get_eye_site(i=len(eye_sites), n_basis=4))

for nuc in sim.molecules[1].nuclei:
    eye_sites.append(get_eye_site(i=len(eye_sites), n_basis=nuc.multiplicity))
    eye_sites.append(get_eye_site(i=len(eye_sites), n_basis=nuc.multiplicity))

dammy_op = 1
for op in eye_sites:
    dammy_op *= op

H = hyperfine + zeeman + exchange + dipolar + haberkorn + 0.0 * dammy_op
H = H.simplify()
am = AssignManager(H)
_ = am.assign()
mpo = am.numerical_mpo(subs=subs)


backend = "jax"
Δt = 1.0e-09 / SCALE * units.au_in_fs  # dt = 1.0 ns

basis = []
# Flavin
for nuc in sim.molecules[0].nuclei:
    # To add anicilla, we need to add two same basis
    for _ in range(2):
        basis.append(Exciton(nstate=nuc.multiplicity))
# Electron basis (pure state)
basis.append(Exciton(nstate=4))
# Tryptophan
for nuc in sim.molecules[1].nuclei:
    for _ in range(2):
        basis.append(Exciton(nstate=nuc.multiplicity))
basinfo = BasInfo([basis], spf_info=None)

nsite = len(basis)


nsite = len(basis)


# key = ((0, 0), (1,), (2, 2), (3,), (4, 4), (5,), (6, 6))
# while
# leg is (0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6)
key = []
for i in range(nsite):
    if i == ele_site:
        act_site = (i, i)
    else:
        if i % 2 == 1:
            act_site = (i, i)
        else:
            act_site = (i,)
    key.append(act_site)
key = tuple(key)  # list is not hashable


op_dict = {key: TensorOperator(mpo=mpo, legs=tuple(chain.from_iterable(key)))}
H = TensorHamiltonian(nsite, potential=[[op_dict]], kinetic=None, backend=backend)


def purified_state() -> list[np.ndarray]:
    hp = []
    for nuc in sim.molecules[0].nuclei:
        mult = nuc.multiplicity
        core_anci = np.zeros((1, mult, mult))
        core_phys = np.zeros((mult, mult, 1))
        core_anci[0, :, :] = np.eye(mult)
        if isinstance(nuc, rp.data.FuseNucleus):
            core_phys[:, :, 0] = nuc.initial_density_matrix
            core_phys /= np.linalg.norm(core_phys)
        else:
            core_phys[:, :, 0] = np.eye(mult)
            core_phys /= np.sqrt(mult)
        hp.append(core_anci)
        hp.append(core_phys)
    # electron site = singlet
    core_phys = np.zeros((1, 4, 1))
    # core_anci = np.zeros((4, 4, 1))
    core_phys[0, 2, 0] = 1.0
    # core_anci[0, 2, 0] = 1.0
    hp.append(core_phys)
    # hp.append(core_anci)
    for nuc in sim.molecules[1].nuclei:
        mult = nuc.multiplicity
        core_phys = np.zeros((1, mult, mult))
        core_anci = np.zeros((mult, mult, 1))
        if isinstance(nuc, rp.data.FuseNucleus):
            core_phys[:, :, 0] = nuc.initial_density_matrix
            core_phys /= np.linalg.norm(core_phys)
        else:
            core_phys[0, :, :] = np.eye(mult)
            core_phys /= np.sqrt(mult)
        core_anci[:, :, 0] = np.eye(mult)
        hp.append(core_phys)
        hp.append(core_anci)
    assert len(hp) == nsite
    return hp


def process(H):
    operators = {"hamiltonian": H}
    # LPMPS is simulated in Hilbert space.
    model = Model(basinfo=basinfo, operators=operators, space="Hilbert")
    model.m_aux_max = m
    hp = purified_state()
    model.init_HartreeProduct = [hp]

    jobname = f"aiso_chi{m}_cutoff{cutoff}_{CorD}_B{B0}_angle{angle}pi_over_8"
    simulator = Simulator(
        jobname=jobname,
        model=model,
        backend=backend,
        verbose=0,
    )
    # Initiate the propagation setting with maxstep=0
    nstep = 151
    ener, wf = simulator.propagate(
        reduced_density=(
            [(ele_site,)],
            1,
        ),
        maxstep=nstep,
        stepsize=Δt,
        autocorr=False,
        energy=False,
        norm=False,
        populations=False,
        observables=False,
        conserve_norm=False,  # Since Haberkorn term
        integrator="arnoldi",
    )
    data = read_nc(f"{jobname}_prop/reduced_density.nc", [(ele_site,)])
    time_data = data["time"]
    density_data = data[(ele_site,)]
    return density_data, time_data


dm, time_data = process(H)
time_data_μs = time_data * SCALE * 1e06 / units.au_in_fs
