"""Locally purified MPS"""

import sys
from itertools import chain
import netCDF4 as nc
import numpy as np
import radicalpy as rp
from pympo import (
    AssignManager,
    OpSite,
    SumOfProducts,
    get_eye_site,
)
from sympy import Symbol

from pytdscf import BasInfo, Exciton, Model, Simulator, units
from pytdscf.dvr_operator_cls import TensorOperator
from pytdscf.hamiltonian_cls import TensorHamiltonian

B0 = float(sys.argv[1])  # in mT
m = int(sys.argv[2])  # Bond dimension >= 3
assert m >= 3

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


# Define electronic spin operators
# in LMPS, the length of MPS is twice longer than MPS!
ele_site = len(sim.molecules[0].nuclei) * 2
S1S2_op = OpSite(
    r"\hat{S}_1\cdot\hat{S}_2",
    ele_site,
    value=(sx_1 @ sx_2 + sy_1 @ sy_2 + sz_1 @ sz_2),
)
E_op = OpSite(r"\hat{E}", ele_site, value=np.eye(*sx_1.shape))

Qs_op = OpSite(r"\hat{Q}_S", ele_site, value=Qs)
Qt_op = OpSite(r"\hat{Q}_T", ele_site, value=Qt)

Sx_ops = []
Sy_ops = []
Sz_ops = []
Sx_ops.append(OpSite(r"\hat{S}_x^{(1)}", ele_site, value=sx_1))
Sy_ops.append(OpSite(r"\hat{S}_y^{(1)}", ele_site, value=sy_1))
Sz_ops.append(OpSite(r"\hat{S}_z^{(1)}", ele_site, value=sz_1))
Sx_ops.append(OpSite(r"\hat{S}_x^{(2)}", ele_site, value=sx_2))
Sy_ops.append(OpSite(r"\hat{S}_y^{(2)}", ele_site, value=sy_2))
Sz_ops.append(OpSite(r"\hat{S}_z^{(2)}", ele_site, value=sz_2))

Sr_ops = [Sx_ops, Sy_ops, Sz_ops]


# Define nuclear spin operators
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


# Define Hyperfine coupling Hamiltonian
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

# Define Zeeman Hamiltonian
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

# Define Exchange Hamiltonian
exchange = SumOfProducts()
Jsym = Symbol("J")
subs[Jsym] = J * SCALE
exchange += -Jsym * abs(g_ele_sym[0]) * (2 * S1S2_op + 0.5 * E_op)
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
                Dsym * abs(g_ele_sym[0]) * Sr_ops[k][0] * Sr_ops[l][1]
            )
dipolar = dipolar.simplify()
dipolar.symbol

# Define Haberkorn term
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

# Dammy identity site for ancilla legs
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

# Construct MPO
hamiltonian = hyperfine + zeeman + exchange + dipolar + haberkorn + 0.0 * dammy_op
hamiltonian = hamiltonian.simplify()
am = AssignManager(hamiltonian)
_ = am.assign()
mpo = am.numerical_mpo(subs=subs)

# Locally Purified MPS simulation
# "JAX" is suitable for large bond dimension or GPU
backend = "jax"
# dt = 1.0 ns, units.au_in_fs is magic word, because of time unit in PyTDSCF
Δt = 1e-09 / SCALE * units.au_in_fs / dn


# Define basis for LPMPS
basis = []
# Flavin
for nuc in sim.molecules[0].nuclei:
    # To add anicilla, we need to add two same basis
    for _ in range(2):
        basis.append(Exciton(nstate=nuc.multiplicity))
# Electron basis
basis.append(Exciton(nstate=4))
# Tryptophan
for nuc in sim.molecules[1].nuclei:
    for _ in range(2):
        basis.append(Exciton(nstate=nuc.multiplicity))
basinfo = BasInfo([basis], spf_info=None)

nsite = len(basis)


# Define where MPO has "legs"
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


# Define initial singlet state for LPMPS
def purified_state() -> list[np.ndarray]:
    hp = []
    for nuc in sim.molecules[0].nuclei:
        mult = nuc.multiplicity
        core_anci = np.zeros((1, mult, mult))
        core_phys = np.zeros((mult, mult, 1))
        core_anci[0, :, :] = np.eye(mult)
        core_phys[:, :, 0] = np.eye(mult)
        core_phys /= np.sqrt(mult)
        hp.append(core_anci)
        hp.append(core_phys)
    # electron site = singlet
    core_phys = np.zeros((1, 4, 1))
    core_phys[0, 2, 0] = 1.0
    hp.append(core_phys)
    # hp.append(core_anci)
    for nuc in sim.molecules[1].nuclei:
        mult = nuc.multiplicity
        core_phys = np.zeros((1, mult, mult))
        core_anci = np.zeros((mult, mult, 1))
        core_phys[0, :, :] = np.eye(mult)
        core_anci[:, :, 0] = np.eye(mult)
        core_phys /= np.sqrt(mult)
        hp.append(core_phys)
        hp.append(core_anci)
    assert len(hp) == nsite
    return hp


def process(H):
    operators = {"hamiltonian": H}
    model = Model(basinfo=basinfo, operators=operators, space="Hilbert")
    model.m_aux_max = m
    hp = purified_state()
    model.init_HartreeProduct = [hp]
    jobname = f"radicalpair_pmps_r{m}_B{B0}"
    simulator = Simulator(jobname=jobname, model=model, backend=backend, verbose=1)
    # Save diagonal element of reduced density matrix every 1 steps
    ener, wf = simulator.propagate(
        reduced_density=(
            [(ele_site,)],
            dn,
        ),
        maxstep=200*dn,
        stepsize=Δt,
        autocorr=False,
        energy=False,
        norm=False,
        populations=False,
        observables=False,
        conserve_norm=False,  # Since Haberkorn term is included
        integrator="arnoldi",  # or "Lanczos" if Hamiltonian is (skew-) Hermitian
    )

    with nc.Dataset(f"{jobname}_prop/reduced_density.nc", "r") as file:
        density_data_real = file.variables[f"rho_({ele_site},)_0"][:]["real"]
        density_data_imag = file.variables[f"rho_({ele_site},)_0"][:]["imag"]
        time_data = file.variables["time"][:]

    density_data = np.array(density_data_real) + 1.0j * np.array(density_data_imag)
    time_data = np.array(time_data)

    return density_data, time_data


density_data, time_data = process(H)
time_data_μs = time_data * SCALE * 1e06 / units.au_in_fs
