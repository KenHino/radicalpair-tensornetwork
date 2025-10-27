"""Stochastic MPS"""

"""Set number of thread before importing numpy, scipy etc"""
import multiprocessing as mp
import os

mp.set_start_method("fork", force=True)
nthreads = 1  # number of threads
max_workers = os.cpu_count() - 1   # number of processes
os.environ["OMP_NUM_THREADS"] = f"{nthreads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{nthreads}"
os.environ["MKL_NUM_THREADS"] = f"{nthreads}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{nthreads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{nthreads}"

import concurrent.futures
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor

import netCDF4 as nc
import numpy as np
import radicalpy as rp
from pympo import (
    AssignManager,
    OpSite,
    SumOfProducts,
)
from scipy.linalg import expm
from sympy import Symbol
from tqdm.auto import tqdm

from pytdscf import BasInfo, Exciton, Model, Simulator, units
from pytdscf.dvr_operator_cls import TensorOperator
from pytdscf.hamiltonian_cls import TensorHamiltonian


B0 = float(sys.argv[1])  # in mT
m = int(sys.argv[2])  # Bond dimension
n_sample = int(sys.argv[3]) # Number of samples

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
ele_site = len(sim.molecules[0].nuclei)
S1S2_op = OpSite(
    r"\hat{S}_1\cdot\hat{S}_2",
    ele_site,
    value=(sx_1 @ sx_2 + sy_1 @ sy_2 + sz_1 @ sz_2).real,
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
    Ix_ops[(0, j)] = OpSite(
        r"\hat{I}_x^{" + f"{(1, j + 1)}" + "}", j, value=nuc.pauli["x"]
    )
    Iy_ops[(0, j)] = OpSite(
        r"\hat{I}_y^{" + f"{(1, j + 1)}" + "}", j, value=nuc.pauli["y"]
    )
    Iz_ops[(0, j)] = OpSite(
        r"\hat{I}_z^{" + f"{(1, j + 1)}" + "}", j, value=nuc.pauli["z"]
    )

for j, nuc in enumerate(sim.molecules[1].nuclei):
    Ix_ops[(1, j)] = OpSite(
        r"\hat{I}_x^{" + f"{(2, j + 1)}" + "}",
        ele_site + 1 + j,
        value=nuc.pauli["x"],
    )
    Iy_ops[(1, j)] = OpSite(
        r"\hat{I}_y^{" + f"{(2, j + 1)}" + "}", ele_site + 1 + j, value=nuc.pauli["y"]
    )
    Iz_ops[(1, j)] = OpSite(
        r"\hat{I}_z^{" + f"{(2, j + 1)}" + "}",
        ele_site + 1 + j,
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
            dipolar += Dsym * abs(g_ele_sym[0]) * Sr_ops[k][0] * Sr_ops[l][1]
dipolar = dipolar.simplify()

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


# Construct MPO
hamiltonian = hyperfine + zeeman + exchange + dipolar + haberkorn
hamiltonian = hamiltonian.simplify()
am = AssignManager(hamiltonian)
_ = am.assign()
mpo = am.numerical_mpo(subs=subs)


# Mixed state simulation by Monte Carlo ensemble
# "JAX" is not suitable for parallelisation
backend = "numpy"
# dt = 1.0 ns, units.au_in_fs is magic word, because of time unit in PyTDSCF
Δt = 1e-09 / SCALE * units.au_in_fs / dn

# Define basis function for MPS
basis = []
for nuc in sim.molecules[0].nuclei:
    basis.append(Exciton(nstate=nuc.multiplicity))
basis.append(Exciton(nstate=4))
for nuc in sim.molecules[1].nuclei:
    basis.append(Exciton(nstate=nuc.multiplicity))
basinfo = BasInfo([basis], spf_info=None)

nsite = len(basis)

# Define Hamiltonian
op_dict = {tuple([(isite, isite) for isite in range(nsite)]): TensorOperator(mpo=mpo)}
H = TensorHamiltonian(nsite, potential=[[op_dict]], kinetic=None, backend=backend)


def spin_coherenet_state(pair):
    """
    J. Chem. Phys. 154, 084121 (2021); doi: 10.1063/5.0040519

    Sample from spin coherent state
    |Ω⁽ᴵ⁾⟩ = cos(θ/2)²ᴵ exp(tan(θ/2)exp(iϕ)Î₋) |I,I⟩
    """
    hp = []
    for isite in range(nsite):
        if isite == ele_site:
            hp.append([0, 0, 1, 0])  # Singlet
        else:
            mult = basis[isite].nstate
            I = (mult - 1) / 2
            nind = isite - int(ele_site <= isite)

            theta = np.arccos(pair[2 * nind] * 2 - 1.0)
            # same as
            # theta = np.arcsin(pair[2*nind] * 2 - 1.0)
            # if theta < 0:
            #    theta += np.pi
            phi = pair[2 * nind + 1] * 2 * np.pi
            weights = np.zeros((mult, 1))
            weights[0, 0] = 1.0
            weights = (
                (np.cos(theta / 2) ** (2 * I))
                * expm(
                    np.tan(theta / 2) * np.exp(1.0j * phi) * sim.nuclei[nind].pauli["m"]
                )
                @ weights
            )
            assert abs((weights.conjugate().T @ weights).real[0, 0] - 1.0) < 1.0e-14, (
                weights.conjugate().T @ weights
            )[0, 0]
            hp.append(weights.reshape(-1).tolist())
    return hp


def process_pair(pair, i, H):
    operators = {"hamiltonian": H}
    model = Model(basinfo=basinfo, operators=operators)
    model.m_aux_max = m
    # Get initial Hartree product (rank-1) state
    hp = spin_coherenet_state(pair)
    model.init_HartreeProduct = [hp]

    jobname = f"radicalpair_{i}"
    simulator = Simulator(jobname=jobname, model=model, backend=backend, verbose=0)
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
        conserve_norm=False,  # Because of Haberkorn term
        integrator="arnoldi",  # or Lanczos if Hamiltonian is (skew-) Hermitian
    )

    with nc.Dataset(f"{jobname}_prop/reduced_density.nc", "r") as file:
        density_data_real = file.variables[f"rho_({ele_site},)_0"][:]["real"]
        density_data_imag = file.variables[f"rho_({ele_site},)_0"][:]["imag"]
        time_data = file.variables["time"][:]

    # remove propagation directory
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)
    os.remove(f"wf_{jobname}.pkl")

    density_data = np.array(density_data_real) + 1.0j * np.array(density_data_imag)
    time_data = np.array(time_data)

    return density_data, time_data


futures = []
density_sum = None
engine = np.random.default_rng(0)
# sample from [0, 1]^N hypercubic
pairs = engine.random((n_sample, 2 * (nsite - 1)))
density_sums = []
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    try:
        density_sum = None
        active_futures = []
        i = 0  # number of submitted jobs
        j = 0  # number of finished jobs

        pbar = tqdm(total=len(pairs), desc="Processing pairs")
        while i < len(pairs) or active_futures:
            # Submit new jobs up to max_active
            while len(active_futures) < max_workers and i < len(pairs):
                future = executor.submit(process_pair, pairs[i], i, H)
                active_futures.append((future, i))
                i += 1

            # Wait for at least one job to complete
            done, not_done = concurrent.futures.wait(
                [f for f, _ in active_futures],
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            # Process completed jobs
            remaining_futures = []
            for future, job_i in active_futures:
                if future in done:
                    density_data, time_data = future.result()
                    if density_sum is None:
                        density_sum = density_data
                    else:
                        density_sum += density_data
                    j += 1
                    if j >= 32 and j.bit_count() == 1:
                        # when j in [32, 64, 128, ...] record result to estimate convergence of Monte Carlo
                        density_sums.append(density_sum / j)
                        # Save intermediate result as npz
                        np.savez(
                            f"radicalpair_sse_{j}samples_{m}m_{B0}B.npz",
                            density=density_sum / j,
                            time=time_data * SCALE * 1e06 / units.au_in_fs,
                        )
                    pbar.update(1)
                else:
                    remaining_futures.append((future, job_i))
            active_futures = remaining_futures

    except KeyboardInterrupt:
        print("\nCancelling active tasks...")
        for future, _ in active_futures:
            future.cancel()
        executor.shutdown(wait=False)
        pbar.close()
        raise

    pbar.close()

time_data_μs = time_data * SCALE * 1e06 / units.au_in_fs
density_data = density_sum / len(pairs)
