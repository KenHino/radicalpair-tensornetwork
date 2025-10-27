import radicalpy as rp
import numpy as np
import shutil
import pathlib
from typing import Literal, Sequence
from scipy.constants import hbar, physical_constants
import polars as pl

_simulation_type = Literal[
    "trace_sampling", "exact_dynamics", "symmetry_dynamics", "semi_classical"
]


def _validate_input(
    *,
    sim: rp.simulation.HilbertSimulation,
    input_file_name: pathlib.Path | str,
    isotropic: bool,
    J: float,
    D: np.ndarray | float,
    kS: float,
    kT: float,
    simulation_type: _simulation_type,
    output_folder: pathlib.Path | str,
    seed: tuple[int, int],
    B: Sequence[float] | float,
    initial_state: Literal["singlet", "triplet"],
    N_samples: int,
    simulation_time: float,
    dt: float,
    N_krylov: int,
    integrator_tolerance: float,
    remove_files: bool,
) -> tuple[
    pathlib.Path,
    pathlib.Path,
    tuple[float, float, float, float, float, float, float, float, float],
    tuple[float, ...],
]:
    """
    Validate input parameters
    """
    assert isinstance(sim, rp.simulation.HilbertSimulation), (
        f"given sim is not a HilbertSimulation: {type(sim)=}"
    )
    input_file_path = pathlib.Path(input_file_name).resolve()
    if input_file_path.exists() and remove_files:
        input_file_path.unlink()
    elif input_file_path.exists() and not remove_files:
        raise FileExistsError(
            f"input_file_name {input_file_name} already exists. Please remove it or use a different input_file_name."
        )

    if not isotropic:
        raise NotImplementedError("anisotropic calculation is not yet implemented")

    assert isinstance(J, float), f"given J is not a float: {type(J)=}"
    if isinstance(D, float):
        assert D <= 0.0, f"given D is not negative: {D=}"
        D = 2 / 3 * np.diag((-1.0, -1.0, 2.0)) * D  # * abs(sim.radicals[0].gamma_mT)
    assert isinstance(D, np.ndarray), f"given D is not a np.ndarray: {type(D)=}"
    np.testing.assert_allclose(D, D.T, atol=1.0e-12)
    if isotropic:
        assert D[0, 0] == D[1, 1] == -0.5 * D[2, 2], f"given D is not isotropic: {D=}"
        assert D[0, 1] == D[0, 2] == D[1, 2] == 0.0, f"given D is not isotropic: {D=}"
    _D_flat = D.reshape(-1, order="F")  # / abs(sim.radicals[0].gamma_mT)
    assert _D_flat.shape == (9,), f"given D is not a 3x3 array: {D.shape=}"
    D_flat = tuple(_D_flat)
    assert isinstance(kS, float), f"given kS is not a float: {type(kS)=}"
    assert kS >= 0.0, f"given kS is negative: {kS=}"
    assert isinstance(kT, float), f"given kT is not a float: {type(kT)=}"
    assert kT >= 0.0, f"given kT is negative: {kT=}"
    assert isinstance(simulation_type, str), (
        f"given simulation_type is not a str: {type(simulation_type)=}"
    )
    _simulation_type = simulation_type.lower()
    assert _simulation_type in [
        "trace_sampling",
        "exact_dynamics",
        "symmetry_dynamics",
        "semi_classical",
    ], (
        f"given simulation_type is not in ['trace_sampling', 'exact_dynamics', 'symmetry_dynamics', 'semi_classical']: {simulation_type=}"
    )
    if _simulation_type not in [
        "trace_sampling",
        "exact_dynamics",
        "symmetry_dynamics",
    ]:
        raise NotImplementedError(
            f"simulation_type {simulation_type} is not yet debugged"
        )
    assert isinstance(output_folder, (pathlib.Path, str)), (
        f"given output_folder is not a pathlib.Path or str: {type(output_folder)=}"
    )
    # full path
    output_folder_path = pathlib.Path(output_folder).resolve()
    if output_folder_path.exists() and remove_files:
        print("remove output folder")
        shutil.rmtree(output_folder_path)
        print(f"create output folder {output_folder_path}")
        output_folder_path.mkdir(parents=True, exist_ok=True)
    elif output_folder_path.exists() and not remove_files:
        raise FileExistsError(
            f"output_folder {output_folder} already exists. Please remove it or use a different output_folder."
        )
    else:
        print(f"create output folder {output_folder}")
        output_folder_path.mkdir(parents=True, exist_ok=True)

    assert isinstance(seed, tuple), f"given seed is not a tuple: {type(seed)=}"
    assert len(seed) == 2, f"given seed is not a tuple of length 2: {len(seed)=}"
    assert seed[0] != seed[1], f"given seed is not a tuple of different seeds: {seed=}"

    if isinstance(B, float):
        B_tuple: tuple[float, ...] = (float(B),)
    elif isinstance(B, Sequence):
        B_tuple = tuple(float(b) for b in B)
    else:
        raise TypeError(
            f"given B is neither a float nor a sequence of floats: {type(B)=}"
        )

    assert isinstance(initial_state, str), (
        f"given initial_state is not a str: {type(initial_state)=}"
    )
    assert initial_state.lower() in ["singlet", "triplet"], (
        f"given initial_state is not in ['singlet', 'triplet']: {initial_state=}"
    )

    assert isinstance(N_samples, int), (
        f"given N_samples is not an int: {type(N_samples)=}"
    )
    assert N_samples > 0, f"given N_samples is negative: {N_samples=}"

    assert isinstance(simulation_time, float), (
        f"given simulation_time is not a float: {type(simulation_time)=}"
    )
    assert simulation_time > 0.0, (
        f"given simulation_time is negative: {simulation_time=}"
    )

    assert isinstance(dt, float), f"given dt is not a float: {type(dt)=}"
    assert dt > 0.0, f"given dt is negative: {dt=}"

    assert isinstance(N_krylov, int), f"given N_krylov is not an int: {type(N_krylov)=}"
    assert N_krylov > 3, f"given N_krylov is too small: {N_krylov=}"

    assert isinstance(integrator_tolerance, float), (
        f"given integrator_tolerance is not a float: {type(integrator_tolerance)=}"
    )
    assert integrator_tolerance > 0.0, (
        f"given integrator_tolerance is negative: {integrator_tolerance=}"
    )
    assert integrator_tolerance < 1.0e-01, (
        f"given integrator_tolerance is too large: {integrator_tolerance=}"
    )

    # Finally, create empty input file
    with open(input_file_path, "w") as f:
        f.write("")

    return input_file_path, output_folder_path, D_flat, B_tuple


def _dump_system_variables(
    input_file_name: pathlib.Path | str,
    *,
    J: float,
    D: tuple[float, float, float, float, float, float, float, float, float],
    kS: float,
    kT: float,
) -> None:
    """
    Dump system variables to input file
    """
    with open(input_file_name, "a") as f:
        f.write("[system variables]\n")
        f.write(f"J = {J}\n")
        f.write(f"D = {D[0]} {D[1]} {D[2]} {D[3]} {D[4]} {D[5]} {D[6]} {D[7]} {D[8]}\n")
        f.write(f"kS = {kS}\n")
        f.write(f"kT = {kT}\n")


def _dump_hyperfine_tensor(
    input_file_name: pathlib.Path | str,
    *,
    mol: rp.simulation.Molecule,
    isotropic: bool,
) -> None:
    """
    Dump hyperfine tensor to input file
    """
    with open(input_file_name, "a") as f:
        for i in range(len(mol.nuclei)):
            if isotropic:
                A = np.eye(3) * mol.nuclei[i].hfc.isotropic
            else:
                A = mol.nuclei[i].hfc.anisotropic
            A = A.reshape(-1, order="F")
            A_string = " ".join([str(a) for a in A])
            f.write(f"A{i + 1} = {A_string}\n")


def _dump_electron_i_variables(
    input_file_name: pathlib.Path | str,
    *,
    i: int,
    mol: rp.simulation.Molecule,
    isotropic: bool,
) -> None:
    """
    Dump electron i variables to input file
    """
    with open(input_file_name, "a") as f:
        f.write(f"[electron {i + 1}]\n")
        # Gamma = g * mu_B / hbar
        f.write(
            f"g = {-mol.radical.gamma_mT / physical_constants['Bohr magneton'][0] * hbar * 1e3:.7f}\n"
        )
        I_string = " ".join([str(n.multiplicity) for n in mol.nuclei])
        N_I_string = " ".join([str(1) for _ in mol.nuclei])
        f.write(f"I = {I_string}\n")
        f.write(f"N_I = {N_I_string}\n")
    _dump_hyperfine_tensor(
        input_file_name=input_file_name,
        mol=mol,
        isotropic=isotropic,
    )


def _dump_electron_variables(
    input_file_name: pathlib.Path | str,
    *,
    sim: rp.simulation.HilbertSimulation,
    isotropic: bool,
) -> None:
    """
    Dump electron variables to input file
    """
    for i in range(len(sim.molecules)):
        _dump_electron_i_variables(
            input_file_name=input_file_name,
            i=i,
            mol=sim.molecules[i],
            isotropic=isotropic,
        )


def _dump_simulation_parameters(
    input_file_name: pathlib.Path | str,
    *,
    simulation_type: _simulation_type,
    output_folder: pathlib.Path | str,
    seed: tuple[int, int],
    B: Sequence[float],
    initial_state: Literal["singlet", "triplet"],
    N_samples: int,
    simulation_time: float,
    dt: float,
    N_krylov: int,
    integrator_tolerance: float,
    M1: int,
    M2: int,
    block_tolerance: float,
) -> None:
    """
    Dump simulation parameters to input file
    """
    with open(input_file_name, "a") as f:
        f.write("[simulation parameters]\n")
        f.write(f"simulation_type = {simulation_type}\n")
        f.write(f"output_folder = {output_folder}\n")
        f.write(f"seed = {seed[0]} {seed[1]}\n")
        B_string = " ".join([str(b) for b in B])
        f.write(f"B = {B_string}\n")
        f.write(f"initial_state = {initial_state}\n")
        f.write(f"simulation_time = {simulation_time}\n")
        f.write(f"dt = {dt}\n")
        f.write(f"N_krylov = {N_krylov}\n")
        f.write(f"integrator_tolerance = {integrator_tolerance}\n")
        if simulation_type in ["trace_sampling", "symmetry_dynamics"]:
            f.write(f"N_samples = {N_samples}\n")
        if simulation_type == "symmetry_dynamics":
            f.write(f"M1 = {M1}\n")
            f.write(f"M2 = {M2}\n")
            f.write(f"block_tolerance = {block_tolerance}\n")


def dump_input(
    sim: rp.simulation.HilbertSimulation,
    input_file_name: pathlib.Path | str = "input.ini",
    *,
    remove_files: bool = True,
    isotropic: bool = True,
    J: float = 0.0,
    D: np.ndarray | float = 0.0,
    kS: float = 47.0,
    kT: float = 0.0,
    simulation_type: _simulation_type = "trace_sampling",
    output_folder: pathlib.Path | str = "out",
    seed: tuple[int, int] = (42, 99),
    B: Sequence[float] | float = (0.0,),
    initial_state: Literal["singlet", "triplet"] = "singlet",
    N_samples: int = 200,
    simulation_time: float = 50.0,
    dt: float = 1.0,
    N_krylov: int = 7,
    integrator_tolerance: float = 1e-5,
    M1: int = 1,
    M2: int = 1,
    block_tolerance: float = 1e-5,
) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Dump input file for Spin_chemistry from radicalpy.HilbertSimulator

    Args:
        sim (rp.simulation.HilbertSimulation):
            Hilbert simulator
        input_file_name (pathlib.Path | str):
            Input file name
        J (float):
            Exchange coupling constant in mT
        D (np.ndarray | float):
            D-tensor in mT. Array should be (3, 3) in C-order.
        kS (float):
            Haberkorn kinetic constant in 1 / sec for singlet dephasing
        kT (float):
            Haberkorn kinetic constant in 1 / sec for triplet dephasing
        simulation_type (Literal["trace_sampling", "exact_dynamics", "symmetry_dynamics", "semi_classical"]):
            Simulation type
        output_folder (pathlib.Path | str):
            Output folder
        seed (tuple[int, int]):
            Seed
        B (Sequence[float] | float):
            Magnetic field in mT. Direction of angle????
        initial_state (Literal["singlet", "triplet"]):
            Initial state
        N_samples (int):
            Number of samples for trace sampling
        simulation_time (float):
            Simulation time in sec
        dt (float):
            Time step in sec
        N_krylov (int):
            Number of Krylov vectors
        integrator_tolerance (float):
            Integrator tolerance
        remove_files (bool):
            Remove input/output file if it exists

    Returns:
        tuple[pathlib.Path, pathlib.Path]:
            Path to the input file and output folder

    Example:
       >>> import radicalpy as rp
       >>> from utils import dump_input
       >>> flavin = rp.simulation.Molecule.fromisotopes(isotopes=["1H"], hfcs=[0.4])
       >>> Z = rp.simulation.Molecule.fromisotopes(isotopes=["1H"], hfcs=[0.5])
       >>> sim = rp.simulation.HilbertSimulation([flavin, Z])
       >>> input_path, output_path = dump_input(sim=sim, output_folder="out")
       >>> print(input_path)
       >>> with open(input_path) as f:
       ...     print(f.read()) # cat contents of input file
    """

    input_path, output_path, D_flat, B_tuple = _validate_input(
        sim=sim,
        input_file_name=input_file_name,
        isotropic=isotropic,
        J=J,
        D=D,
        kS=kS,
        kT=kT,
        simulation_type=simulation_type,
        output_folder=output_folder,
        seed=seed,
        B=B,
        initial_state=initial_state,
        N_samples=N_samples,
        simulation_time=simulation_time,
        dt=dt,
        N_krylov=N_krylov,
        integrator_tolerance=integrator_tolerance,
        remove_files=remove_files,
    )
    _dump_system_variables(
        input_file_name=input_path,
        J=J,
        D=D_flat,
        kS=kS * 1e-6,  # convert to 1 / microsec
        kT=kT * 1e-6,  # convert to 1 / microsec
    )
    _dump_electron_variables(
        input_file_name=input_path,
        sim=sim,
        isotropic=isotropic,
    )
    _dump_simulation_parameters(
        input_file_name=input_path,
        simulation_type=simulation_type,
        output_folder=output_folder,
        seed=seed,
        B=B_tuple,
        initial_state=initial_state,
        N_samples=N_samples,
        simulation_time=simulation_time * 1e09,  # convert to ns
        dt=dt * 1e09,  # convert to ns
        N_krylov=N_krylov,
        integrator_tolerance=integrator_tolerance,
        M1=M1,
        M2=M2,
        block_tolerance=block_tolerance,
    )

    return input_path, output_path


def parse_output(
    output_path: pathlib.Path | str,
    subdir: str,
    dt: float,
) -> pl.DataFrame:
    """
    Parse output file
    out/.200/
    ├── S_prob.data
    ├── T0_prob.data
    ├── Tm_prob.data
    └── Tp_prob.data

    Data format
    0.000
    1.000
    2.000
    """
    subdir_path = pathlib.Path(output_path) / subdir
    data_dict = {}

    for file in subdir_path.glob("*.data"):
        if file.stem[-5:] != "_prob":
            continue
        data = np.loadtxt(file)
        column_name = file.stem
        data_dict[column_name] = data

    if data_dict:
        first_data = next(iter(data_dict.values()))
        time_values = np.arange(len(first_data)) * dt * 1e06  # convert to microsec
        data_dict["time"] = time_values
    else:
        raise FileNotFoundError(f"No data files found in {subdir_path}")

    df = pl.DataFrame(data_dict)
    return df
