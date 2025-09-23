# radicalpair-tensornetwork
Script to reproduce manuscript

# Installation

## Install Python (for tensor network methods, symmetry reduction, and plotting)
We recommend using the [uv](https://docs.astral.sh/uv/) package manager. If you don't have uv installed, you can install it with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
All the dependencies are listed in `pyproject.toml`. You can install them with:
```bash
$ uv sync
```
The implementation of tensor network methods is based on this repository [https://github.com/QCLovers/PyTDSCF](https://github.com/QCLovers/PyTDSCF).

## Install Julia (for SC and SW calculations)
We recommend using [juliaup](https://github.com/JuliaLang/juliaup) to install Julia.
```
curl -fsSL https://install.julialang.org | sh
```
All the dependencies are listed in `Project.toml`. You can install them with:
```bash
$ julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
The implementation of SC and SW calculations is based on this repository [https://github.com/KenHino/ElectronSpinDynamics.jl](https://github.com/KenHino/ElectronSpinDynamics.jl).

## Install Fortran (for stochastic full wavefunction simulation)
See this repository [https://github.com/KenHino/Spin_dynamics](https://github.com/KenHino/Spin_dynamics) for details and set `PATH` to `spinchem` command.


# Run the code

## Benchmark for flavin anion and tryptophan cation
```bash
$ cd benchmark
```

### Stochastic MPS
```bash
$ uv run smps.py 0.5 16 128
```
where `0.5` is the magnetic field in mT, `16` is the bond dimension, and `128` is the number of samples.

### vectorised MPDO
```bash
$ uv run vmpdo.py 0.5 16
```
where `0.5` is the magnetic field in mT, and `16` is the bond dimension.

### Locally purified MPS
```bash
$ uv run lpmps.py 0.5 16
```
where `0.5` is the magnetic field in mT, and `16` is the bond dimension.


### SC and SW calculations
```bash
$ julia --project=.. --threads=12 tutorial.jl
```
where `12` is the number of threads. (Increase depending on the number of cores you have.)

### Stochastic full wavefunction simulation
```bash
$ spinchem input.ini
```


## Benchmark for toy model of identical nuclei

### Locally purified MPS
```bash
$ uv run toy-pmps.py 16 12
```
where `16` is the bond dimension, and `12` is the number of identical nuclei `N1=N2=12`.

### Exact solution
```bash
$ uv run symmetry_reduction.py 12
```


## Anisotropy of cryptochrome

### C-D hopping model by vMPDO
```bash
$ uv run crypto-aiso-twopairs.py 0.3 0
```
where `0.3` is the cutoff in mT, `0` is the angle in pi/8. We can set `angle` to 0, 1, 2, 3, 4, 5, 6, 7. 

### C or D only model by LPMPS
```bash
$ uv run crypto-aiso-onepair.py 0.3 0 C
```
where `0.3` is the cutoff in mT, `0` is the angle in pi/8, and `C` is the C or D.


# Plot the results

Please refer Jupyter notebooks. (change file paths if necessary)
The requirements is the same as the installation. (Thus, you can activate `uv run jupyter lab` to open the notebooks.)