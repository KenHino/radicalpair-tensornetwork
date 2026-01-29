# radicalpair-tensornetwork
Script to reproduce manuscript
```bibtex
@misc{hino2025introductionmodellingradicalpair,
      title={Introduction of modelling radical pair quantum spin dynamics with tensor networks}, 
      author={Kentaro Hino and Damyan S. Frantzov and Yuki Kurashige and Lewis M. Antill},
      year={2025},
      eprint={2509.22104},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2509.22104}, 
}
```

# Installation

## Install Python (for tensor network methods, symmetry reduction, and plotting)
We recommend using the [uv](https://docs.astral.sh/uv/) package manager. If you don't have uv installed, you can install it with:
```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```
All the dependencies are listed in `pyproject.toml`. You can install them with:
```bash
$ uv sync
```
The implementation of tensor network methods is based on [PyTDSCF](https://github.com/QCLovers/PyTDSCF).
The implementation of general radical pair operation is based on [RadicalPy](https://github.com/Spin-Chemistry-Labs/radicalpy).

See [GPU support](https://github.com/QCLovers/PyTDSCF?tab=readme-ov-file#gpu-support) as needed.

## Install Julia (for SC and SW calculations)
We recommend using [juliaup](https://github.com/JuliaLang/juliaup) to install Julia.
```
$ curl -fsSL https://install.julialang.org | sh
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
<img width="597" height="554" alt="image" src="https://github.com/user-attachments/assets/0c49c80c-fdce-4c4f-911c-8202be3f4da6" />

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
The definition of parameters are in `input_SC.ini`.

### Stochastic full wavefunction simulation
```bash
$ spinchem input.ini
```


## Benchmark for toy model of identical nuclei
<img width="926" height="413" alt="image" src="https://github.com/user-attachments/assets/3d422a4f-58fd-4349-abe6-1d6f434ef84a" />

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
<img width="337" height="286" alt="image" src="https://github.com/user-attachments/assets/57485abc-1581-42cc-88c1-ccb9043f733d" />

```bash
$ cd anisotropy
```

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

The notebook and input files we used to generate the geometres are in `coordinate` directory. See json files for the resulting hyperfine tensors.

### Nuclear polarisation
<img width="579" height="338" alt="image" src="https://github.com/user-attachments/assets/4b9f943c-0eda-446e-9df0-e5f31f6059a2" />

```bash
$ uv run crypto-aiso-twopairs-dnp.py 0.1 0
```
will execute trace-preserving Lindblad simulation and export all one-site reduced dentisy matrices every time step.
Then, `plot-dnp.ipynb` shows $\langle I(t)\rangle$ trajectories including spheres.


# Plot results

Please refer Jupyter notebooks. (change file paths if necessary)
The requirement is the same as the installation. (Thus, you can activate `uv run jupyter lab` to open notebooks.)
