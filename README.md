[![Python Test](https://github.com/K-Suzuki-Jij/compnal/actions/workflows/python_test.yml/badge.svg)](https://github.com/K-Suzuki-Jij/compnal/actions/workflows/python_test.yml) [![C++ Test](https://github.com/K-Suzuki-Jij/compnal/actions/workflows/cpp_test.yml/badge.svg)](https://github.com/K-Suzuki-Jij/compnal/actions/workflows/cpp_test.yml) [![codecov](https://codecov.io/gh/K-Suzuki-Jij/compnal/graph/badge.svg?token=TQ8O865AA0)](https://codecov.io/gh/K-Suzuki-Jij/compnal)

# Description
`CO`ndensed `M`atter `P`hysics `N`umerical `A`nalytics `L`ibrary (`COMPNAL`) is a numerical calculation library in the field of condensed matter physics. This library aims to provide a comprehensive set of numerical methods and algorithms tailored for analyzing various condensed matter systems.

# API Reference
[C++ Reference](https://k-suzuki-jij.github.io/compnal/)

# Features
`COMPNAL` can calculate the following models on the following lattices by the following solvers.

## Lattice
- One-dimensional chain
- Two-dimensional square lattice
- Three-dimensional cubic lattice
- Fully-connected lattice

## Model
### Classical models
- Ising model
- Polynomial Ising model

## Solver
### For Classical models
- Classical Monte Carlo method
    - Single spin flip
    - Parallel tempering

# Upcoming Features
We are actively working on expanding `COMPNAL` with the following upcoming features.

#### Lattice
- [ ] Two-dimensional triangular lattice
- [ ] Two-dimensional honeycomb lattice
- [ ] User-defined lattice

#### Model
- [ ] Classical model
    - [ ] Potts model

- [ ] Quantum model
    - [ ] Transverse field Ising model
    - [ ] Heisenberg model
    - [ ] Hubbard model
    - [ ] Kondo Lattice model

#### Algorithm
- [ ] Classical Monte Carlo method
    - [ ] Suwa-Todo algorithm
    - [ ] Wolff algorithm
    - [ ] Swendsen-Wang algorithm
- [ ] Exact Diagonalization
    - [ ] Lanczos method
    - [ ] Locally Optimal Block Preconditioned Conjugate Gradient method
- [ ] Density Matrix Renormalization Group

# Installation
## Install from PyPI
Only for Linux and MacOS.
```bash
pip install compnal
```

## Install from GitHub
To install the latest release of `compnal` from the source, use the following command:

```bash
pip install git+https://github.com/K-Suzuki-Jij/compnal.git
```
Before installation, make sure that the following dependencies are installed.


## Build from source
`COMPNAL` depends on the following libraries.
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [pybind11](https://github.com/pybind/pybind11)
- [OpenMP](https://www.openmp.org/)

### On MacOS
First, install the dependencies using [Homebrew](https://brew.sh/).
```bash
brew install cmake libomp
```

Then, clone this repository and install `COMPNAL`.
```bash
python -m pip install . -vvv
```

Run the test to check if the installation is successful.
```bash
python -m pytest tests
```

### On Linux
First, install the dependencies using apt.
```bash
sudo apt install cmake
```

Then, clone this repository and install `COMPNAL`.
```bash
python -m pip install . -vvv
```

Run the test to check if the installation is successful.
```bash
python -m pytest tests
```
