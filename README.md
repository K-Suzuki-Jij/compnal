# compnal
`CO`ndensed `M`atter `P`hysics `N`umerical `A`nalytics `L`ibrary.
Currently, We are working on the classical monte carlo, exact diagonalization method and the density matrix renormalization group.
Please note that this is beta version and there may be some bugs.

## Support

* macOS/Linux
* Python3.9 or higher
* OpenMP

We test on 
* macOS 13.1 on Apple Silicon and Intel CPU
* Ubuntu 20.04.5 LTS on Ryzen 7950x

## Install
We recommend using virtual environment.

```python
python -m venv .venv
```

Activate `venv`
```shell
source .venv/bin/activate
```

Install on `venv`

```python
python -m pip instal .
```

## Usage
### Classical Monte Carlo
We explain classical monte carlo simulation for two-dimensional Ising model with nearest neighbor interactions on the square lattice under the periodic boundary condition ( PBC):
$$
E=J\sum_{i, j}\sigma_{i}\sigma_{j} + h\sum_{i=1}\sigma_i,\;\;\;\sigma_{i}\in \{-1, +1\}
$$

First, we import `compnal` and set up lattice:
```python
import compnal

lattice = compnal.lattice.Square(
    x_size = 6,
    y_size = 6,
    boundary_condition = compnal.BoundaryCondition.PBC
)
```
Here, we set the system size as $N=6\times 6=36$ and the boundary condition as PBC.
Then, we make Ising model:
```python
model = compnal.model.Ising(
    lattice = lattice,
    linear = 0.0,
    quadratic = -1.0
)
```
Here, we set linear interaction as $h=0.0$ and quadratic interaction $J=-1.0$.
Next, we prepare classical monte carlo solver:
```python
solver = compnal.solver.ClassicalMonteCarlo(
    model=model
)
solver.algorithm = compnal.Algorithm.METROPOLIS
solver.num_samples = 100
solver.num_sweeps = 1000
solver.num_threads = 4
solver.temperature = 1.0
```
Here, we set the updater algorithm as Metropolis method, the number of samples as 100, the number of sweeps as 1000, the number of threads in the simulation, and the temperature as $T=1.0$.
Finally, we execute sampling:
```python
solver.run()
```
After the simulation, you can get spin configurations:

```python
samples = solver.get_samples()
```

Since we set $T=1.0$, the system in the ferromagnetic phase. We can check this by calculating 
$$
\langle |m| \rangle = \frac{1}{N\times\rm num\_samples}\sum^{\rm num\_samples}_{i=1}\left|\sum^{N}_{j=1}\sigma^{(i)}_{j}\right|.
$$
Here, $\sigma^{(i)}_{j}$ is the $j$-th spin obtained from the $i$-th simulation.
Let us calculate this.
```python
print(sum([abs(sum(s))/len(s) for s in samples])/solver.num_samples)
```
You should get a value almost equal to 1.
