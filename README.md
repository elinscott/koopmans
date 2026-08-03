<!--
<p align="center">
  <img src="https://github.com/elinscott/koopmans/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  koopmans
</h1>

<p align="center">
    <a href="https://github.com/elinscott/koopmans/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/elinscott/koopmans/actions/workflows/tests.yml/badge.svg" /></a>
    <a href="https://pypi.org/project/koopmans">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/koopmans" /></a>
    <a href="https://pypi.org/project/koopmans">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/koopmans" /></a>
    <a href="https://github.com/elinscott/koopmans/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/koopmans" /></a>
    <a href='https://koopmans.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/koopmans/badge/?version=latest' alt='Documentation Status' /></a>
    <a href="https://codecov.io/gh/elinscott/koopmans/branch/main">
        <img src="https://codecov.io/gh/elinscott/koopmans/branch/main/graph/badge.svg" alt="Codecov status" /></a>
    <a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;"></a>
    <a href="https://github.com/elinscott/koopmans/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/></a>
</p>

Koopmans spectral functional calculations with [Quantum ESPRESSO](https://www.quantum-espresso.org/).

Koopmans functionals give you orbital energies that can be compared against a photoemission spectrum — ionization potentials, electron affinities and band structures — at a cost close to that of density-functional theory. Obtaining them takes a chain of calculations: a ground state, a set of localized variational orbitals, a screening parameter for each of them, and a final band structure. `koopmans` takes an input file describing your system and runs that chain for you.

This package is the successor to the original, [ASE](https://wiki.fysik.dtu.dk/ase/)-based [`koopmans`](https://github.com/epfl-theos/koopmans), which it will replace. The physics is the same and the input files are close to unchanged; what is new is that the calculations are run and recorded by [AiiDA](https://www.aiida.net/), so a workflow that stops halfway can be picked up where it left off, and every number keeps a record of the calculation that produced it.

## Thirty seconds

Write an input file describing the system and what to do with it. This one asks for the KI ionization potential and electron affinity of an ozone molecule:

```json
{
    "workflow": {
        "task": "singlepoint",
        "correction": "ki",
        "screening_method": "dscf",
        "init_orbitals": "kohn-sham",
        "alpha_numsteps": 1,
        "pseudo_library": "SG15/1.2/PBE/SR"
    },
    "atoms": {
        "cell_parameters": {
            "vectors": [[14.1738, 0.0, 0.0],
                        [0.0, 12.0, 0.0],
                        [0.0, 0.0, 12.66]],
            "units": "angstrom",
            "periodic": false
        },
        "atomic_positions": {
            "units": "angstrom",
            "positions": [
                ["O", 7.0869, 6.0, 5.89],
                ["O", 8.1738, 6.0, 6.55],
                ["O", 6.0, 6.0, 6.55]
            ]
        }
    },
    "calculator_parameters": {
        "ecutwfc": 65.0,
        "nbnd": 10,
        "kcp": {
            "system": {
                "ecutrho": 260.0
            }
        }
    }
}
```

Run it:

```console
$ koopmans run ozone.json
```

The individual calculations are dispatched as the workflow works out what it needs, and their progress is reported as they finish. At the end you get a directory named after the input file — here `ozone/` — holding one numbered folder per calculation, each with its own `inputs/` and `outputs/`. The ionization potential and electron affinity of ozone are the negative of the highest occupied and the lowest unoccupied orbital energy of the final KI calculation.

## Installation

`koopmans` needs Python 3.12 or 3.13, and a Quantum ESPRESSO installation on your `PATH`. Install the package, then set up the calculation engine:

```console
$ uv pip install git+https://github.com/elinscott/koopmans.git
$ koopmans install
```

See the [installation instructions](https://koopmans.readthedocs.io/en/latest/installation.html) for the details, including which Quantum ESPRESSO executables each kind of calculation needs.

## Documentation

Tutorials, the input file reference, and a short introduction to the theory are at [koopmans.readthedocs.io](https://koopmans.readthedocs.io).

## Citation

If you use this code, please cite

> E. B. Linscott, N. Colonna, R. De Gennaro, N. L. Nguyen, G. Borghi, A. Ferretti, I. Dabo and N. Marzari, *koopmans: An Open-Source Package for Accurately and Efficiently Predicting Spectral Properties with Koopmans Functionals*, J. Chem. Theory Comput. **19**, 7097 (2023). [doi:10.1021/acs.jctc.3c00652](https://doi.org/10.1021/acs.jctc.3c00652)

## Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See [CONTRIBUTING.md](https://github.com/elinscott/koopmans/blob/main/.github/CONTRIBUTING.md) for more information on getting involved.

## License

The code in this package is licensed under the MIT License.

## For developers

<details>
  <summary>See developer instructions</summary>

### Development installation

```console
$ git clone git+https://github.com/elinscott/koopmans.git
$ cd koopmans
$ uv pip install -e .
```

### Pre-commit

You can optionally use [pre-commit](https://pre-commit.com) to run the code quality checks on each commit:

```console
$ uvx pre-commit install
```

### Testing

Install `tox` with `uv tool install tox --with tox-uv`, then run the test suite:

```console
$ tox -e py
```

The same tests run on every push in a [GitHub Action](https://github.com/elinscott/koopmans/actions?query=workflow%3ATests).

### Building the documentation

```console
$ tox -e docs
$ open docs/build/html/index.html
```

Sphinx extensions belong in the `docs` dependency group in [`pyproject.toml`](pyproject.toml) and in the `extensions` list in [`docs/source/conf.py`](docs/source/conf.py). `tox -e docs-test` rebuilds in an isolated environment with warnings treated as errors, which is what CI and [ReadTheDocs](https://readthedocs.io) do.

</details>

## For maintainers

<details>
  <summary>See maintainer instructions</summary>

### Making a release

With the package installed in development mode and `tox` available, run:

```console
$ tox -e finish
```

This strips the `-dev` suffix from the version in `pyproject.toml`, `src/koopmans/version.py` and [`docs/source/conf.py`](docs/source/conf.py), builds an archive and a wheel with `uv build`, uploads to PyPI with `uv publish`, pushes to GitHub, and bumps to the next patch version. Uploading needs a PyPI API token registered with `keyring`:

```console
$ uv tool install keyring
$ keyring set https://upload.pypi.org/legacy/ __token__
```

Then draft the release at https://github.com/elinscott/koopmans/releases/new, selecting the tag that `tox -e finish` created.

</details>
