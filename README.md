# Tattva

[![PyPI](https://img.shields.io/pypi/v/tattva.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/tattva.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/tattva)][python version]
[![License](https://img.shields.io/pypi/l/tattva)][license]

[![Read the documentation at https://tattva.readthedocs.io/](https://img.shields.io/readthedocs/tattva/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/adityaprakash-work/tattva/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/adityaprakash-work/tattva/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/tattva/
[status]: https://pypi.org/project/tattva/
[python version]: https://pypi.org/project/tattva
[read the docs]: https://tattva.readthedocs.io/
[tests]: https://github.com/adityaprakash-work/tattva/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/adityaprakash-work/tattva
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

- Supports arbitrary dimensions and channel numbers for cellular automata simulations
- Utilizes Jax for GPU acceleration to speed up simulations
- Supports FFT convolution for larger simulations
- Provides an intuitive interface for setting up and running simulations
- Includes a variety of pre-built rules and configurations for easy experimentation
- Offers the ability to customize rules and configurations for more advanced users
- Supports visualization of simulations in real-time or via saved images and videos
- Includes tools for analyzing simulation data and generating statistics
- Provides extensive documentation and a community forum for support and collaboration

## Requirements

- Python 3.x
- Jax
- NumPy
- SciPy

## Installation

You can install _Tattva_ via [pip] from [PyPI]:

```console
$ pip install tattva
```
or alternatively, 
```console
$ pip install git+https://github.com/adityaprakash-work/tattva.git
```
## Usage

Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [Apache 2.0 license][license],
_Tattva_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/adityaprakash-work/tattva/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/adityaprakash-work/tattva/blob/main/LICENSE
[contributor guide]: https://github.com/adityaprakash-work/tattva/blob/main/CONTRIBUTING.md
[command-line reference]: https://tattva.readthedocs.io/en/latest/usage.html
