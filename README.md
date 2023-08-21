# CVSX
Cardiovascular system models in JAX with Diffrax

## Introduction

This repository contains code for the paper _"A variable heart rate multi-compartmental coupled model of the cardiovascular and respiratory systems"_, submitted to Journal of the Royal Society Interface. The codebase is based on https://github.com/slishak/MedicalTimeSeriesPrediction, which was written for a previous MSc thesis; the main difference is that the implementation is now in JAX rather than PyTorch.

The work contains Python implementations of Smith's inertial/non-cardiovascular models [[1]](#references) and Jallon's heart-lung model [[2]](#references). Parameters from Paeme [[3]](#references) are also used. Three novel contributions are introduced in this work:
1. Support for variable heart-rate in the form of a continuous function of time
2. Stabilisation of Jallon's respiratory model
3. Support for autodifferentiation through backpropagation (which is intended to enable more efficient parameter estimation)

## Installation

Python 3.9+ is recommended for this work. JAX does not need to be compiled for the GPU; in fact, to run the basic examples, it will be faster on the CPU.

We suggest the use of a virtual environment (e.g. `conda`). The `pyproject.toml` file describes the dependencies. From within your virtual environment, you can run the following to install the package:
`pip install -e .`

Following this, you can test the installation by running the following command, which will generate all plots in the paper:
`python examples/generate_paper_plots.py`

## Architecture

The module `cvsx` is a library for creating models of the cardiovascular system. It is laid out as follows:
- `cvs.cardiac_drivers` defines a variety of functions $e(t)$ which simulate time-varying elasticity of the ventricles. They also allow the definition of a function $e(s \mod 1)$ where $\frac{ds}{dt} = {\rm HR}(t)$, which permits simulation of variable heart rate.
- `cvs.components` defines basic componends of the cardiovascular system, for example `PressureVolume` compartments, `BloodVessel`, `Valve` etc. There are also some other more exotic valve configurations which are not covered in the paper.
- `cvs.models` composes the components into full models that can be integrated with Diffrax. These are `SmithCVS` and `JallonHeartLungs`.
- `cvs.parameters` defines parameterisations for the above models, in particular a function `build_parameter_tree` creates a parameterisation which can be used to instantiate a model.
- `cvs.respiratory` contains the components required for simulation of the respiratory system, `RespiratoryPatternGenerator` and `PassiveRespiratorySystem`.
- `cvs.unit_conversions` is a utility for converting between common units.

The function `main` in `examples/simulate.py` demonstrates parameterising and instantiating a model, and then simulating it with `diffrax`. There is a lot of extra functionality (e.g. parameter estimation, regurgitating valves) which is not yet officially supported, and will be modified in the future - please bear with us! 

## References
1. Bram W Smith et al. _‘Minimal haemodynamic system model including ventricular interaction and valve dynamics’_. en. In: Med. Eng. Phys. 26.2 (Mar. 2004), pp. 131–139.
2. Julie Fontecave Jallon et al. _‘A model of mechanical interactions between heart and lungs’_. In: Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 367.1908 (2009), pp. 4741–4757. doi: 10.1098/rsta.2009.0137. eprint: url: https://royalsocietypublishing.org/doi/abs/10.1098/rsta.2009.0137.
3. Sabine Paeme et al. _‘Mathematical multi-scale model of the cardiovascular system including mitral valve dynamics. Application to ischemic mitral insufficiency’_. In: BioMedical Engineering OnLine 10.1 (Sept. 2011), p. 86. issn: 1475-925X. doi: 10.1186/1475-925X-10-86. url: https://doi.org/10.1186/1475-925X-10-86.
