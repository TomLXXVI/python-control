# A(nother) Python package about Control Theory.

This package is inspired by the book **Control Systems Engineering**, Eight 
Edition (EMEA Edition) written by **Norman S. Nise**. 

The package is mainly built upon  [Sympy](https://www.sympy.org/en/index.html) 
and [Python Control Systems Library](https://github.com/python-control/python-control). 
But some other fundamental third-party packages are also used in this package. 
All third-party packages that put this package together, are listed in the 
`requirements.txt` and `pyproject.toml` files.

The package is composed of three main subpackages. 
- Subpackage `core` defines the core functions and classes for control 
engineering, like `TransferFunction`, `StateSpace`, `SignalFlowGraph`, 
`FeedbackSystem`, `RootLocus`, `FrequencyResponse`,... 
- Subpackage `design` contains the functions to design the compensator or 
controller of feedback systems using root-locus, frequency-response, or 
state-space design techniques.
- Subpackage `modeling` contains classes for modeling electrical circuits, 
translational and rotational mechanical systems, and thermal systems.

In the `docs` folder, examples (Jupyter notebooks and Python scripts) are 
included that demonstrate the usage of the package. The notebooks and scripts in
the folder `book_examples` solve (almost all of) the examples in the above 
mentioned book (however, the images in the examples of the book haven't been 
included to the repository). The file names of the Jupyter notebooks and scripts
refer to the number of the chapters in the book. By going through these examples
and with the aid of the docstrings in the source code, one should be able to 
explore the capabilities of the package quite easily.

At this stage, the package is limited to single-input-single-output (SISO) and 
linear time-invariant (LTI) systems. However, this package is still under 
construction.
