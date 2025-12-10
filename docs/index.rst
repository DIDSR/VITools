VITools: Tools for Conducting Virtual Imaging Trials
====================================================

.. image:: https://github.com/DIDSR/VITools/actions/workflows/build.yml/badge.svg
   :target: https://github.com/DIDSR/VITools/actions/workflows/build.yml
   :alt: Build Status

.. image:: https://img.shields.io/pypi/v/VITools.svg
   :target: https://pypi.org/project/VITools/
   :alt: PyPI

.. image:: https://img.shields.io/conda/vn/conda-forge/VITools.svg
   :target: https://anaconda.org/conda-forge/VITools
   :alt: Conda-Forge

.. image:: https://img.shields.io/pypi/l/VITools.svg
   :target: https://github.com/DIDSR/VITools/blob/main/LICENSE
   :alt: License

.. image:: _static/VITools.png
   :width: 400
   :align: center

**VITools** is a Python library designed to simplify the process of conducting virtual imaging trials. It provides high-level, object-oriented wrappers for the `XCIST CT Simulation framework <https://github.com/xcist>`_, making it easier to set up, run, and manage complex imaging simulations.

Whether you are generating synthetic datasets for AI/ML model training, evaluating image reconstruction algorithms, or studying the impact of scanner parameters, VITools provides the building blocks to streamline your workflow.

Key Features
------------
*   **Object-Oriented Interface**: Simplifies complex simulations with intuitive `Phantom`, `Scanner`, and `Study` classes.
*   **Extensible by Design**: Easily add new phantoms via a `pluggy`-based plugin system.
*   **Automated Workflow**: Handles the end-to-end process from phantom definition to DICOM image generation.
*   **Scalable Studies**: The `Study` class enables the management and execution of large-scale experiments, with support for parallel execution on SGE clusters.
*   **Configuration-Based**: Leverages the powerful configuration system of XCIST for detailed control over scanner physics, geometry, and protocols.

Installation
------------

**For Users:**

To install the latest stable version of VITools, you can install directly from the git repository:

.. code-block:: bash

    pip install git+https://github.com/DIDSR/VITools.git

**For Developers:**

If you plan to contribute to VITools or want to install it in an editable mode, follow these steps:

.. code-block:: bash

    # 1. Clone the repository
    git clone https://github.com/DIDSR/VITools.git
    cd VITools

    # 2. Install in editable mode
    pip install -e .

This will install the package and its dependencies, and any changes you make to the source code will be immediately effective.

Core Concepts
-------------
A virtual imaging trial is composed of three primary components:

1. a `Phantom`
2. a `Scanner`
3. a `Study`


Phantom
~~~~~~~
The `Phantom` class is a container for a numpy array and some metadata, like the patient's age and the voxel spacings.

Scanner
~~~~~~~
A `Scanner` takes a `Phantom` and simulates a CT scan, generating projection data and a reconstructed image.

Study
~~~~~
A `Study` takes a `Phantom` and a `Scanner` and runs a series of simulations, varying parameters like the phantom's position, the scanner's settings, and the reconstruction parameters.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
