Welcome to VITools's documentation!
===================================

.. image:: https://github.com/DIDSR/virtual-imaging-trials/actions/workflows/build.yml/badge.svg
   :target: https://github.com/DIDSR/virtual-imaging-trials/actions/workflows/build.yml
   :alt: Build Status

.. image:: https://img.shields.io/pypi/v/virtual-imaging-trials.svg
   :target: https://pypi.org/project/virtual-imaging-trials/
   :alt: PyPI

.. image:: https://img.shields.io/conda/vn/conda-forge/virtual-imaging-trials.svg
   :target: https://anaconda.org/conda-forge/virtual-imaging-trials
   :alt: Conda-Forge

.. image:: https://img.shields.io/pypi/l/virtual-imaging-trials.svg
   :target: https://github.com/DIDSR/virtual-imaging-trials/blob/main/LICENSE
   :alt: License

Tools for running virtual imaging trials.

Installation
------------
.. code-block::

   pip install virtual-imaging-trials

Core Concepts
-------------
A virtual imaging trial is composed of three primary components:
1. a `Phantom`
2. a `Scanner`
3. a `Study`

The `Phantom` class is a container for a numpy array and some metadata, like the patient's age and the voxel spacings.
A `Scanner` takes a `Phantom` and simulates a CT scan, generating projection data and a reconstructed image.
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
