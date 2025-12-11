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
   :target: https://github.com/DIDSR/VITools/blob/master/LICENSE
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

VITools is built around three core components that represent the key elements of a virtual imaging trial:

1.  **`Phantom <https://github.com/DIDSR/VITools/blob/master/src/VITools/phantom.py>`_**:
    Represents the subject or object to be imaged. A phantom is defined by a 3D NumPy array of CT numbers (in Hounsfield Units) and the corresponding voxel spacings.

2.  **`Scanner <https://github.com/DIDSR/VITools/blob/master/src/VITools/scanner.py>`_**:
    Represents the imaging device. It wraps the XCIST simulator and is configured with a specific `Phantom`. The scanner's behavior is defined by XCIST configuration files that specify its geometry, source, and detector characteristics.

3.  **`Study <https://github.com/DIDSR/VITools/blob/master/src/VITools/study.py>`_**:
    Manages a collection of scans. This class is used to design large-scale experiments, where you might want to vary parameters like phantom type, scanner model, mA, or kVp across many simulations. It can generate study plans and execute them in series or in parallel.

Basic Usage
-----------

Here is a complete example of how to create a simple phantom, simulate a scan, and save the result as a DICOM file.

.. code-block:: python

    import numpy as np
    from VITools import Phantom, Scanner

    # 1. Create a simple phantom
    # A 100x100x100 voxel phantom with a 50x50x50 high-density sphere inside.
    print("Creating a phantom...")
    image_shape = (100, 100, 100)
    img = np.full(image_shape, -1000, dtype=np.int16)  # Air
    center = tuple(s // 2 for s in image_shape)
    radius = 25
    z, x, y = np.ogrid[-center[0]:image_shape[0]-center[0], -center[1]:image_shape[1]-center[1], -center[2]:image_shape[2]-center[2]]
    mask = x*x + y*y + z*z <= radius*radius
    img[mask] = 100  # Set sphere to a value like soft tissue

    # Define voxel spacings in mm (z, x, y)
    spacings = (0.5, 0.5, 0.5)
    phantom = Phantom(img, spacings, patient_name="TestSphere", patientid=1)

    # 2. Initialize the Scanner with the phantom
    # This will prepare the phantom for simulation (voxelization).
    print("Initializing the scanner...")
    scanner = Scanner(phantom, scanner_model="Scanner_Default")

    # 3. Run the scan and reconstruction
    print("Running the simulation...")
    scanner.run_scan(mA=200, kVp=120, views=100)
    scanner.run_recon(fov=250, slice_thickness=1.0)

    # 4. Save the output to DICOM
    print("Writing output to DICOM files...")
    output_dcm_path = "./output/dicom/test_sphere.dcm"
    dcm_files = scanner.write_to_dicom(output_dcm_path)

    print(f"Successfully created {len(dcm_files)} DICOM files in ./output/dicom/")

Interactive Examples
--------------------

For more in-depth, runnable examples of each component, please refer to the following Jupyter notebooks:

*   `01_phantoms.ipynb <https://github.com/DIDSR/VITools/blob/master/notebooks/01_phantoms.ipynb>`_
*   `02_scanners.ipynb <https://github.com/DIDSR/VITools/blob/master/notebooks/02_scanners.ipynb>`_
*   `03_studies.ipynb <https://github.com/DIDSR/VITools/blob/master/notebooks/03_studies.ipynb>`_

Advanced Usage: The `Study` Class
----------------------------------

For more complex experiments, the `Study` class can automate running hundreds or thousands of simulations. You can define a study plan in a CSV file or generate one programmatically.

.. code-block:: python

    from VITools import Study

    # Generate a study plan with 5 different cases
    study_plan = Study.generate_from_distributions(
        phantoms=['MyCustomPhantom'], # Requires a registered custom phantom
        study_count=5,
        output_directory='my_large_study',
        kVp=[100, 120],
        mA=[150, 200, 250]
    )

    # Create a Study object and run all simulations
    # This can run in parallel on a supported cluster (e.g., SGE)
    study = Study(study_plan)
    study.run_all(parallel=True)

Extensibility: Creating Custom Phantoms
---------------------------------------
VITools uses a plugin architecture based on `pluggy` that allows you to create your own phantom generators and make them available to the `Study` class. To create a new phantom, you need to:

1.  Create a new installable Python package.
2.  In your package, create a class that inherits from `VITools.Phantom`.
3.  Register your new phantom class using the `register_phantom_types` hook.

For a detailed example, please refer to one of the repositories using `VITools`.

Repositories using `VITools`
----------------------------

-   `InSilicoICH <https://github.com/DIDSR/InSilicoICH>`_: For generating synthetic non-contrast CT datasets of intracranial hemorrhage (ICH).
-   `PedSilicoLVO <https://github.com/brandonjnelsonFDA/PedSilicoLVO>`_: For generating synthetic large vessel occlusion (LVO) non-contrast CT datasets.
-   `PedSilicoAbdomen <https://github.com/DIDSR/PedSilicoAbdomen>`_: For generating synthetic abdominal non-contrast CT datasets of liver metastases.
-   `InSilicoGUI <https://github.com/DIDSR/InSilicoGUI>`_: Provides a graphical user interface to the phantoms and imaging simulations.


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
