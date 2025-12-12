Usage
=====

This page provides a guide on how to use the `VITools` library for running virtual imaging trials. We will cover the core components: `Phantom`, `Scanner`, and `Study`.

Creating a Phantom
------------------

The `Phantom` class is a container for a `numpy` array and some metadata, such as the patient's age and voxel spacings. Here's how to create a `Phantom` object:

.. code-block:: python

   import numpy as np
   from VITools.phantom import Phantom
   # Create a simple phantom from a numpy array
   image_array = np.zeros((10, 20, 30))
   my_phantom = Phantom(img=image_array, spacings=(1.0, 0.5, 0.5))

>>> print(my_phantom)
Phantom Class: Phantom
Age (years): 0
Shape (voxels): (10, 20, 30)
Size (mm): (10.0, 10.0, 15.0)

Using the Scanner
-----------------

The `Scanner` class takes a `Phantom` and simulates a CT scan, generating projection data and a reconstructed image.

First, you need to instantiate a `Phantom`. For this example, we will use a phantom from the available phantom plugins:

.. code-block:: python

   from VITools.study import get_available_phantoms
   # Get a list of available phantoms
   available_phantoms = get_available_phantoms()

>>> print(f"Available phantoms: {list(available_phantoms.keys())}")
Available phantoms: ['Water Phantom']

Water Phantom comes builtin by default but other phantoms can be extend by installing plugins or defining custom phantoms.
See `src/VITools/examples.py <../../../src/VITools/examples.py>`_ for an example.

.. code-block:: python

   phantom_class = available_phantoms['Water Phantom']
   phantom = phantom_class()

>>> phantom
Phantom Class: WaterPhantom
Age (years): 0
Shape (voxels): (100, 100, 100)
Size (mm): (100.0, 200.0, 200.0)

Now, we can pass the `Phantom` object to the `Scanner`:

.. code-block:: python

   from VITools.scanner import Scanner
   scanner = Scanner(phantom=phantom, scanner_model='Scanner_Default')

>>> scanner
Initializing Scanner object...
----------
Scanner default_series
Scanner: Scanner_Default
Simulation Platform: CATSIM

.. code-block:: python

   scanner.run_scan(mA=250, kVp=120, views=100, startZ=-4, endZ=4)

After the scan is complete, you can run a reconstruction:

.. code-block:: python

   scanner.run_recon(kernel='soft', fov=300)
   reconstructed_image = scanner.recon

>>> print(f"Reconstructed image shape: {reconstructed_image.shape}")
Reconstructed image shape: (7, 512, 512)

You can also save the reconstructed image to a DICOM series:

>>> scanner.write_to_dicom('output_dicom/my_scan.dcm')
[PosixPath('output_dicom/my_scan_000.dcm'),
 PosixPath('output_dicom/my_scan_001.dcm'),
 PosixPath('output_dicom/my_scan_002.dcm'),
 PosixPath('output_dicom/my_scan_003.dcm'),
 PosixPath('output_dicom/my_scan_004.dcm'),
 PosixPath('output_dicom/my_scan_005.dcm'),
 PosixPath('output_dicom/my_scan_006.dcm')]

Setting up a Study
------------------

The `Study` class manages a series of simulations, allowing you to vary parameters and run them in a structured way.

You can create a study and add a scan like this:

.. code-block:: python

   from VITools.study import Study

   # Create a Study instance
   study = Study()

   # Append a new scan to the study
   study.append(
       phantom='Water Phantom',
       scanner_model='Scanner_Default',
       kVp=120,
       mA=200,
       pitch=0.0,
       views=100,
       scan_coverage=[-4, 5],
       recon_kernel='standard',
       output_directory='my_study_results'
   )

   # You can now run all the scans defined in the study
   study.run_all(parallel=False)

Command Line Interface
----------------------

VITools provides a command-line interface (CLI) to easily run simulations defined in a CSV file. This is particularly useful for batch processing or running studies on remote servers.

The `generate` command takes a CSV file path as an argument and executes the study plan defined within it.

.. code-block:: bash

   generate example_study_plan.csv

The input CSV file must contain specific columns that define the parameters for each scan. The required columns are:

- **case_id**: A unique identifier for the case (e.g., 'case_0001').
- **phantom**: The name of the phantom to use (must be a registered phantom).
- **scanner_model**: The name of the scanner model (e.g., 'Scanner_Default').
- **kVp**: The tube voltage in kV.
- **mA**: The tube current in mA.
- **views**: The number of views per rotation.
- **pitch**: The helical pitch.
- **scan_coverage**: The start and end Z positions of the scan (e.g., '[-50, 50]' or 'dynamic').
- **recon_kernel**: The reconstruction kernel (e.g., 'standard', 'soft').
- **slice_thickness**: The slice thickness in mm.
- **slice_increment**: The slice increment in mm.
- **fov**: The Field of View in mm.
- **output_directory**: The directory where the results will be saved.
- **remove_raw**: (Optional) Boolean indicating whether to remove raw data after reconstruction.

**Sample CSV Content**

Below is an example of a valid CSV file content using the built-in 'Water Phantom':

.. csv-table:: Sample Study Plan CSV
   :file: example_study_plan.csv
   :header-rows: 1
   :widths: auto

This provides a basic overview of how to use the `VITools` library. For more advanced usage and details on the available classes and functions, please refer to the API documentation.
