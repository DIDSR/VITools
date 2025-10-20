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

   print(my_phantom)

Using the Scanner
-----------------

The `Scanner` class takes a `Phantom` and simulates a CT scan, generating projection data and a reconstructed image.

First, you need to instantiate a `Phantom`. For this example, we will use a phantom from the available phantom plugins:

.. code-block:: python

   from VITools.study import get_available_phantoms

   # Get a list of available phantoms
   available_phantoms = get_available_phantoms()
   print(f"Available phantoms: {list(available_phantoms.keys())}")

   # Instantiate a phantom
   phantom_class = available_phantoms['torso']
   phantom = phantom_class()

Now, we can pass the `Phantom` object to the `Scanner`:

.. code-block:: python

   from VITools.scanner import Scanner

   # Create a Scanner instance
   scanner = Scanner(phantom=phantom, scanner_model='Scanner_Default')

   # Run a scan with specified parameters
   scanner.run_scan(mA=250, kVp=120, pitch=1.0)

After the scan is complete, you can run a reconstruction:

.. code-block:: python

   # Run the reconstruction
   scanner.run_recon(kernel='soft', fov=300)

   # The reconstructed image is now available in the `recon` attribute
   reconstructed_image = scanner.recon
   print(f"Reconstructed image shape: {reconstructed_image.shape}")

You can also save the reconstructed image to a DICOM series:

.. code-block:: python

   # Write the output to DICOM files
   scanner.write_to_dicom('output_dicom/my_scan.dcm')

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
       phantom='torso',
       scanner_model='Scanner_Default',
       kVp=120,
       mA=200,
       pitch=1.0,
       recon_kernel='standard',
       output_directory='my_study_results'
   )

   # You can now run all the scans defined in the study
   study.run_all(parallel=False)

This provides a basic overview of how to use the `VITools` library. For more advanced usage and details on the available classes and functions, please refer to the API documentation.
