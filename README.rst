VITools: tools for conducting virtual imaging trials
====================================================

Tools for running virtual imaging trials, including object oriented wrappers for the `XCIST CT Simulation framework <https://github.com/xcist>`_

.. image:: assets/VITools_class_diagram.png
        :width: 800
        :align: center

Virtual Imaging Tools (VITools) provides basic building blocks for designing and running virtual imaging trials:

.. code-block:: python

        from VITools import Phantom, Scanner, Study
        phantom = Phantom(img, spacings)
        scanner = Scanner(phantom)
        study = Study(scanner)
        study.run_study()

1. `Phantom <https://github.com/DIDSR/VITools/blob/master/src/VITools/phantoms.py#L58-L71>`_: defines the subject to be imaged. Parameterized by a voxel array `img` and voxel size `spacings`.
2. `Scanner <http://github.com/DIDSR/VITools/blob/master/src/VITools/image_acquisition.py#L117-L152>`_: defines the imaging device. Parameterized by geometry, source, and detector characteristics defined in configuration files. Initialized with a `Phantom`.
3. `Study <https://github.com/DIDSR/VITools/blob/master/src/VITools/study.py>`_ defines the study to be simulated and organizes metadata. Initialized by a `Scanner`
4. Hooks and subclassing: New phantoms, scanners, and studys can be extended by subclassing or providing hook implementations. See Repositories using `VITools` for examples

Installation
------------

.. code-block:: bash

        pip install git+https://github.com/DIDSR/VITools.git

Repositories using `VITools`
---------------------------- 

- `InSilicoICH <https://github.com/DIDSR/InSilicoICH>`_ for generating synthetic non contrast CT datasets of intracranial hemorrhage (ICH)
- `PedSilicoLVO <https://github.com/brandonjnelsonFDA/PedSilicoLVO>`_ for generating synthetic large vessel occlusion (LVO) non contrast CT datasets
- `PedSilicoAbdomen <https://github.com/DIDSR/PedSilicoAbdomen>`_ for generating synthetic abdominal non contrast CT datasets of liver metastases
- `InSilicoGUI <https://github.com/DIDSR/InSilicoGUI>`_ Provides a graphical user interface to the phantoms and imaging simulations
