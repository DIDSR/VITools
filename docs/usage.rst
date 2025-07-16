Usage
=====

Intended Purpose
----------------

Tools for running virtual imaging trials, including object-oriented wrappers for the `XCIST CT Simulation framework <https://github.com/xcist/main/tree/master>`_

Low level example:

.. code-block:: python

    from VITools import Phantom, Scanner
                            from utils import create_circle_phantom
                            img = create_circle_phantom()
                            spacings = (200, 1, 1)
    phantom = Phantom(img, spacings)
    scanner = Scanner(phantom)
    scanner.run_scan()
    scanner.run_recon()


High level example:

.. code-block:: python

    from VITools import Study
    study = Study()
    study.append(...)
    study.run_all()