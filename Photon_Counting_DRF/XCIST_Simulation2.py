#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 14:26:11 2025

@author: sriharsha.marupudi
"""


import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
from gecatsim.pyfiles.CommonTools import *
from gecatsim.pyfiles.CommonTools import my_path

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os


my_path.add_search_path("/home/sriharsha.marupudi/xcist/gecatsim/examples/cfg/")


# Load Geant4 detector response (.mat)

mat_file = "/gpfs_projects/sriharsha.marupudi/PCD/Harsha/2025/Geant4/PC_spectral_response_CZT_detector_response_data_ideal_PS0.1_SP_0.75.mat"
# mat_file = "/gpfs_projects/sriharsha.marupudi/PCD/Harsha/2025/Geant4/PC_spectral_response_CZT0.25x0.25x1.6_G4.mat"
data = scipy.io.loadmat(mat_file)

# Extract variables
Dvec0 = data["Dvec0"].squeeze()          # detector response bins (len = 104)
Evec0 = data["Evec0"].squeeze()          # incident energies
res_mat0 = data["res_mat"]               # shape (160, 104)


if res_mat0.shape[1] == 104:
    # Add one more column (duplicate the last column or use zeros)
    extra_column = res_mat0[:, -1:].copy()  # duplicate last column
    res_mat0 = np.hstack([res_mat0, extra_column])
    
    # Also extend Dvec0
    Dvec0 = np.append(Dvec0, Dvec0[-1] + (Dvec0[-1] - Dvec0[-2]))

print("Loaded response matrix:")
print("  res_mat0 shape =", res_mat0.shape)
print("  len(Dvec0)     =", len(Dvec0))
print("  len(Evec0)     =", len(Evec0))

plt.figure(figsize=(8, 6))
plt.imshow(
    np.log10(res_mat0),             # log scale for better contrast
    aspect='auto',
    origin='lower',
    extent=[Dvec0[0], Dvec0[-1], Evec0[0], Evec0[-1]],
    cmap='viridis'
)
cbar = plt.colorbar()
cbar.set_label('Log10(Probability Density)')
plt.xlabel('Detected Energy (keV)')
plt.ylabel('Incident Energy (keV)')
plt.title('Detector Response Function (Central Pixel) - Log Scale')
plt.show()


# Initialize CatSim

ct = xc.CatSim("Scanner_PCCT", "Phantom_Sample",
               "Protocol_Sample_axial", "Physics_Sample")

# Detector setup
ct.scanner.detectorMaterial = "CZT"                # detector sensor material
ct.scanner.detectorDepth = 0.75                     # sensor depth (mm)
ct.scanner.detectionCallback = "Detection_PC"      # detection model
ct.scanner.detectionResponseFilename = os.path.basename(mat_file)  # use same file



ct.scanner.detectorBinThreshold = list(Dvec0)  # length 105
ct.scanner.detectorSumBins = 0                     # 0: output per bin; 1: summed


# Protocol setup

ct.resultsName = "test"
ct.protocol.viewsPerRotation = 500
ct.protocol.viewCount = ct.protocol.viewsPerRotation
ct.protocol.stopViewId = ct.protocol.viewCount - 1
ct.protocol.mA = 500
ct.protocol.spectrumFilename = "tungsten_tar7.0_120_filt.dat"

# Physics setup
ct.physics.energyCount = 120

# Filtration
ct.protocol.bowtie = "large.txt"
ct.protocol.flatFilter = ['Al', 3.0]  # Aluminium filtration

# Detector geometry
ct.scanner.detectorRowsPerMod = 2
ct.scanner.detectorRowCount = ct.scanner.detectorRowsPerMod

# Reconstruction
ct.recon.fov = 300.0
ct.recon.sliceCount = 1
ct.recon.sliceThickness = 0.1421


# Run simulation

if not ct.scanner.detectorSumBins:
    ct.do_prep = 0

ct.run_all()  # run scans


# Reconstruction (only if bins are summed)

if ct.scanner.detectorSumBins == 1:
    ct.do_Recon = 1
    recon.recon(ct)



if ct.scanner.detectorSumBins == 1:
    imgFname = "%s_%dx%dx%d.raw" % (
        ct.resultsName,
        ct.recon.imageSize,
        ct.recon.imageSize,
        ct.recon.sliceCount
    )
    img = xc.rawread(imgFname,
                     [ct.recon.sliceCount,
                      ct.recon.imageSize,
                      ct.recon.imageSize],
                     'float')
    plt.imshow(img[0, :, :], cmap='gray', vmin=-200, vmax=200)
    plt.show()
else:
    scanFname = "%s.air" % ct.resultsName

    # Number of bins is thresholds - 1
    nBin = len(ct.scanner.detectorBinThreshold) - 1

    # Read air scan [row, col, bin]
    air = xc.rawread(
        scanFname,
        [ct.scanner.detectorRowCount,
         ct.scanner.detectorColCount,
         nBin],
        'float'
    )

    # Plot counts for first detector row across columns (each line = a bin)
    plt.figure(figsize=(10, 6))
    for b in range(6):
        plt.plot(air[0, :, b], label=f"Bin {b+1}")
    plt.xlabel("Detector column")
    plt.ylabel("Counts")
    plt.title("Air scan across detector columns (row 0)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()

    # Plot spectrum at central detector column
    center_col = int(ct.scanner.detectorColCount / 2)
    plt.figure()
    plt.plot(np.arange(nBin), air[0, center_col, :], marker='o')
    plt.xlabel("Energy bin index")
    plt.ylabel("Counts")
    plt.title(f"Spectrum at detector column {center_col}")
    plt.show()
