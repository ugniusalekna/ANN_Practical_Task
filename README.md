# Blood Flow Reconstruction using Physics-Informed Neural Networks (PINNs)
A research project for the course Artificial Neural Networks


## Introduction
  The main task of this project is to employ Physics-Informed Neural Networks (PINNs) to reconstruct
the flow field of blood from limited, noisy data. This is particularly valuable in situations where
obtaining high-quality blood flow measurements is challenging due to low Signal-to-Noise Ratio (SNR).
The project‘s goal is to combine readily available noisy data with the governing physical laws, enabling
the reconstruction of both the flow field and the boundary shape using PINNs.

## Description

**Synthetic Data Generation:** The synthetic fluid flow data is obtained by finite-element method (FEM) based
numerical simulations, performed using the _FEniCS Project_ - an open-source computing library for solving PDEs.
4 different types of vessel geometries were created - a healthy vessel, a vessel with a branch, a vessel with an
isthmus, and a vessel with an aneurysm - each parameterized, so that many different meshes could be generated. 

For simulating fluid dynamics within generated meshes, the incompressible Navier-Stokes equations were solved using 
a modified Chorin’s projection method, called the incremental pressure correction scheme. To make the simulated fields
applicable for training, interpolation was executed, where the simulated fluid flow data was mapped onto a high-resolution square grid.

To replicate MRV data, the resolution of the true velocity data was reduced by half, and Gaussian noise was added. 
Each sample was augmented by flipping it across the horizontal and vertical axes, resulting in a total of 9656 samples.

