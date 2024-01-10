# Blood Flow Reconstruction using Physics-Informed Neural Networks (PINNs)
A research project for the course Artificial Neural Networks


## Introduction
  The main task of this project is to employ Physics-Informed Neural Networks (PINNs) to reconstruct
the flow field of blood from limited, noisy data. This is particularly valuable in situations where
obtaining high-quality blood flow measurements is challenging due to low Signal-to-Noise Ratio (SNR).
The project‘s goal is to combine readily available noisy data with the governing physical laws, enabling
the reconstruction of both the flow field and the boundary shape using PINNs.

## Repository Structure

* `code/`: Contains two main categories:
  * `generate_data/`: All files related to synthetic data generation
  * `neural_network/`: All files for PINN implementation
* `data/`: Test datasets since the original dataset is too large to upload
* `logs/`: Training and testing losses logs, model parameters
* `results/`: Selected results of the best training runs 

The original dataset was not uploaded to GitHub but can be accessed via:
https://drive.google.com/drive/folders/1qTgwVh8fgAGMJATJzekZUeViLwCu5wbY?usp=sharing


## Description

**Synthetic Data Generation:** The synthetic fluid flow data is obtained by finite-element method (FEM) based
numerical simulations, performed using the _FEniCS Project_ - an open-source computing library for solving PDEs.
4 different types of vessel geometries were created - a healthy vessel, a vessel with a branch, a vessel with an
isthmus, and a vessel with an aneurysm - each parameterized, so that many different meshes could be generated. 

For simulating fluid dynamics within generated meshes, the incompressible Navier-Stokes equations were solved using 
a modified Chorin’s projection method, also called the incremental pressure correction scheme. To make the simulated fields
applicable for training, interpolation was executed, where the simulated fluid flow data was mapped onto a high-resolution square grid.

To replicate magnetic resonance velocimetry (MRV) data, the resolution of the true velocity data was reduced by half, and Gaussian noise was added. 
Each sample was augmented by flipping it across the horizontal and vertical axes, resulting in a total of 9656 samples.

**Physics-Informed Neural Network:** The network comprises multiple convolutional layers with batch normalization, ReLU activation, 
and dropout layers. The physics-informed part is implemented by incorporating Navier-Stokes equations into the training process.
The loss is a composite of two primary components: data loss and physics-informed loss, weighted with a parameter α.

The spatial derivatives of the predicted velocity field are calculated using the finite difference method. Since pressure is
not directly computed in our model, Navier-Stokes equations are transformed into a vorticity transport equation. The residuals of
continuity and vorticity transport equations are then calculated as a physics-informed loss.
