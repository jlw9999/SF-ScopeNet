# SF-ScopeNet: A Stage-Frequency Scope-aware Network for 3D Brain Tumor Segmentation
*Liwei Jin Yanjun Peng JinDong Sun*

<img width="1677" height="848" alt="image" src="https://github.com/user-attachments/assets/19df0b0e-1fa3-4ad4-a894-edc305891e25" />

The remaining model diagrams will be published after the paper is accepted.
Building upon [MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt).

## Environments and Requirements
- Ubuntu 20.04.3 LTS
- 12 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
- RAM 1×1200GB, 320MT/s
- GPU 1 NVIDIA RTX 4090 24G
- CUDA version 12.2
- python version 3.9.7

## Dataset
- MSD [link](http://medicaldecathlon.com/).
- BraTS2021 [link](https://www.synapse.org/Synapse:syn25829067/wiki/610863).
- BraTS2024 [link](https://www.synapse.org/Synapse:syn53708249/wiki/626323).

## Preprocessing

Preprocessing
All MRI images follow a standardized preprocessing pipeline for consistent input representation:

- Intensity Normalization: Applied using nnU-Net default configuration for signal standardization
- Spatial Registration: Ensures anatomical alignment across different scanning protocols
- Voxel Resampling: Images resampled to 1 mm³ isotropic resolution for uniform spatial representation
- 3D Patch Extraction: Uniform voxel patches of 128×128×128 extracted for network input

The preprocessing pipeline ensures data standardization while preserving critical anatomical information for accurate segmentation performance.








