# ACCV_FCIL3D

## Foundation Model-Powered 3D Few-Shot Class Incremental Learning via Training-free Adaptor

This repository contains the code for our proposed method to address Few-Shot Class Incremental Learning (FSCIL) in 3D point cloud environments. Our approach leverages a foundational 3D model and introduces a novel training-free adaptation strategy that mitigates the issues of forgetting and overfitting, enabling the model to learn new tasks without additional training.

## Overview

Recent advances in deep learning for processing point clouds have highlighted the significance of Few-Shot Class Incremental Learning (FSCIL). Our method employs a dual-cache system, utilizing previous test samples based on confidence scores to prevent forgetting and incorporating few-shot samples from novel tasks to enhance generalization.

## Key Features

- **Foundation Model**: Utilizes a pre-trained 3D foundation model for improved performance on incremental tasks.
- **Training-free Adaptation**: Eliminates the need for fine-tuning when learning new classes, reducing the risk of catastrophic forgetting.
- **Dual-cache System**: Combines base task and novel task caches to balance performance across both old and new classes.

## Datasets

We evaluate our method on several datasets, including:

- ModelNet
- ShapeNet
- ScanObjectNN
- CO3D

Refer to the [Datasets](https://github.com/townim-faisal/FSCIL-3D/blob/main/data/dataset/README.md) section for information on how to set up these datasets.

## Results

Our experiments demonstrate that the proposed method outperforms existing FSCIL methods across multiple cross-dataset settings. See the paper for detailed results and analysis.

## Dependencies

To run the code in this repository, you'll need the following libraries and tools installed:

- Python 3.6 or higher
- PyTorch 1.8.0 or higher
- NumPy
- SciPy
- scikit-learn
- Open3D
- Matplotlib

You can install the required dependencies using pip:

```bash
pip install torch torchvision numpy scipy scikit-learn open3d matplotlib

## Citation

If you use this code or methods in your research, please consider citing our paper:

```bibtex
@inproceedings{ahmadi2024foundation,
  title={Foundation Model-Powered 3D Few-Shot Class Incremental Learning via Training-free Adaptor},
  author={Ahmadi, Sahar and Cheraghian, Ali and Ramasinghe, Sameera and Chowdhury, Townim Faisal and Saberi, M and Petersson, Lars},
  booktitle={Proceedings of the ACCV 2024},
  year={2024},
  url={https://doi.org/your_doi_link_here}
}
