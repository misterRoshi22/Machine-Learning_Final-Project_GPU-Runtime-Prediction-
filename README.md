# Prediction of GPU Kernel Performance for Matrix Multiplication

This repository contains the implementation of machine learning models used to predict the computational running time for multiplying two 2048x2048 matrices using a GPU OpenCL SGEMM kernel. The code is based on the research presented in the paper:

**Prediction of the Running Time for Multiplying Two 2048x2048 Matrices using a GPU OpenCL SGEMM Kernel**

## Overview

Matrix multiplication is a fundamental operation in linear algebra, with applications in fields like machine learning, scientific computing, and engineering. This project leverages machine learning to predict the computational performance of a GPU-based SGEMM kernel for matrix multiplication, enabling engineers to evaluate GPU performance efficiently.

### Key Features
- **Dataset**: 241,600 data points with 14 GPU configuration parameters.
- **Machine Learning Models**:
  - Linear Regression
  - Polynomial Regression
  - Decision Trees
  - Random Forests
  - K-Nearest Neighbors (KNN)
- **Ensemble Learning**: Combining Random Forest and KNN models.
- **Performance Metrics**: R², Mean Absolute Error (MAE), and Mean Squared Error (MSE).

## Dataset

The dataset includes the following key parameters:
- **Workgroup Sizes**: MWG, NWG
- **Tiling Sizes**: KWG
- **Local Memory Allocations**: MDIMC, NDIMC, MDIMA, NDIMB
- **Execution Patterns**: STRM, STRN, SA, SB
- **Vector Widths**: VWM, VWN
- **Kernel Shaping**: KWI

The target variable is the logarithmic transformation of the average running time for matrix multiplication tasks.

## Getting Started

### Prerequisites

To run the code, you need:
- Python 3.8 or later
- Libraries:
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn

Install the required dependencies using:
```bash
pip install -r requirements.txt
