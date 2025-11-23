---
title: "NUS Hackers x QRT - Intro to ML Workshop Notes"
date: 2025-03-27
tags:
  - Machine Learning
  - Notes
---

Today I attended NUS Hackers x QRT Workshop: Introduction to Machine Learning.

I decided to share what I learned with my notes. It was a really good experience for both learning and network.

I thought there was pizza but there wasn't, sorry NF. QAQ

## Data

### Types of data

* Structured Data: Key \\(\rightarrow\\) Value, Relational DB, Documents, Graph DB, Vectors, etc.
* Semi-Structured Data: JSON, XML
* Unstructured Data: Videos, Images

### Data Cleaning

* Missing Values
* Duplicates
* Outliers
* Imbalanced Dataset
  * Oversampling (minority)
  * Undersampling (majority)
  * Hybrid
* Data Standardization
  * Z-Score, might break the relation between 2 datasets
* Data Normalization
  * Put all value into an interval e.g. \\([0, 1]\\).

### Data Split

* Train/Validation/Test datasets with proper ratio
* Cross-validation: K-fold, stratified K-fold, leave-one-out
  * Stratified: Split the data following the original distribution

### Data Augmentation 

* Modify the original data, mostly in CV (e.g. flipping images) and NLP (e.g. translating back and forth)
* Increase robustness

### Data Loading

* Batch processing at every stage
* Asynchronous execution
* Prefetching queue hides latency (buffering)
* Longer queue \\(\rightarrow\\) bigger memory footprint

## Models

### Model Layers

* Embedding Layers
  * Function: convert categorical data into vectors
  * Usage: Often used in NLP
* Dense Layers (fully connected)
  * Function: A complete bipartite. Letting the network to learn complex representation by integrating all input features
  * Usage: Final Stage of regressions.
* Conv. Layer
  * Function: apply convolution to datas. Capturing Spatial hierarchies by learning local patterns
  * Usage: Image recognitions
* Pooling Layers
  * Function: Similar to conv. layer. Selecting or calculating a value within a pooling window, retaining the most important feature.
  * Usage: Reduce computation for CNN
* Recurrent Layers
  * Function: Process sequential data by maintaining a state that carries information across time steps
  * Usage: Often involves time series data or sequence, e.g. Language Model and speech recognition.
  * Downside: The model might forget previous data
* Attention Layer
  * Function: To improve recurrent layer. Build a query-key to create inner product to expand their influence. Allow the model to weight the importance of element within same input sequence, capturing long-range dependences
  * Usage: NLP, Transformers. To model relationships between words irrespective of their position
  * [Reference: Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

### Model Architecture

* Dense vs. Sparse (High computation cost vs. Performance)
  * Using different way to store data, just like your trade*off between adjacent matrix and adjacent list.
* Mixture of Experts (Deepseek R1?)
  * Multi-model with each one being expert at one task, route the query into tasks

### Loss Function

* Regression losses
  * Mean Square Error (MSE)
  * Mean Absolute Error (MAE)
  * Root Mean Square Error (RMSE)
  * \\(R^2\\) value: \\(1 - \frac{\sum{(y_i-\hat y)^2}}{\sum{(y_i-\bar y)^2}}\\)
* L1, L2 Regularization
  * To avoid over/underfitting.
  * Penalize high-value, correlated coefficients.
  * Learn more [here](https://www.geeksforgeeks.org/regularization-in-machine-learning/)
  * L1: Least Absolute Shrinkage and Selection Operator (LASSO)
  * L2: Ridge Regression

### Evaluation
* Accuracy(Precision, How many retrieved items are relevant), Recall(Sensitivity, How many relevant item are retrieved)
* Confusion Matrix

| Actual vs. Predict | Position                         | Negative                          |
| ------------------ | -------------------------------- | --------------------------------- |
| Positive           | True Positive(TP)                | False Negative(FN), Type II Error |
| Negative           | False Positive(FP), Type I Error | True Negative(TN)                 |

## Training

### GPU Training

GPU provides many core to perform calculations
Allows multithreading better than CPU
* CUDA Core
* Tensor Core
* Read more [here](https://www.nvidia.com/content/pdf/fermi_white_papers/p.glaskowsky_nvidia%27s_fermi-the_first_complete_gpu_architecture.pdf)

### Parallelism

* Data Parallelism (running datas on different GPU with same model)
  * Frameworks support: PyTorch DDP, TensorFlow distribution strategies, Megatron, DeepSpeed
* Tensor Parallelism (cutting model)
  * Sharding large matrices across devices
* Pipeline Parallelism
  * Micro-batch processing
* Read more [here](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html)

### Fine-tuning

For most people, we prepare our own data with other big companies' data.
Lower Cost!
* Transfer Learning
* Few-shot Learning
* One-shot Learning

### Profiling

* Performance monitoring: Memory Usage, Runtime
* Bottleneck Identification: CPU, GPU, I/O bottlenecks
* Optimization Strategies: Based of profiling result
* Tools: TensorBoard Profiler, PyTorch Profiler, NVIDIA Nsight

## Inference

### Model compression

* Quantization
  * Integer Quantization
  * Float16/BFloat16 Precision
    * Mapping an interval to another interval, compressing the size
  * GPU calculates lower precision data faster
* Knowledge Distillation
  * Teacher-Student Framework
  * Distillation Loss on soft label
* Low-rank Factorization
  * SVD-based factorization

## Application

* Computer Vision
  * Image Classification
  * Object Detection
  * Re-Identification
  * Image Segmentation (Medical)
  * Image Search
* Natural Language Programming
  * Sentiment Analysis
  * Spam Email Detection
  * Tebular Data Query
  * Large Language Models
* Stock Price Prediction
* Fraud Detection in Finance

## OSEMN

Obtain \\(\rightarrow\\) Scrub \\(\rightarrow\\) Explore \\(\rightarrow\\) Model \\(\rightarrow\\) Interpret

## Notebooks

* [CNN and Transfer Learning](https://www.kaggle.com/code/jonaspalucibarbosa/chest-x-ray-pneumonia-cnn-transfer-learning) 
* [Imbalanced Dataset](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)