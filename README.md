# PLM-Effector 🚀
**A Hybrid Deep Learning Framework for Accurate Prediction of Bacterial Secreted Proteins**

[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Web Server](https://img.shields.io/badge/Web%20Server-Online-orange.svg)](http://www.mgc.ac.cn/zhengdd/PLM-Effector/)

---

## Overview

Protein secretion is a key process in bacteria, enabling proteins to reach extracellular spaces, other microbes, or host cells. Bacteria employ diverse secretion systems, each with distinct structures, substrates, and biological roles. Effector proteins delivered by these systems manipulate host processes—from immune evasion to cytoskeletal disruption—highlighting the importance of accurate prediction for understanding bacterial pathogenicity.  

We developed **PLM-Effector**, a hybrid framework that combines pre-trained protein language models with deep learning architectures to achieve robust, type-specific prediction of secreted proteins, including **T1SEs, T2SEs, T3SEs, T4SEs, and T6SEs**.  

- Benchmarks multiple embeddings  
- Evaluates both N-terminal and C-terminal regions  
- Identifies the most informative features per secretion system  
- Integrates features through a **two-layer ensemble stacking strategy**  

By leveraging discriminative sequence representations and optimized neural models, PLM-Effector outperforms existing predictors across these secretion types, providing a generalizable, high-performing framework for bacterial secreted protein prediction.
![Workflow diagram](workflow.tif)


---

## Dataset for Training Your Own Models

You can download our [datasets.zip](#) containing training and test sets for each effector type:

| Secretion System | Training Set | Test Set |
|-----------------|--------------|----------|
| T1SE | T1SE_train.fasta | T1SE_test.fasta |
| T2SE | T2SE_train.fasta | T2SE_test.fasta |
| T3SE | T3SE_train.fasta | T3SE_test.fasta |
| T4SE | T4SE_train.fasta | T4SE_test.fasta |
| T6SE | T6SE_train.fasta | T6SE_test.fasta |

> **Note:** This is optional if you want to train your own models. For demo/testing, the example FASTA files are sufficient.

---

## Web Server 🌐

For **online predictions**, visit our web server:  
[PLM-Effector Web Server](http://www.mgc.ac.cn/zhengdd/PLM-Effector/)

---

## Local GPU Usage 💻

PLM-Effector can be run on local GPU servers for **whole-genome scale analyses**.
Click to expand: Setup & Prediction Commands

**Step 1: Create Conda Environment**
```bash
conda env create -f py39_cuda11.3.yml

**Step 2: Activate Environment**
conda activate py39_cuda11.3

**Step 3: Run Predictions**
python run_pipeline.py --usefile_id <example_id> --effector_type <T1SE/T2SE/...>

**Quick Demo**
To quickly test PLM-Effector without downloading the full training dataset, we provide example FASTA files for each effector type in the **tmp** folder.
conda activate py39_cuda11.3
python run_pipeline.py --usefile_id T1SE_example --effector_type T1SE
python run_pipeline.py --usefile_id T2SE_example --effector_type T2SE
python run_pipeline.py --usefile_id T3SE_example --effector_type T3SE
python run_pipeline.py --usefile_id T4SE_example --effector_type T4SE
python run_pipeline.py --usefile_id T6SE_example --effector_type T6SE
**Note**: The pretrained models are large. Before running the demo, please download the models from [http://www.mgc.ac.cn/PLM-Effector/downloads.html](URL):
from Sourcecode fold and place them in the trained_models folder within the PLM-Effector repository.

