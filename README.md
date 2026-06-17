# CTPD (Cross-Task Preference Distillation)

## 📋 Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Training Pipeline](#training-pipeline)
  - [Step 1: SFT Training](#step-1-sft-training)
  - [Step 2: DPO Training](#step-2-dpo-training)
  - [Step 3: Weight Generation](#step-3-weight-generation)
  - [Step 4: CTPD Training](#step-5-ctpd-training)
- [Evaluation](#evaluation)
- [File Structure](#file-structure)
- [Usage Examples](#usage-examples)

## 📖 Overview

CTPD implements a preference distillation framework that trains student models to learn from teacher models using aligned span weighting. The pipeline includes:

1. Dataset processing and preparation
2. Supervised Fine-Tuning (SFT) for both student and teacher models
3. Direct Preference Optimization (DPO) training with positive and negative examples
4. Weight generation for aligned spans
5. Final CTPD framework training

## 🛠 Environment Setup

### Training Environment
```bash
conda env create -f train_env.yaml
conda activate ctpd-train
```
### Evaluation Environment
```bash
conda env create -f eval_env.yaml
conda activate ctpd-eval
```

## 📊 Dataset Preparation
Step 1: Process Preference Dataset
Run the Jupyter notebook to get and process the Ultrafeedback dataset processed-preference-dataset.ipynb
Step 2: Generate Aligned Spans and Dummy Weights
Run the notebook to find aligned spans and generate dummy weights: gen-parent-and-dummy-weight.ipynb

## 🚀 Training Pipeline
### Step 1: SFT Training
Train SFT checkpoints for both student and teacher models:

```bash
# Edit std_train_sft.sh to update model paths
bash script/training/std_train_sft.sh
# Edit tea_train_sft.sh to update model paths
bash script/training/tea_train_sft.sh
```
### Step 2: DPO Training
Train positive and negative teacher models using DPO:
```bash
# Update teacher SFT checkpoint path in tea_train_dpo.sh
bash script/training/tea_train_dpo.sh
# Update teacher SFT checkpoint path in tea_train_dpo_negative.sh
bash script/training/tea_train_dpo_negative.sh
```
### Step 3: Weight Generation
Generate aligned span weights for the dataset:
```bash
# Update paths to positive and negative teacher models in weight.sh
bash script/training/weight.sh
```
Data post-processing
```bash
python post_process.py
```
### Step 4: CTPD Training
Final training using the CTPD framework:
```bash
# Update the following paths in CTPD.sh:
# - Student SFT checkpoint path
# - Teacher SFT checkpoint path  
# - Processed data with weights path
bash script/training/CTPD.sh
```

## 📈 Evaluation
Use the evaluation environment and scripts in run_eval forlder to evaluate trained models.

## 📁 File Structure
```
ctpd/
├── README.md
├── train_env.yaml              # Training environment
├── eval_env.yaml               # Evaluation environment
├── script/
│   ├── training/               # Training scripts
│   │   ├── CTPD.sh
│   │   ├── std_train_sft.sh
│   │   ├── tea_train_sft.sh
│   │   ├── tea_train_dpo.sh
│   │   ├── tea_train_dpo_negative.sh
│   │   └── weight.sh
│   └── run_eval/               # Evaluation scripts
├── src/
│   └── prefkd/                 # Main source code
├── baseline/                   # Baseline implementations
│   ├── DSKD-main/
│   └── TIS-DPO-main/
├── mtbench/                    # MT-Bench evaluation
└── temp/                       # Temporary files and notes
```

## 💡 Usage Examples
Quick Start
1. Set up the training environment
2. Process the Ultrafeedback dataset using the provided notebook
3. Train SFT models for student and teacher
4. Train DPO models (positive and negative teachers)
5. Generate weights and post-process data
6. Run CTPD training with the processed data

Configuration Notes
- Model Paths: Update all script files with your specific model paths
- Data Paths: Ensure processed data paths are correctly specified
- Checkpoint Paths: Verify SFT checkpoint paths before DPO training
- Weight Data: Confirm weight-enhanced data paths for CTPD training

🔧 Important Notes

- Always update file paths in shell scripts before execution
- Ensure proper environment activation for each training phase
- Monitor GPU memory usage during training
- Check checkpoint compatibility between training steps
