# Scalable Project 2

## Overview
This project automates various aspects of file management, image generation, classification, and deep learning model training. It’s designed to be scalable, modular, and customizable, making it suitable for complex workflows in machine learning and data processing. 

## Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Download and Compare Files](#1-download-and-compare-files)
  - [2. Image Generation](#2-image-generation)
  - [3. Data Splitting](#3-data-splitting)
  - [4. Model Training](#4-model-training)
  - [5. Classification and Prediction](#5-classification-and-prediction)
  - [6. Output Conversion](#6-output-conversion)
- [Project Components](#project-components)
  - [Access Files Module](#access-files-module)
  - [Image Processing Module](#image-processing-module)
- [Dependencies](#dependencies)

## Project Structure
The main components are organized as follows:
```
scalable_project2/
├── src/
│   ├── access_files/
│   │   ├── compareCsv.py
│   │   └── download_files.py
│   └── image_processing/
│       ├── classify/
│       │   └── classify.py
│       ├── generate_process/
│       │   ├── convert_to_csv.py
│       │   ├── divide.py
│       │   └── generate.py
│       ├── output/
│       │   ├── output.csv
│       │   └── output.txt
│       ├── models/
│       │   └── 24_10_2024_12epoochs_cnn_crossentropy/
│       │       ├── captcha_model.keras.json
│       │       └── captcha_model.keras.keras
│       ├── train/
│       │   ├── train.py
│       │   └── train_helper.py
└── run.sh
```

## Setup
1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd scalable_project2
    ```

2. **Run the setup script**:
   Execute the following command to create a virtual environment, activate it, and install dependencies.
    ```bash
    ./run.sh
    ```

3. **File Configuration**:
   Make sure the required CSV and symbol files (`symbols.txt`, data CSVs) are properly located in the specified directories.

## Usage

### 1. Download and Compare Files
The access_files module contains scripts for handling file downloads and comparisons.
- **download_files.py**: Fetches all necessary files from a CSV list.
- **compareCsv.py**: Identifies missing files and prepares for re-download.

### 2. Image Generation
Use `generate.py` to create a custom set of images with specified width, height, symbol set, and font. This process outputs generated images in `src/data/test`.

### 3. Data Splitting
Run `divide.py` to split generated images into training and validation datasets.

### 4. Model Training
Train the deep learning model using `train.py`, which accepts training/validation directories, model configuration parameters, and outputs a trained model.

### 5. Classification and Prediction
The `classify.py` script utilizes the trained model to classify images in a specified directory and outputs predictions to `output.txt`.

### 6. Output Conversion
Convert prediction results in `output.txt` to CSV format using `convert_to_csv.py`.

---
