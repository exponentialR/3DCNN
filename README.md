
# 3D Convolutional Neural Network (3DCNN) for Video Action Prediction

![3DCNN](images/3d-CNN.png)  
*Image generated by DALL·E* *circa 2023.*

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Key Features](#key-features)  
3. [Directory Structure](#directory-structure)  
4. [Technical Overview](#technical-overview)  
    - [Model Architecture](#model-architecture)  
    - [Data Preprocessing](#data-preprocessing)  
5. [Dataset](#dataset)  
6. [Installation and Setup](#installation-and-setup)  
7. [Configuration](#configuration)  
8. [Training and Testing](#training-and-testing)  
9. [Output and Results](#output-and-results)  
10. [Future Enhancements](#future-enhancements)  
11. [Contact](#contact)

---

## Introduction
Recognising human actions in videos is a crucial task in **Computer Vision** and **Machine Learning**, with applications ranging from surveillance and human-computer interaction to sports analysis and autonomous systems. This repository offers a **3D Convolutional Neural Network (3DCNN)** implemented in PyTorch Lightning to classify video-based actions. By capturing both spatial and temporal features, 3DCNNs are well-suited for tasks where motion and context over time are essential.

This project is part of my personal portfolio showcasing data science and deep learning skills, including data preparation, CNN architecture design, hyperparameter tuning, and experimentation with spatiotemporal data.

---

## Key Features
- **3D Convolutions**: Learns spatial and temporal representations simultaneously.  
- **Modular Codebase**: Separate modules for dataset loading, model construction, training, and testing.  
- **PyTorch Lightning**: Simplifies training loops and experiment management.  
- **Configurable**: Easy to customise hyperparameters via a single `config.ini` file.  
- **State-of-the-Art Dataset**: Trained and tested on UCF101, a benchmark dataset for video action recognition.

---

## Directory Structure
Below is a high-level overview of the project’s organisation:

```
3DCNN/
├── images/               # Visual outputs (e.g., confusion matrices, sample frames)
├── src/                  # Source code
│   ├── config.ini        # Configuration file for training/testing
│   ├── datasets.py       # Dataset loading and preprocessing logic
│   ├── models.py         # Model architecture definition (3DCNN)
│   ├── pl_model.py       # PyTorch Lightning wrapper for modular training
│   ├── test_factory.py   # Model evaluation and testing scripts
│   ├── trainer_factory.py# Primary training workflow scripts
│   ├── utils.py          # Utility functions (logging, seeding, metrics etc.)
│   └── video_trainer.py  # Main entry point for training the 3DCNN
└── README.md             # Project documentation
```

---

## Technical Overview

### Model Architecture
The network is defined in [`models.py`](src/models.py) as an **Example3DCNN** class. Key layers include:
1. **3D Convolution** layers with ReLU and Batch Normalisation to learn spatiotemporal features.
2. **3D Max Pooling** layers to reduce dimensionality and aggregate important features.
3. **Fully Connected** layers for final classification into the desired action category.

```python
class Example3DCNN(nn.Module):
    def __init__(self):
        # ...
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1)
        # ...
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, input):
        # ...
        x = self.fc2(x)
        return x
```
  
**Why 3D Convolutions?**  
Traditional 2D convolutions only capture spatial features (height and width). By extending to 3D convolutions, we incorporate the time dimension (depth), allowing the network to detect how an action unfolds across consecutive frames.

### Data Preprocessing
1. **Frame Extraction**: Videos are read frame-by-frame using OpenCV, after which a subset of frames is selected or repeated to maintain a fixed length (e.g. 16 or 64 frames).
2. **Resizing and Normalisation**: Frames are resized (e.g., 128×128) to ensure uniform input sizes and speed up training. Normalisation ensures stable training.
3. **Augmentations** (Optional): Random cropping, flipping, or colour jitter can be applied to increase data diversity.

The logic is encapsulated in [`datasets.py`](src/datasets.py). We load videos from the UCF101 dataset, select only the necessary frames, and transform them into tensors ready for training.

---

## Dataset
We use the **[UCF101 Action Recognition Dataset](https://www.crcv.ucf.edu/data/UCF101.php/)**, containing 13,320 videos across 101 action categories (e.g., *CricketShot*, *Swimming*, *HandStandWalking*).

1. **Splits**: Typically divided into train, validation, and test subsets, e.g. `trainlist01.txt` and `testlist01.txt`.
2. **Frame Extraction**: The script automatically extracts frames and normalises them to the designated size.
3. **Classes to Use**: The configuration file (`config.ini`) allows restricting or specifying certain classes for partial training or quick tests.

*If you plan to use a custom dataset, adapt the code in `datasets.py` accordingly.*

---

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/exponentialR/3DCNN.git
   cd 3DCNN
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv slyk-venv
   ```
3. **Activate the virtual environment**:
   - Windows:
     ```bash
     slyk-venv\Scripts\activate
     ```
   - Unix/MacOS:
     ```bash
     source slyk-venv/bin/activate
     ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
---

## Configuration
The project uses a single file, [`config.ini`](src/config.ini), to control hyperparameters and paths. Some crucial fields include:

- **[hyperparameters]**  
  - `use_valid`: Whether to use a validation split (`yes` or `no`).  
  - `batch_size`: Batch size for training.  
  - `num_gpus`: Number of GPUs to utilise.  
  - `epoch`: Total training epochs.  
  - `data_dir`: Path to your dataset (e.g., UCF101).  
  - `classes_to_use`: Class indices to train on (subset of UCF101).  
  - `lr`: Learning rate for the optimiser.  
  - `num_workers`: Number of subprocesses for data loading.

- **[outputs]**  
  - `resume_ckpt`: Path to a checkpoint for resuming training.  
  - `output_model`: Destination path for saving the trained model.

Adjust these parameters according to your setup before running the training script.

---

## Training and Testing

1. **Training**  
   In the `src` directory, run:
   ```bash
   cd src
   python video_trainer.py --mode train
   ```
   This will:
   - Load your dataset from the location specified in `config.ini`.
   - Instantiate the 3DCNN model.
   - Perform training for the specified number of epochs, logging metrics (loss, accuracy) via PyTorch Lightning.

2. **Testing**  
   Once training is completed, you can test using the same script:
   ```bash
   python video_trainer.py --mode test
   ```
   Ensure `resume_ckpt` in `config.ini` points to a valid checkpoint file (e.g., `EXPERIMENTAL3DCNN-14-0.0001-4.ckpt`).

During training and testing, logs and checkpoints will be saved in the `OUTPUT` directory (or as configured).

---

## Output and Results
- **Logs**: TensorBoard logs for losses, accuracy, and other metrics.  
- **Model Checkpoints**: Stored in the `OUTPUT` directory.  
- **Visualisations**: Optionally, you can generate confusion matrices or sample predictions using your own scripts within the `images/` directory.  

To explore results, launch TensorBoard:
```bash
tensorboard --logdir=training-logs
```
This allows you to visualise training curves, learning rates, and track model improvements over epochs.

---

## Future Enhancements
- **Data Augmentation**: Incorporate more robust strategies like random temporal sampling or advanced geometric transformations.  
- **Advanced Architectures**: Experiment with I3D (Inflated 3D ConvNet) or S3D models.  
- **Multi-Head Attention**: Combine 3D convolutions with Transformers for long-sequence modelling.  
- **Hyperparameter Optimisation**: Integrate libraries like Optuna for automatic hyperparameter tuning.  
- **Deployment**: Convert the final model to TensorRT or ONNX for real-time inference on edge devices.

---

## Contact
For any queries or suggestions, feel free to reach out via:
- **Email**: [samueladebayo@ieee.org](mailto:samueladebayo@ieee.org)
- **LinkedIn**: [Samuel Adebayo](https://linkedin.com/in/samneering)
- **GitHub**: [Samuel A.](https://github.com/exponentialR)

Happy coding and best of luck with your 3D Action Recognition tasks!

---

*© 2025 Your Name. This project is provided as-is without warranty of any kind.*