# 3D Convolutional Neural Network (3DCNN)
![3DCNN](images/3d-CNN.png)
## Overview
This repository contains the implementation of a 3D Convolutional Neural Network (3DCNN). The 3DCNN model is adept at processing and analyzing three-dimensional data and finds applications in diverse fields such as medical imaging, video processing, and robotics.
Don't forget to change the following in ```src/config.ini```
-- ``use_valid``
-- ``batch_size``
-- `num_gpus`
-- `classes_to_use`
-- `learning_rate`
-- `num_workers`

## Dataset use
Since it is huge dataset and we are only experimenting - we will be using 50 clips of each classes and 10 classes.
- [UCF 101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
- After downloading please change the ```data_dir``` in ```src/config.ini``` to where your dataset is located.


## Installation

### Prerequisites
- Python 3.x
- Python packages as listed in `requirements.txt`

### Setup
1. Clone the repository.
2. Set up a virtual environment (recommended):
   ```
   python -m venv slyk-venv
   ```
3. Activate the virtual environment:
   - Windows: `slyk-venv\Scripts\activate`
   - Unix or MacOS: `source slyk-venv/bin/activate`
4. Install the necessary packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
The source code is housed in the `src` directory. Run the ```video_trainer.py``` in this directory to execute the 3DCNN model:
   ```
   cd src
   python3 video_trainer.py --mode train 
   ```
## Output
The `OUTPUT` directory stores all generated outputs, including images, model checkpoints, and logs.
