# Brain Tumor Detection using Deep Learning (CNN)

This project develops a Computer Vision model using Convolutional Neural Networks (CNNs) to detect the presence of a brain tumor from MRI scan images. It utilizes a Kaggle dataset containing labeled MRI images. 

## Dataset Structure

The dataset consists of four main tumor classes:
- `glioma`
- `meningioma`
- `notumor` (No tumor)
- `pituitary`

The data is split into two directories:
- `data/Training/` 
- `data/Testing/`

## Project Folder Structure

```
hands-on-2/
│── data/                   # Contains Training and Testing MRI images
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
│── src/                    # Source code files
│   ├── train.py            # Script to train CNN model and save .h5 output
│   └── predict.py          # Inference script to predict tumor status
│
│── models/                 # Directory where the trained models are saved
│   └── brain_tumor_model.h5
│
│── requirements.txt        # Required python libraries
└── README.md
```

## Setup & Installation

It is recommended to use a virtual environment. Install required dependencies:
```bash
pip install -r requirements.txt
```

## How to Train

Run the training script to build the CNN model from the datasets located in the `data/` folder:
```bash
cd src
python train.py
```
This script will produce `brain_tumor_model.h5` inside the `models/` directory.

## How to Predict

To evaluate a new MRI image and predict whether there is a tumor, run:
```bash
cd src
python predict.py <path_to_image>
```

Example:
```bash
python predict.py ../data/Testing/meningioma/image(1).jpg
```

This will print the predicted class, confidence percentage, and whether it identified the presence of a tumor.
