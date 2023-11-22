# Solafune-cloud
## 衛星画像の雲領域検出 Masking Clouds in Satellite Imageries Contest 2023

Welcome to this repository. This project is an attempt for the Solafune contest for masking clouds in satellite. I try to use techniques and models to detect cloud regions within satellite images. The aim of this contest is to improve the quality and accuracy of meteorological data interpretation.

## 🚀 Getting Started

### 📚 Dataset 
You can either run the script if you have an account or download the dataset used in this project from [Solafune衛星画像の雲領域検出](https://solafune.com/ja/competitions/65571524-39b0-4972-9001-ba6b61d6b20f?menu=data&tab=&modal=%22%22). Please make sure to review and respect the dataset’s usage terms and conditions.

## To download dataset:
```
sh download_data.sh
```

## 🐍 Create & Activate conda environment:
Creating a virtual environment helps manage dependencies and ensure that your project remains reproducible and consistent. Follow these steps to create and activate a Conda environment:
```shell
conda create -n ENVNAME python=3.10 
conda activate ENVNAME
```

## 📦 Install Dependencies:
```
pip install -r requirements.txt
```
Ensure consistent code execution by installing necessary dependencies as listed in requirements.txt

## 🛠 Usage Instructions
### 🚄 Run Model Training:

```
sh scripts/run_train.sh
```
Upon execution of the training script A directory will be automatically created to store the model checkpoints
based on the performance.
### 🧠 Run Inference:
```
sh scripts/run_inference.sh
```
The best models will be utilized in an ensemble to generate predictions on unseen data.
## 🤖 Used models:
```
DeepLabV3Plus with efficientnet-b1,b2, b3, and Resnet101 encoders.
UnetPlusPlus with efficientnet-b1, and Resnet101 encoders.
```

The models used in this project are renowned architectures in the field of semantic image segmentation, employing the PyTorch framework for implementation and training. 

- **DeepLabV3Plus:** A state-of-the-art deep learning model for semantic image segmentation, where the objective is to classify each pixel in the input image into a category. Different backbones, namely efficientnet-b1, efficientnet-b2, efficientnet-b3, and Resnet101 encoder, are utilized to extract features from input images.

- **UnetPlusPlus:** An advanced version of the traditional U-Net model, designed to provide more precise segmentation results, especially for smaller objects. Here, we use efficientnet-b1, efficientnet-b2, efficientnet-b3, and Resnet101 encoder as the encoders.

Both model architectures are implemented using the [Segmentation Models PyTorch (SMP)](https://pypi.org/project/segmentation-models-pytorch/) library.

## 🤖 Output predictions:

![predictions](https://github.com/hussainsan/solafune-cloud/blob/8d7b8ea8414e86522c4acbecfc8e81ebf1672fdc/predictions_masks.png)