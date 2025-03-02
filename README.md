# FCBTYOLO: A Lightweight and High-Performance Fine Grain Detection Strategy for Rice Pests
🧙‍♀️Yiyu Yang  [[LinkedIn](https://www.linkedin.com/in/yiyu-yang/)] ✉️[Email: yangalita73@gmail.com]  

## Introduction
This repository contains the implementation of **FCBTYOLO**, a lightweight and high-performance object detection model designed for fine-grained detection of rice pests. The model is based on the YOLOv8 architecture, enhanced with a novel **Fully Connected Bottleneck Transformer (FCBT)** module. The goal of this project is to provide an efficient and accurate solution for detecting rice pests in agricultural fields, enabling timely pest control and reducing crop losses.

The model achieves a mean Average Precision (mAP@50) of **93.6%** with a model size of only **6.7MB**, making it suitable for deployment on embedded devices in real-world agricultural scenarios.  

<img src="./docs/imgs/img 13.png" alt="img 13" style="width:80%; display: block; margin: auto;"> 

<p style="font-size: 10px;"><b>FIGURE 1.</b> The attention heatmap drawn by FCBTYOLO. (a), (d) are the original images. (b), (e) are the attention heatmaps after visualizing the feature maps of YOLOv8n. (c), (f) are the attention heatmaps after visualizing the feature maps of FCBTYOLO.</p>

## Key Features
- **FCBT Module**: A novel module that combines convolutional layers with Multi-head Self-attention (MHSA) to enhance feature extraction and improve detection accuracy while maintaining a lightweight architecture.
- **GPest14 Dataset**: A large-scale dataset containing 14 categories of rice pests, constructed using FastGAN image generation, web crawling, and manual selection. The dataset is designed to address class imbalance and improve model generalization.
- **Lightweight and Efficient**: The model is optimized for deployment on resource-constrained devices, with a small model size (6.7MB) and fast inference time (16.8ms per image).
- **High Performance**: Achieves state-of-the-art performance on both the GPest14 dataset and the public Pest24 dataset, with a 1.1% improvement in mAP over the baseline YOLOv8n model.

## Installation


### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/esemsc-yy1824/FCBTYOLO.git
   cd FCBTYOLO
   ```
2. Create a virtual environment and install dependencies:
   ```sh
   conda create -n fcbt python=3.10
   conda activate fcbt
   pip install -r requirements.txt
   ```

## Dataset
### GPest14 Dataset
The dataset consists of 13,877 images covering 14 rice pest categories. It was augmented using FastGAN for better class balance.

<img src="./docs/imgs/img 1.png" alt="img 1" style="width:80%; display: block; margin: auto;">

<p style="font-size: 10px"><b>FIGURE 2.</b> Example images from the Gpest14 dataset</p>


#### Dataset Structure
The dataset is organized as follows:
  ``` 
    data/
    └── GPest14/
        ├── train/       # Training images and labels
        ├── val/         # Validation images and labels
        └── test/        # Test images and labels
  ```
- **Images:** The images are stored in the respective `train`, `val`, and `test` folders.
- **Labels:** The corresponding annotation files (in YOLO format) are stored alongside the images.

#### Dataset Privacy
The GPest14 dataset is private and protected. It is not publicly available due to licensing and privacy restrictions. However, it could be replaced with other compliant datasets.

## Model Architecture

The FCBTYOLO model is based on the YOLOv8 architecture, with the following key modifications:

- **FCBT Module**: The FCBT module is integrated at the end of the YOLOv8 backbone. It replaces some spatial convolution channels with Multi-head Self-attention (MHSA) layers, allowing the model to capture global dependencies while maintaining a lightweight structure.
- **Lightweight Design**: The model is designed to be compact, with a total of 4.247 million parameters and a model size of 6.7MB, making it suitable for deployment on embedded devices.

<img src="./docs/imgs/img 9.png" alt="img 1" style="width:80%; display: block; margin: auto;"> 

<p style="font-size: 10px"><b>FIGURE 3.</b>  The structure of the FCBTYOLO</p>

## Training
Train the FCBTYOLO model using the following command:
```sh
python train.py
```

### Training Parameters:
- **Image Size**: 640x640
- **Optimizer**: SGD
- **Initial Learning Rate**: 0.01
- **Epochs**: 150
- **Batch Size**: 16

## Evaluation
Evaluate the model's performance on the validation set:
```sh
python val.py
```

## Inference
Run inference on a test image:
```sh
python detect.py
```

## Results

### Performance on GPest14 Dataset

| Model       | mAP@50 | Precision | Recall | Model Size (MB) | Inference Time (ms) |
|-------------|--------|-----------|--------|-----------------|---------------------|
| FCBTYOLO    | 93.6%  |0.92      | 0.89   | 6.7             | 16.8                |
| YOLOv8n     | 92.2%  | 0.91      | 0.87   | 6.2             | 17.9                |

<img src="./docs/imgs/img 14.png" alt="img 14" style="width:70%; display: block; margin: auto;"> 

<p style="font-size: 10px"><b>FIGURE 4.</b> Comparison of the accuracy and weight file size of the FCBTYOLO model with CNN and SSD detection frameworks.</p>

## Citation
If you use these code in your research, please cite the [original paper](https://ieeexplore.ieee.org/document/10250915).


## License
This work is licensed under a **GNU General Public License v3.0 (GPL v3.0)** License. See [LICENSE](LICENSE) for details.

## Acknowledgments
This work was supported by the *Sichuan Provincial Department of Science and Technology* and the *National Innovation and Entrepreneurship Training Program for College Students*.
