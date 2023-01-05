# DeepAlgPro: an interpretable deep neural network model for predicting allergenic proteins
## Introduction
Allergies have become an emerging public health problem worldwide.It is critical to evaluate potential allergens, especially today when the number of modified proteins in food, therapeutic drugs, and biopharmaceuticals is increasing rapidly. Here, we proposed a software, called DeepAlgPro, that combined a convolutional neural network (CNN) with Multi-Headed Self-Attention (MHSA) and was suitable for large-scale prediction of allergens. As shown in Figure 1, the hybrid model consists of embedding module, feature extraction module, and two-label output module.

![image](https://user-images.githubusercontent.com/113486741/210779159-d7b5590b-2e8b-4ace-ac2b-44cb6b5077d2.png)

## Installation
python 3.9<br>
pytorch==1.12.1+cu116<br>
numpy==1.23.4<br>
```
pip3 install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
