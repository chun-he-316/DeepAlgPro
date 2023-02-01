# DeepAlgPro: an interpretable deep neural network model for predicting allergenic proteins
## Introduction
Allergies have become an emerging public health problem worldwide.It is critical to evaluate potential allergens, especially today when the number of modified proteins in food, therapeutic drugs, and biopharmaceuticals is increasing rapidly. Here, we proposed a software, called DeepAlgPro, that combined a convolutional neural network (CNN) with Multi-Headed Self-Attention (MHSA) and was suitable for large-scale prediction of allergens. 

## Dependency
- python 3.9<br>
- pytorch==1.12.1+cu116<br>
- sklearn<br>
- torchmetrics<br>
- numpy<br>
- Bio<br>
- argparse<br>
- pandas<br>
## Installation
1. Download DeepAlgPro
```
git clone https://github.com/chun-he-316/DeepAlgPro.git
```
2. Install required packages<br>
You can choose to download the above dependencies separately, or use the following command to build the DeepAlgPro environment and download all the dependencies at the same time. If you use the latter, please modify the prefix in environment.yaml to be the correct path of the environment you intend to create.
```
conda env create -f environment.yaml
conda avtivate DeepAlgPro
pip install -r requirements.txt
```
## Train and Test the model
```
usage: main.py [-h] [-i INPUTS] [--epochs N] [--lr LR] [-b N] [--mode {train,test}]
```
### Optional arguments
```
  -h, --help            show this help message and exit
  -i INPUTS, --inputs INPUTS
  --epochs N            number of total epochs to run
  --lr LR, --learning-rate LR
                        learning rate
  -b N, --batch-size N
  --mode {train,test}
```
### Example
```
python main.py -i data/all.train.fasta --epochs 120 --lr 0.0001 -b 72 --mode train
```
## Use DeepAlgPro to predict allergens
```
usage: predict.py [-h] [-i INPUTS] [-b N] [-o OUTPUT]
```
### Optional arguments
```
  -h, --help            show this help message and exit
  -i INPUTS, --inputs INPUTS
                        input file
  -b N, --batch-size N
  -o OUTPUT, --output OUTPUT
                        output file
```
### Example
```
python predict.py -i data/all.test.fasta -o allergen.predict.txt
```
