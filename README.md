# DeepAlgPro: an interpretable deep neural network model for predicting allergenic proteins
## Introduction
Allergies have become an emerging public health problem worldwide.It is critical to evaluate potential allergens, especially today when the number of modified proteins in food, therapeutic drugs, and biopharmaceuticals is increasing rapidly. Here, we proposed a software, called DeepAlgPro, that combined a convolutional neural network (CNN) with Multi-Headed Self-Attention (MHSA) and was suitable for large-scale prediction of allergens. 

## Platform requirements


## Installation
1. Download DeepAlgPro
```
git clone https://github.com/chun-he-316/DeepAlgPro.git
```
2. Install required packages<br>
- python 3.9<br>
- Bio==1.5.3
- numpy==1.23.4
- pandas==1.5.0
- scikit_learn==1.2.1
- torch==1.12.1+cu116
- torchmetrics==0.9.3
```
pip3 install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
## Train and Test the model
```
usage: python main.py [-h] [-i INPUTS] [--epochs N] [--lr LR] [-b N] [--mode {train,test}]
```
#### Optional arguments
```
  -h, --help            show this help message and exit
  -i INPUTS, --inputs INPUTS
  --epochs N            number of total epochs to run
  --lr LR, --learning-rate LR
                        learning rate
  -b N, --batch-size N
  --mode {train,test}
```
#### Example
```
python main.py -i data/all.train.fasta --epochs 120 --lr 0.0001 -b 72 --mode train
```
#### Output files
The training process generates a number of files. The types of files are listed below.
- train.log file —— Record loss values for each batch
- .everyepoch.valid.txt files —— Record the validation results after each training epoch
- .pt file —— The model obtained by training
- valid.log —— Results of 10-fold cross-validation
## Use DeepAlgPro to predict allergens
```
usage: python predict.py [-h] [-i INPUTS] [-b N] [-o OUTPUT]
```
#### Optional arguments
```
  -h, --help            show this help message and exit
  -i INPUTS, --inputs INPUTS
                        input file
  -b N, --batch-size N
  -o OUTPUT, --output OUTPUT
                        output file
```
#### Input files
The input file specified by -i is a protein sequence file; each sequence has a unique id and starts with >. The input protein sequence number must be divisible by the batch size.
#### Example
```
python predict.py -i data/all.test.fasta -o allergen.predict.txt
```
#### Output files
  The default result file is `allergenic_predict.txt`, a file with tabs as spacers. You can also specify the output file with `-o`. The first column in the output file is the id of the input protein, the second column is the score between 0 and 1 predicted by the model, and the third column value is the predicted result, allergenicity or non-allergenicity.

