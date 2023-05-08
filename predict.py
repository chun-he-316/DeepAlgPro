from model import convATTnet
from Bio import SeqIO
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import os


def main():
    argparser = argparse.ArgumentParser(
        description="DeepAlgPro Network for predicting allergens.")
    argparser.add_argument('-i', '--inputs', default='./',
                           type=str, help='input file')
    argparser.add_argument('-b', '--batch-size', default=1, type=int,
                           metavar='N')
    argparser.add_argument(
        '-o', '--output', default='allergenic_predict.txt', type=str, help='output file')

    args = argparser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    predict(args)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("We will use "+torch.cuda.get_device_name())
else:
    device = torch.device('cpu')

codeadict = {'A': "1", 'C': "2", 'D': "3", 'E': "4", 'F': "5", 'G': "6", 'H': "7", 'I': "8", 'K': "9", 'L': "10",
             'M': "11", 'N': "12", 'P': "13", 'Q': "14", 'R': "15", 'S': "16", 'T': "17", 'V': "18", 'W': "19", 'Y': "20"}


class MyDataset(Dataset):
    def __init__(self, sequence, labels):
        self._data = sequence
        self._label = labels

    def __getitem__(self, idx):
        sequence = self._data[idx]
        label = self._label[idx]
        return sequence, label

    def __len__(self):
        return len(self._data)


def format(predict_fasta):
    formatfasta = []
    recordid = []
    for record in SeqIO.parse(predict_fasta, 'fasta'):
        fastalist = []
        length = len(record.seq)
        if length <= 1000:
            for i in range(1, 1000-length+1):
                fastalist.append(0)
            for a in record.seq:
                fastalist.append(int(codeadict[a]))
        formatfasta.append(fastalist)
        recordid.append(record.id)
    inputarray = np.array(formatfasta)
    idarray = np.array(recordid, dtype=object)
    return(inputarray, idarray)


def predict(args):
    profasta = torch.tensor(format(args.inputs)[0], dtype=torch.long)
    proid = format(args.inputs)[1]
    data_ids = MyDataset(profasta, proid)
    data_loader = DataLoader(
        dataset=data_ids, batch_size=args.batch_size, shuffle=False)

    # load the model
    model = convATTnet()
    model.to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('model.pt'), strict=True)
    else:
        model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')),
                              strict=True)
    with torch.no_grad():
        pred_r = []
        for i, data in enumerate(data_loader, 0):
            inputs, inputs_id = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            if device == torch.device('cpu'):
                probability = outputs[0].item()
            else:
                probability = outputs.item()
            if probability > 0.5:
                pred_r.append(
                    [''.join(inputs_id), probability, 'allergenicity'])
            else:
                pred_r.append(
                    [''.join(inputs_id), probability, 'non-allergenicity'])
    # generate outfile file
    df = pd.DataFrame(pred_r, columns=['protein', 'scores', 'predict result'])
    df.to_csv(args.output, sep='\t', header=True, index=True)


if __name__ == '__main__':
    main()
