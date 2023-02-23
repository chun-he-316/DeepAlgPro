import argparse
import Train
import Test


def main():
    argparser = argparse.ArgumentParser(
        description="DeepAlgPro Network for predicting allergens.")
    argparser.add_argument('-i', '--inputs', default='./', type=str)
    argparser.add_argument('--epochs', default=100, type=int,
                           metavar='N',
                           help='number of total epochs to run')
    argparser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                           metavar='LR', help='learning rate', dest='lr')
    argparser.add_argument('-b', '--batch-size', default=72, type=int,
                           metavar='N')
    argparser.add_argument('--mode', default='train', type=str,
                           choices=['train', 'test'])
    # argparser.add_argument('--model', default='convATTnet', type=str,
    #                        choices=['convATTnet', 'conv1d', 'ATTnet'])

    args = argparser.parse_args()
    if args.mode == 'train':
        Train.train(args)
    if args.mode == 'test':
        Test.test(args)


if __name__ == '__main__':
    main()
