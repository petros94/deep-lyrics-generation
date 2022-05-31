import torch
import sys, getopt, os
import re
import warnings
warnings.filterwarnings("ignore")


## Use cuda if available
from config import device
from dataset import LyricsDatasetBPE, LyricsDatasetRegex
from train import predict, predict_many

## Define model
from model import LSTM

dataset = LyricsDatasetRegex([[]])
dataset = torch.load('pretrained_models/train_set_rt3.pt')
model = LSTM(len(dataset.token_set), padding_idx=dataset.padding_idx, embedding_size=100, hidden_size=256, num_layers=2)
model.load_state_dict(torch.load('pretrained_models/model_rt3.pt', map_location=torch.device(device)))
model.to(device)
model.eval()

def main(argv):
    ## Read args from user
    try:
        opts, args = getopt.getopt(argv,"ht:",["top-words="])
    except getopt.GetoptError:
        print('generate_multi_lyric.py -t <top_words>')
        sys.exit(2)

    top_words = 5
    for opt, arg in opts:
        if opt == '-h':
            print('generate_multi_lyric.py -t <top_words>')
            sys.exit()
        elif opt in ("-t", "--top-words"):
            top_words = int(arg)

    ## Print instructions
    print("Multi-Lyric Generation in greek!")
    print("Type 'quit' to exit the program")

    ## While loop
    regex = re.compile("\w+|\\.")
    val = input("Type a word or phrase and press Enter, to get an AI-generated lyric: ")
    seq = regex.findall(val)
    while val != 'quit':
        preds = predict_many(model, dataset, seq, 100, top_only=top_words, no_unk=True)

        for i,p in enumerate(preds):
            print(f'{i}: {p}')

        val = input("Select the next subword (number) to continue: ")

        if val == 'quit':
            break

        seq.append(preds[int(val)])
        print("--> " + " ".join(seq))

if __name__ == "__main__":
    main(sys.argv[1:])