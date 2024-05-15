import torch
import argparse
import re
import pickle
import os
from model import BigramLanguageModel


stoi = {}
itos = {}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
model = None

def load_model(model_file, enc_file):
    global model, stoi, itos
    print('Loading model..',model_file)
    model = torch.load(model_file)
    #print(model)
    # with open(model_file, 'rb') as f:
        # print('Loading model..',model_file)
        # model = pickle.load(f)
    with open(enc_file, 'rb') as f:
        print('Loading encoder_decoder..',enc_file)
        enc_dec = pickle.load(f)
        stoi = enc_dec['encoder']
        itos = enc_dec['decoder']


def run_cli():
    global model
    #print(model)
    print('_'*30)
    print(' '*10,'Welcome to the CLI tool')
    print('_'*30)
    print('\n')
    output_tokens = 100
    print("Enter your prompt (Type 'exit' to terminate, Type t11 to set output tokens to 11 (and so on))")
    command = input('> ')
    while command != 'exit':
        update_tokens = re.match("t([0-9]+)",command)
        if update_tokens is not None:
            output_tokens = int(update_tokens[1])
            print('Output tokens set to',output_tokens)
        else:
            context = torch.tensor(encode(command)).unsqueeze(0)
            print(decode(model.generate(context, max_new_tokens=output_tokens)[0].tolist()))
        command = input('\n> ')







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prompting Tool")
    parser.add_argument('model_file', type=str, help='Name of output model')
    parser.add_argument('enc_file', type=str, help='File with the encoder and decoder maps')
    args = parser.parse_args()

    load_model(args.model_file, args.enc_file)
    run_cli()