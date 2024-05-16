from model import BigramLanguageModel
import torch
import argparse
import os
import pickle
from torch.utils.tensorboard import SummaryWriter
import datetime


'''
TODO:
1. Train the model on a public dataset
2. Create a cli interface that takes a model, and allows the user to write prompts and get a response
3. Publish the model trained on a public dataset to hugging face and integrate its interface to prompt that
'''



# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
context_length = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
embed_dim = 32
num_heads = 4
num_layers = 4
dropout = 0.0
mlp_dropout = 0.0
input_file = 'input.txt'
model_name = 'mini-gpt.pth'
preload_model = ''
writer = SummaryWriter(log_dir=os.path.join('training_runs','experiment_3'))


def save_model(model):
    os.makedirs('models',exist_ok=True)
    # with open(os.path.join('models',f'{model_name}.enc'), 'wb') as f:
    #     pickle.dump(model, f)
    torch.save(model, os.path.join('models',model_name))

def load_model():
    if os.path.exists(preload_model):
        print(f'Model {preload_model} found, preloading it...')
        return torch.load(preload_model)
    print(f'Could not find model {preload_model}, preload skipped')
    return None
        


def save_encoder_decoder(encoder_map, decoder_map):
    os.makedirs('models',exist_ok=True)
    with open(os.path.join('models',f'{model_name}.enc'), 'wb') as f:
        pickle.dump({'encoder':encoder_map, 'decoder':decoder_map}, f)
    


def train_model():
    torch.manual_seed(1337)

    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    save_encoder_decoder(stoi, itos)

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long, device=device)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    # data loading
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - context_length, (batch_size,))
        x = torch.stack([data[i:i+context_length] for i in ix])
        y = torch.stack([data[i+1:i+context_length+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


    model = None
    if preload_model is not None and preload_model != '':
        model = load_model()
    
    if not model:
        model = BigramLanguageModel(
            num_layers=num_layers,
            num_heads=num_heads,
            context_length=context_length,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            device=device,
            dropout=dropout,
            mlp_dropout=mlp_dropout
        )

    model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            t1 = datetime.datetime.now()
            losses = estimate_loss()
            t2 = datetime.datetime.now()
            writer.add_scalar('Loss/val', losses['val'], iter)
            writer.add_scalar('Time/val', (t2 - t1).total_seconds(), iter)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

       
        t1 = datetime.datetime.now()
        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        logits, loss = model(xb, yb)
        t2 = datetime.datetime.now()
        writer.add_scalar('Loss/train', loss, iter)
        writer.add_scalar('Time/train', (t2 - t1).total_seconds(), iter)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    save_model(model)
    writer.close()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Training a small language model 'mini-gpt'")


    # Add arguments
    parser.add_argument('--model_name', type=str, default='mini-gpt.pth', help='Name of output model')
    parser.add_argument('--input_file', type=str, default='input.txt', help='The input file to train from')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batches to be computed in parallel')
    parser.add_argument('--context_length', '-c', type=int, default=128, help='Context Length for LM')
    parser.add_argument('--iters', '-i', type=int, default=2000, help='Training Iterations')
    parser.add_argument('--eval_interval', '-e', type=int, default=100, help='Evaluation Interval')
    parser.add_argument('--eval_iters', '-ei', type=int, default=200, help='Evaluation Iterations')
    parser.add_argument('--embed_dim', '-ed', type=int, default=32, help='Embedding Space Dimensions')
    parser.add_argument('--n_heads', '-hd', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', '-l', type=int, default=8, help='Number of Attention Blocks')
    parser.add_argument('--dropout', '-d', type=float, default=0, help='Dropout rate')
    parser.add_argument('--mlp_dropout', '-md', type=float, default=0, help='MLP Dropout rate')
    parser.add_argument('--preload', '-pl', type=str, default='', help='Path to the model that must be preloaded')

    # Parse the arguments
    args = parser.parse_args()
    model_name = args.model_name
    input_file = args.input_file
    batch_size = args.batch_size
    context_length = args.context_length
    max_iters = args.iters
    eval_interval = args.eval_interval
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = args.eval_iters
    embed_dim = args.embed_dim
    num_heads = args.n_heads
    num_layers = args.n_layers
    dropout = args.dropout
    mlp_dropout = args.mlp_dropout
    preload_model = args.preload

    print(batch_size, context_length, max_iters, eval_interval, eval_iters, embed_dim, num_heads, num_layers, dropout, mlp_dropout)

    # Use the arguments
    train_model()

