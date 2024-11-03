# Pre-training a GPT like language model from scratch
A project to pre-train a GPT-like language model from scratch. Code was referred from Andrej Karpathy's lecture on building GPT from scratch. Modifications to the original architecture like Fast Attention have been omitted to keep the project reflective of the first principles that govern this beautiful architecture. The original repo by Andrej Kaprpathy has those modifications. For more serious, production-level applications it makes sense to adopt those time-tested modifications.


## Misc

1. Use the following to run the training as a background process with buffering disabled to stdout
```
stdbuf -oL -eL ./train.sh ../data/input.txt minigpt-1.pth &
```

## References
1. [Conceptual Background](https://pratik-doshi-99.github.io/posts/implementing-gpt/)
2. [Reference Project](https://github.com/karpathy/ng-video-lecture)
3. [Youtube Tutorial](https://youtu.be/kCc8FmEb1nY?si=cMstqwDnCIbfgdEM)
4. [Dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
5. [Google Colab Link](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)

