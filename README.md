# Word-level-language-modeling
Benchmarking tricks to improve a base RNN word-level language model.

We implement the following tricks:
 - early-stopping
 - learning rate decay: annealing the learning rate when validation perplexity starts increasing
 - dropout rates: at the input, and variational dropout (https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf)
 - tying the encoder weights (=the word embeddings matrix) with the decoder weights (=the softmax output matrix) as in https://arxiv.org/pdf/1611.01462.pdf
 
 
 Thanks to the Pytorch/examples repository for inspiration:
 https://github.com/pytorch/examples/tree/master/word_language_model
