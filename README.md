# Word-level-language-modeling
Benchmarking tricks to improve a base RNN word-level language model.

We implement the following tricks: 
_early-stopping
_learning rate decay: annealing the learning rate when validation perplexity starts increasing
_tying the encoder weights (=the word embeddings matrix) with the decoder weights (=the softmax output matrix)
