# Deep Lyrics Generation

Generate greek pop lyrics using deep learning.

## Description

In this project we attempt to use an AI model to generate pop lyrics. 
The model was trained on thousands of greek lyrics in a self-supervised way.
To learn more about the training process and the different models that we evaluated,
we encourage you to read the full report available as a jupyter notebook (notebook/report.ipynb)

Note: All lyrics belong to their rightful owners. This project uses the lyrics for academic purposes only.
## How to run

There are two files that you can run to generate lyrics:

### generate_single_lyric.py
Input the beginning of a lyric and let the model do the rest.
This method uses a one-layer LSTM with 128 hidden neurons and regex tokenization with ~3000 vocabulary size.

Run with:
`python ./generate_single_lyric.py -t 20` 

The 't' parameter controls the randomness of the algorithm. The lower the value, the more deterministic the results.
We suggest a value between 5 - 50.

### generate_multi_lyric.py
Input some words and the model will generate a whole verse.
This method uses a two-layer LSTM with 256 hidden neurons and byte pair encoding tokenization with ~10000 vocabulary size.

Run with:
`python ./generate_multi_lyric.py -t 20`