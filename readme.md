# Deep Lyrics Generation

Generate greek pop lyrics using deep learning.

## Description

In this project we attempt to use an AI model to generate pop lyrics. 
The model was trained on thousands of greek lyrics in a self-supervised way.
To learn more about the training process and the different models that we evaluated,
we encourage you to read the full report available as a jupyter notebook (notebook/report.ipynb)

Note: All lyrics belong to their rightful owners. This project uses the lyrics for academic purposes only.
## How to run

First install the requirements with

`pip install -r requirements.txt`

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
This method uses a two-layer LSTM with 256 hidden neurons and regex tokenization with ~3000 vocabulary size.

Run with:
`python ./generate_multi_lyric.py -t 4`
For 't' we suggest a value between 3 - 6.

### generate_multi_lyric_interactive.py
The model with suggest the next top words. The user proceeds with choosing the word and the model will make the next suggestion, and so on...
In this way you can create a verse with help from the model.
This method uses a two-layer LSTM with 256 hidden neurons and regex tokenization with ~3000 vocabulary size.

Run with:
`python ./generate_multi_lyric_interactive.py_ -t 6`
't' is the number of suggested predictions to choose from.

**Note:** If you get the following error:

`UnicodeEncodeError: 'charmap' codec can't encode characters in position 1-2: character maps to <undefined>`
 
set the env variable PYTHONIOENCODING to utf-8.
 
* Linux: export PYTHONIOENCODING=utf-8
* Windows: set PYTHONIOENCODING=utf-8



