import torch
import numpy as np
from random import choices
from config import device

def predict(model, dataset, test_seq, max_seq, deterministic=False, no_unk=False, top_only=False):
    """
    Make a forward-pass and predict the next word.

    :param model: an LSTM instance of class defined in model.py
    :param dataset: a dataset instance of class defined in dataset.py
    :param test_seq: a sequence of words to feed to the model. e.g. ['Δεν', 'έχει', 'σίδερα', 'η']
    :param max_seq: the max length of generated sequence
    :param deterministic: If true, the model will always return the most likely prediction
    :param no_unk: Prevent model from generating <unk> words
    :param top_only: int. If set, the model will chose only among the 'n' top words (weighted by their probability) for the next prediction
    :return: the original + predicted sequence as a string
    """
    model.eval()
    test_seq = " ".join(['#'] + test_seq)
    input = dataset.tokens_to_ids(test_seq)
    h, c = model.init_state(1)

    with torch.no_grad():
        values = []

        for i in range(max_seq):

            # Reinitialize weights
            h, c = model.init_state(1)

            # Prepare input
            length = len(input)
            x = torch.tensor(input)
            x = torch.nn.utils.rnn.pad_sequence([x], padding_value=dataset.padding_idx)

            # Predict next token probabilities
            out, (h, c) = model.forward(x.to(device), [length], (h, c))
            p = torch.nn.functional.softmax(out[-1].squeeze(0)).detach().cpu().numpy()

            idx = 1
            if deterministic:
                idx = p.argmax()
            else:
                if top_only:
                    indices = p.argsort()[::-1][0:top_only]
                    if no_unk:
                        while idx == 1:
                            idx = choices(indices, p[indices] / sum(p[indices]))[0]
                    else:
                        idx = choices(indices, p[indices] / sum(p[indices]))[0]
                else:
                    if no_unk:
                        while idx == 1:
                            idx = choices(np.arange(0, model.n_vocab), p)[0]
                    else:
                        idx = choices(np.arange(0, model.n_vocab), p)[0]

            idx = int(idx)
            values.append(idx)

            decoded = dataset.ids_to_tokens([idx])

            if type(decoded) == list:
                token_pred = decoded[0]
            else:
                token_pred = decoded

            if token_pred in ('&', '_&'):
                break

            input.append(idx)

        return dataset.ids_to_tokens(values)


def train(model, train_set, validation_set, max_epochs, batch_size, weights):
    """
    Train the model.

    :param model: an LSTM instance of class defined in model.py
    :param train_set: a dataset instance of class defined in dataset.py
    :param validation_set: a dataset instance of class defined in dataset.py
    :param max_epochs: max number of epochs to train
    :param batch_size: the batch size
    :param weights: the cross entropy loss weights (optional)
    :return: the evaluation data (train/validation loss and perplexity)
    """
    evaluation_data = {
        'train_loss': [],
        'validation_loss': [],
        'validation_perplexity': [],
    }

    # Define model and loss functions
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=train_set.padding_idx, weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    # Define dataloaders
    def collate_pad(batch):
        in_ = []
        out_ = []
        seq_len_x = []
        seq_len_y = []
        for x, y in batch:
            in_.append(x)
            out_.append(y)
            seq_len_x.append(len(x))
            seq_len_y.append(len(y))

        return torch.nn.utils.rnn.pad_sequence(in_,
                                               padding_value=train_set.padding_idx).cuda(), torch.nn.utils.rnn.pad_sequence(
            out_, padding_value=train_set.padding_idx).cuda(), seq_len_x, seq_len_y

    dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, collate_fn=collate_pad)
    validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=64, collate_fn=collate_pad)

    # Train loop
    for epoch in range(max_epochs):
        model.train()

        ## Train in batches
        for batch, (x, y, seq_len_x, seq_len_y) in enumerate(dataloader):
            h, c = model.init_state(batch_size)

            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            y_pred, (h, c) = model(x, seq_len_x, (h, c))

            loss = criterion(y_pred.transpose(0, 1).transpose(1, 2), y.transpose(0, 1))
            loss.backward()
            optimizer.step()

        ## Evaluate performance on train set
        with torch.no_grad():
            model.eval()
            epoch_loss = 0
            loss_counter = 0
            for batch, (x, y, seq_len_x, seq_len_y) in enumerate(dataloader):
                h, c = model.init_state(batch_size)

                x = x.to(device)
                y = y.to(device)

                y_pred, (h, c) = model(x, seq_len_x, (h, c))

                loss = criterion(y_pred.transpose(0, 1).transpose(1, 2), y.transpose(0, 1))

                epoch_loss += loss.item()
                loss_counter += 1

            epoch_loss /= loss_counter

        ## Evaluate performance on validation set
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            loss_counter = 0
            for batch, (x, y, seq_len_x, seq_len_y) in enumerate(validation_dataloader):
                h, c = model.init_state(64)

                x = x.to(device)
                y = y.to(device)

                y_pred, (h, c) = model(x, seq_len_x, (h, c))

                loss = criterion(y_pred.transpose(0, 1).transpose(1, 2), y.transpose(0, 1))

                valid_loss += loss.item()
                loss_counter += 1

            valid_loss /= loss_counter
            perplexity = np.exp(valid_loss)

        ## update evaluation data
        evaluation_data['train_loss'].append(epoch_loss)
        evaluation_data['validation_loss'].append(valid_loss)
        evaluation_data['validation_perplexity'].append(perplexity)

        print("Train Epoch {} Loss - {} | Validation loss - {}".format(epoch, epoch_loss, valid_loss))
        print(f"perplexity on validation set: {perplexity}")
        seq = "Θέλω".split(' ')
        max_seq = 100
        print(
            f"Probabilistic predictions (top 5): {['Θέλω ' + predict(model, train_set, seq, max_seq, deterministic=False, no_unk=True, top_only=5) for i in range(5)]}")
        print("----------------------------------------")
    return evaluation_data

def predict_many(model, dataset, test_seq, max_seq, no_unk=False, top_only=False):
    """
    Make a forward-pass and predict the top words.

    :param model: an LSTM instance of class defined in model.py
    :param dataset: a dataset instance of class defined in dataset.py
    :param test_seq: a sequence of words to feed to the model. e.g. ['Δεν', 'έχει', 'σίδερα', 'η']
    :param max_seq: the max length of generated sequence
    :param deterministic: If true, the model will always return the most likely prediction
    :param no_unk: Prevent model from generating <unk> words
    :param top_only: int. If set, the model will chose only among the 'n' top words (weighted by their probability) for the next prediction
    :return: the original + predicted sequence as a string
    """
    model.eval()
    test_seq = " ".join(['#'] + test_seq)
    input = dataset.tokens_to_ids(test_seq)
    h, c = model.init_state(1)

    with torch.no_grad():
        values = []

        # Reinitialize weights
        h, c = model.init_state(1)

        # Prepare input
        length = len(input)
        x = torch.tensor(input)
        x = torch.nn.utils.rnn.pad_sequence([x], padding_value=dataset.padding_idx)

        # Predict next token probabilities
        out, (h, c) = model.forward(x.to(device), [length], (h, c))
        p = torch.nn.functional.softmax(out[-1].squeeze(0)).detach().cpu().numpy()

        indices = p.argsort()[::-1][0:top_only]
        decoded = [dataset.ids_to_tokens([int(idx)]) for idx in indices]

        return decoded