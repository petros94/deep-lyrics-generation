from torch import utils
import torch
from collections import Counter
from bpemb import BPEmb


class LyricsDatasetRegex(utils.data.Dataset):
    """
    Lyrics Torch Dataset based on Regex tokenization.

    Input params
    -------------

    lyrics: list
        a list of lyrics where each lyric is a list of tokens.
        For example [['Δεν', 'εχει', 'σίδερα', 'η', 'καρδιά', 'σου', 'να', 'με', 'κλείσει']]
    vocab: list
        a list of tokens to use as a predefined vocabulary.
    sent_freq: float, between 0.0 and 1.0
        the percentage of unk tokens allowed in each lyric. If a lyrics exceeds the number of unk tokens,
        it won't be added to the final dataset
    token_freq: int between 1 and inf
        the minimum number of times a word should appear in the lyrics corpus. If a word appears less times
        than this number, it will be replaced by <unk>
    lowercase: boolean
        Whether to convert words to lowercase before tokenization.
    """

    def __init__(self, lyrics, vocab=None, sent_freq=1, token_freq=0, lowercase=False) -> None:
        super().__init__()
        s = self.append_start_end(lyrics)
        self.lowercase = lowercase
        tokenized_sentences, self.token_to_idx, self.idx_to_token, self.token_set = self.tokens_to_indices(s, vocab,
                                                                                                           token_freq)
        filtered_sentences = self.filter_sents(tokenized_sentences, sent_freq=sent_freq)

        self.padding_idx = 0
        self.dataset = self.create_dataset(filtered_sentences)
        print(f"Dataset samples: {len(self.dataset)}, vocabulary size: {len(self.token_set)} tokens")

    def tokens_to_indices(self, sentences, vocab, token_freq):
        if vocab:
            unique_token_list = vocab
        else:
            if self.lowercase:
                token_list = [t.lower() for s in sentences for t in s]
            else:
                token_list = [t for s in sentences for t in s]
            token_set = Counter(token_list)
            unique_token_list = [k for k, v in token_set.items() if v >= token_freq]
            unique_token_list = sorted(unique_token_list)
            unique_token_list.insert(0, '<pad>')
            unique_token_list.insert(1, '<unk>')

        self.token_to_idx = {ch: i for i, ch in enumerate(unique_token_list)}
        self.idx_to_token = {i: ch for i, ch in enumerate(unique_token_list)}

        return [[self.token_index(t) for t in s] for s in
                sentences], self.token_to_idx, self.idx_to_token, unique_token_list

    @staticmethod
    def append_start_end(sentences):
        return [['#'] + sent + ['&'] for sent in sentences]

    @staticmethod
    def create_dataset(tokenized_sentences):
        dataset = []
        for sent in tokenized_sentences:
            x = sent[:-1]
            y = sent[1:]
            dataset.append([x, y])
        return dataset

    def filter_sents(self, tokenized_sentences, sent_freq):
        filtered_sentences = []
        for i in range(len(tokenized_sentences)):
            c = 0
            for j, v in enumerate(tokenized_sentences[i]):
                if v == 1:
                    c += 1
            if c / len(tokenized_sentences[i]) <= sent_freq:
                filtered_sentences.append(tokenized_sentences[i])

        unk_counter = 0
        total_words = 0
        for s in filtered_sentences:
            for idx in s:
                total_words += 1
                if idx == self.token_index('<unk>'):
                    unk_counter += 1

        print(
            f'total tokens: {total_words}, unk tokens: {unk_counter}, percentage of unk tokens: {unk_counter / total_words * 100}%')
        print(f"Initial sentences: {len(tokenized_sentences)}, filtered sentences: {len(filtered_sentences)}")
        return filtered_sentences

    def token_index(self, ch):
        try:
            if self.lowercase:
                return self.token_to_idx[ch.lower()]
            else:
                return self.token_to_idx[ch]
        except Exception:
            return self.token_to_idx['<unk>']

    def ids_to_tokens(self, ids):
        return " ".join(list(map(lambda it: self.idx_to_token[it], ids)))

    def tokens_to_ids(self, tokens):
        tokens = tokens.split(' ')
        return list(map(lambda it: self.token_index(it), tokens))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (
            torch.tensor(self.dataset[index][0]),
            torch.tensor(self.dataset[index][1])
        )


class LyricsDatasetBPE(utils.data.Dataset):
    """
        Lyrics Torch Dataset based on Byte Pair Encoding tokenization.
        See: https://github.com/bheinzerling/bpemb

        Input params
        -------------

        lyrics: list
            a list of lyrics where each lyric is a list of tokens.
            For example [['Δεν', 'εχει', 'σίδερα', 'η', 'καρδιά', 'σου', 'να', 'με', 'κλείσει']]
        n_vocab: int
            The vocabulary size. The smaller the size, the more the tokens resemble character tokenization.
    """

    def __init__(self, lyrics, n_vocab=10000) -> None:
        super().__init__()
        s = self.append_start_end(lyrics)

        self.bpemb_el = BPEmb(lang="el", dim=100, vs=n_vocab, add_pad_emb=True)
        self.padding_idx = len(self.bpemb_el.emb) - 1

        self.dataset = self.create_dataset(s)
        self.n_vocab = self.bpemb_el.vs + 1
        print(f"Dataset samples: {len(self.dataset)}")

    @staticmethod
    def append_start_end(sentences):
        return [['#'] + sent + ['&'] for sent in sentences]

    def create_dataset(self, sentences):
        dataset = []
        for sent in sentences:
            x = " ".join([w.lower() for w in sent[:-1]])
            x = self.bpemb_el.encode_ids(x)

            y = " ".join([w.lower() for w in sent[1:]])
            y = self.bpemb_el.encode_ids(y)
            dataset.append([x, y])
        return dataset

    def embed(self, sentence):
        return self.bpemb_el.embed(sentence)

    def token_to_idx(self, token):
        return self.bpemb_el.encode_ids(token)[0]

    def tokens_to_ids(self, tokens):
        return self.bpemb_el.encode_ids(tokens)

    def ids_to_tokens(self, ids):
        return self.bpemb_el.decode_ids(ids)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (
            torch.tensor(self.dataset[index][0]),
            torch.tensor(self.dataset[index][1])
        )
