import numpy as np
from torch import Tensor


class Tokenizer:
    def __init__(self, vocab: str) -> None:
        """
        _summary_

        Parameters
        ----------
        vocab : str
            _description_, by default
        """

        self.vocab = vocab
        self.vocab_size = len(self.vocab) + 2
        self.numbers = np.arange(2, self.vocab_size)
        self.char_to_token = {char: i for i, char in enumerate(self.vocab, start=2)}
        self.token_to_char = {i: char for char, i in self.char_to_token.items()}
        self.token_to_char[0], self.token_to_char[1] = " ", "<end>"  # only for decoding

    def encode(self, text: str) -> list[int]:
        """
        _summary_

        Parameters
        ----------
        text : str
            _description_

        Returns
        -------
        list[int]
            _description_
        """

        tokens = [self.char_to_token.get(char, 2) for char in text]
        tokens.append(1)

        return tokens

    def decode(self, tokens: list[int] | Tensor) -> str:
        """
        _summary_

        Parameters
        ----------
        tokens : list[int] | Tensor
            _description_u

        Returns
        -------
        str
            _description_
        """
        if isinstance(tokens, Tensor):
            tokens = tokens.tolist()

        text_str = [self.token_to_char[token] for token in tokens]
        return "".join(text_str)

    def get_vocab_size(self) -> int:
        return self.vocab_size
