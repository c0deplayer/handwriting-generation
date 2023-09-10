from typing import List, Union

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
        self.token_to_char[0], self.token_to_char[1] = (
            "<start>",
            "<end>",
        )  # only for decoding

    def encode(self, text: str) -> List[int]:
        """
        _summary_

        Parameters
        ----------
        text : str
            _description_

        Returns
        -------
        List[int]
            _description_
        """

        return [0] + [self.char_to_token.get(char, 3) for char in text] + [1]

    def decode(self, tokens: Union[List[int], Tensor]) -> str:
        """
        _summary_

        Parameters
        ----------
        tokens : Union[List[int], Tensor]
            _description_u

        Returns
        -------
        str
            _description_
        """
        if isinstance(tokens, Tensor):
            tokens = tokens.tolist()

        text_str = [
            self.token_to_char[token] for token in tokens if token not in (0, 1)
        ]
        return "".join(text_str)

    def get_vocab_size(self) -> int:
        return self.vocab_size
