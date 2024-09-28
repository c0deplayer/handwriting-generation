import numpy as np
from torch import Tensor


class Tokenizer:
    """
    This class provides methods for encoding and decoding text data for use in deep learning applications.
    It maps characters to tokens and vice versa, allowing text data to be converted into a numeric representation.

    Args:
        vocab (str): The vocabulary representing the characters used for encoding and decoding.

    Attributes:
        vocab (str): The vocabulary used for character mapping.
        vocab_size (int): The size of the vocabulary, including special tokens.
        numbers (numpy.ndarray): An array of numbers corresponding to tokens.
        char_to_token (dict[str, int]): A dictionary mapping characters to tokens.
        token_to_char (dict[int, str]): A dictionary mapping tokens to characters.
            token_to_char[0], token_to_char[1] are reserved for special tokens "<start>" and "<end>".

    Methods:
        encode(text: str) -> List[int]:
            Encode a text string into a list of tokens.

        decode(tokens: Union[List[int], Tensor]) -> str:
            Decode a list of tokens or a tensor into a text string.

        get_vocab_size() -> int:
            Get the size of the vocabulary.

    """

    def __init__(self, vocab: str) -> None:
        self.vocab = vocab
        self.vocab_size = len(self.vocab) + 2
        self.numbers = np.arange(2, self.vocab_size)
        self.char_to_token: dict[str, int] = {
            char: i for i, char in enumerate(self.vocab, start=2)
        }
        self.token_to_char: dict[int, str] = {
            i: char for char, i in self.char_to_token.items()
        }
        self.token_to_char[0], self.token_to_char[1] = (
            "<start>",
            "<end>",
        )

    def encode(self, text: str) -> list[int]:
        """
        Encode a text string into a list of tokens.

        Args:
            text (str): The text string to encode.

        Returns:
            List[int]: The list of tokens representing the encoded text.
        """
        return [0] + [self.char_to_token.get(char, 3) for char in text] + [1]

    def decode(self, tokens: list[int] | Tensor) -> str:
        """
        Decode a list of tokens or a tensor into a text string.

        Args:
            tokens (Union[List[int], Tensor]): The list of tokens or tensor to decode.

        Returns:
            str: The decoded text string.
        """
        if isinstance(tokens, Tensor):
            tokens = tokens.tolist()

        text_str = [
            self.token_to_char[token] for token in tokens if token not in (0, 1)
        ]
        return "".join(text_str)

    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return self.vocab_size
