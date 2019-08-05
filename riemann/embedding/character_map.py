from typing import List, Sequence, MutableMapping, Dict, Tuple
from .token_mapper import TokenMapper, HashTokenMapping
import numpy as np
from unidecode import unidecode
import unicodedata


class CharacterMap:
    """
    Character maps map text to a set of characters defined by {printable ascii} UNION {special unicode characters}.
    """

    # Starting index of permitted ascii characters
    PRINTABLE_ASCII_START: int = 32
    # Ending index (exclusive) of permitted ascii characters
    PRINTABLE_ASCII_END: int = 127

    def __init__(
            self,
            special_chars: Sequence[str] = (),
            num_unks: int=1,
            zero_pad: bool=False,
            collapse_digits: bool=False,
    ):
        """
        :param special_chars: Additional characters that we would like to create mappings for that are non - ascii
        :param num_unks: number of unk indices to map unknown characters to. Defaults to 1.
        :param zero_pad: If True, we do not use the 0'th index.
        :param collapse_digits: If True, we map all digits to the same index.
        """
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: List[str] = []
        self.zero_pad = zero_pad
        self.collapse_digits = collapse_digits
        if self.zero_pad:
            self.char_to_idx["<PAD>"] = 0
            self.idx_to_char.append("<PAD>")
        digit_idx = None
        for i in range(CharacterMap.PRINTABLE_ASCII_START, CharacterMap.PRINTABLE_ASCII_END):
            char: str = chr(i)
            if collapse_digits and char.isdigit():
                if digit_idx is None:
                    digit_idx = len(self.idx_to_char)
                    self.idx_to_char.append(char)
                self.char_to_idx[char] = digit_idx
            else:
                self.char_to_idx[char] = len(self.idx_to_char)
                self.idx_to_char.append(char)

        if special_chars is not None:
            for special_char in special_chars:
                if special_char not in self.char_to_idx:
                    self.char_to_idx[special_char] = len(self.idx_to_char)
                    self.idx_to_char.append(special_char)

        assert num_unks > 0, "Must have at least 1 unk character. {} was passed in...".format(num_unks)
        self.token_mapper = TokenMapper([], [HashTokenMapping(num_unks)])  # just have 1 unk char
        self._token_mapper_indent = self.token_mapper.mapped_output_size()

    def num_mappings(self) -> int:
        return len(self.idx_to_char) + self._token_mapper_indent

    def text_to_indices(self, text: str) -> List[int]:
        """
        Converts text to a list of indices
        :param text:
        :return:
        """
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char] + self._token_mapper_indent)
            else:
                ascii_chars = unidecode(char)
                if len(ascii_chars) == 0:
                    try:
                        ascii_chars = unicodedata.name(char)
                        ascii_chars = "_".join(ascii_chars.split(" "))
                    except ValueError:
                        ascii_chars = char  # this will get mapped to the tokenmapper anyways

                for ascii_char in ascii_chars:
                    if ascii_char in self.char_to_idx:
                        indices.append(self.char_to_idx[ascii_char] + self._token_mapper_indent)
                    else:
                        indices.append(self.token_mapper.map_unk(ascii_char))
        return indices

    def indices_to_text(self, indices: Sequence[int]) -> str:
        """
        Converts text to a list of indices
        :param indices:
        """
        chars = []
        for idx in indices:
            if 0 <= idx < self.num_mappings():
                if idx < self._token_mapper_indent:
                    chars.append(self.token_mapper.debug_token(idx))
                else:
                    chars.append(self.idx_to_char[idx - self._token_mapper_indent])
        return "".join(chars)

    def tokens_to_indices(self, tokens: Sequence[str]) -> List[List[int]]:
        """
        Converts a list of tokens to a list of list of character indices
        :param tokens:
        """
        indices = []
        for token in tokens:
            indices.append(self.text_to_indices(token))
        return indices

    def tokens_to_indices_padded(self, tokens: Sequence[str])-> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of tokens to a padded num_words x max_word_length numpy array of indices and a mask of 1s
        and 0s of the same shape.
        :param tokens: The tokens to be embedded
        :return: The character indices ndarray (dtype = np.int64), The character mask ndarray (dtype = np.float32)
        """
        char_indices_list = self.tokens_to_indices(tokens)
        max_num_chars = max(len(char_indices_tok) for char_indices_tok in char_indices_list)
        char_indices_arr = np.zeros((len(char_indices_list), max_num_chars), dtype=np.int64)
        char_indices_mask = np.zeros((len(char_indices_list), max_num_chars), dtype=np.float32)
        for i, char_indices_tok in enumerate(char_indices_list):
            char_indices_arr[i, :len(char_indices_tok)] = char_indices_tok
            char_indices_mask[i, :len(char_indices_tok)] = 1.0
        return char_indices_arr, char_indices_mask


__all__ = ['CharacterMap']
