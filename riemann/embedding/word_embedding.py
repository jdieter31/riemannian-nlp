import os
import struct
import unicodedata
import zipfile
from typing import List, Sequence, Optional, Dict, Union, Iterable, BinaryIO
from zipfile import ZipFile

import numpy as np

from .token_mapper import TokenMapper, default_token_mapper, HashTokenMapping

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
DEFAULT_GLOVE_PATH = os.path.join(root_path, 'resources/glove.840B.300d.zip')
GLOVE_PATH = os.environ.get("GLOVE_PATH", DEFAULT_GLOVE_PATH)


class Glove:
    def __init__(
            self,
            token_mapper: TokenMapper,
            glove_name: str,
            embedding_dim: int,
            vocab: Sequence[str],
            numbers: np.ndarray,
            numbers_is_zero_padded=False,
    ):
        def create_vocab_dict(vocab_list):
            vocab_dict = dict.fromkeys(vocab_list)
            for i, word in enumerate(vocab_dict):
                vocab_dict[word] = i
            return vocab_dict

        self.token_mapper = token_mapper
        self._mapped_output_size = self.token_mapper.mapped_output_size()  # cache result for speed
        self.glove_name = glove_name
        self.embedding_dim = embedding_dim
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.vocab_dict = create_vocab_dict(self.vocab)
        self.numbers = numbers

        if not numbers_is_zero_padded:
            # Provide a 0 padding at the beginning because it makes it much easier on the pytorch end
            # The shape and layout of self.numbers is important for the pytorch GloveEmbedding layer to work correctly,
            # so be careful about changing / make sure TestGloveEmbedding passes.
            padded_numbers = np.zeros(shape=(self.numbers.shape[0] + 1, self.numbers.shape[1]),
                                      dtype=np.float32)
            padded_numbers[1:] = self.numbers
            self.numbers = padded_numbers

        assert self.numbers.shape[
                   1] == self.embedding_dim, "Embedding dim must be same as the last dimension of numbers matrix"
        assert self.numbers.shape[
                   0] == self.vocab_size + 1, "Numbers matrix is not the correct size."
        assert self.numbers.dtype == np.float32

    @classmethod
    def mock(cls):
        import string
        return Glove(
            token_mapper=TokenMapper([], unk_mappings=[HashTokenMapping(1)]),
            glove_name="mock_glove",
            embedding_dim=300,
            vocab=list(string.ascii_lowercase + string.ascii_uppercase + string.digits),
            numbers=np.random.rand(26 * 2 + 10, 300),
            numbers_is_zero_padded=False
        )

    __glove: Optional['Glove'] = None

    @classmethod
    def canonical(cls) -> 'Glove':
        if cls.__glove is None:
            cls.__glove = cls.from_binary()
        return cls.__glove

    @classmethod
    def from_binary(
            cls,
            embedding_file_: Union[BinaryIO, str] = None,
            # Default filled in below -- want to load it statically.
            token_mapper: TokenMapper = None
    ) -> 'Glove':
        """
        Just responsible for reading binary glove and storing basic parameters
        :param embedding_file_: the raw, original format Glove embeddings
        :param token_mapper: the token mapper to use, if None, it defaults to nn.special_token_mapper
        """
        embedding_file: str = embedding_file_ or GLOVE_PATH
        token_mapper = token_mapper or default_token_mapper()

        with ZipFile(embedding_file) as zf:
            print(f"Loading the embeddings from binary file ({zf.filename})")
            # Save the vocabulary as a list.
            with zf.open("vocab.txt", "r") as f:
                vocab = [v.strip().decode("utf-8") for v in f]

            with zf.open("numbers.npy", "r") as f:
                vectors = np.load(f).flatten()

            # Infer the shape of the vectors
            if len(vectors) % len(vocab) == 0:
                vocab_size = len(vocab)
                embedding_dim = len(vectors) // len(vocab)
                vectors = vectors.reshape(vocab_size, embedding_dim)
                is_zero_padded = False
            elif len(vectors) % (len(vocab) + 1) == 0:
                vocab_size = len(vocab)
                embedding_dim = len(vectors) // (len(vocab) + 1)
                vectors = vectors.reshape(vocab_size + 1, embedding_dim)
                is_zero_padded = True
            else:
                raise ValueError(f"We read a vocabulary of {len(vocab)} words,"
                                 f" which can't possibly divide the {len(vectors)} numbers we got!")

            print(f"Vocab size: {len(vocab)}")
            print(f"Embedding dim: {embedding_dim}")

            return cls(
                token_mapper=token_mapper if token_mapper else default_token_mapper(),
                glove_name=zf.filename,
                embedding_dim=embedding_dim,
                vocab=vocab,
                numbers=vectors,
                numbers_is_zero_padded=is_zero_padded)

    @classmethod
    def from_legacy_binary(
            cls,
            embedding_file_: str = None,  # Default filled in below -- want to load it statically.
            token_mapper: TokenMapper = None
    ) -> 'Glove':
        """
        Just responsible for reading binary glove and storing basic parameters
        :param embedding_file_: the raw, original format Glove embeddings
        :param token_mapper: the token mapper to use, if None, it defaults to nn.special_token_mapper
        """

        embedding_file: str = GLOVE_PATH if not embedding_file_ else embedding_file_
        token_mapper = default_token_mapper() if token_mapper is None else token_mapper
        glove_name = os.path.splitext(os.path.basename(embedding_file))[0]
        print('Loading the embeddings from binary file ({})'.format(embedding_file))

        def read_int(f_):
            return struct.unpack('>i', f_.read(4))[0]

        def read_bytes(f_, num_bytes):
            chunks = []
            while num_bytes > 0:
                to_read = min(10000000, num_bytes)
                chunk = f_.read(to_read)
                if len(chunk) == 0:
                    raise Exception(
                        'Got an empty chunk back! File terminated before expected! Still need ' + str(
                            num_bytes) + ' bytes')
                num_bytes -= len(chunk)
                chunks.append(chunk)
            return b''.join(chunks)

        def read_vocab(characters_, word_begins_, word_lengths_):
            vocab_ = []
            for begin, length in zip(word_begins_, word_lengths_):
                word = ''.join([chr(c) for c in characters_[begin:begin + length]])
                vocab_.append(word)
            return vocab_

        with open(embedding_file, 'rb') as f:
            embedding_dim = read_int(f)
            vocab_size = read_int(f)
            num_characters = read_int(f)
            characters = np.frombuffer(read_bytes(f, num_characters * 4), dtype='>i')
            num_word_begins = read_int(f)
            word_begins = np.frombuffer(read_bytes(f, num_word_begins * 4), dtype='>i')
            num_word_lengths = read_int(f)
            word_lengths = np.frombuffer(read_bytes(f, num_word_lengths), dtype='B')
            num_numbers = read_int(f)
            assert (num_numbers == num_word_lengths * 300)
            numbers = np.frombuffer(read_bytes(f, num_numbers * 4), dtype='>f')
            unk = np.frombuffer(read_bytes(f, embedding_dim * 4), dtype='>f')
            numbers = np.reshape(numbers, [vocab_size, embedding_dim]).astype(np.float32)
            vocab = read_vocab(characters, word_begins, word_lengths)

        print('Embedding dim: ' + str(embedding_dim))
        print('Vocab size: ' + str(vocab_size))
        # info('read start of characters array: '+str(characters))

        assert len(
            vocab) == vocab_size, "Length of vocabulary does not match state vocab size in {}".format(
            embedding_file)

        return cls(
            token_mapper=token_mapper,
            glove_name=glove_name,
            embedding_dim=embedding_dim,
            vocab=vocab,
            numbers=numbers,
        )

    def save(self, file: Union[BinaryIO, str], compress: bool = False):
        """
        Save glove embeddings to a binary file.
        :param file: where to save contents
        :param compress: where to save contents
        :return:
        """
        with ZipFile(file, "w",
                     compression=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED) as zf:
            # Save the vocabulary as a list.
            f = zf.open("vocab.txt", "w")
            for v in self.vocab:
                f.write(v.encode("utf8"))
                f.write(b"\n")
            f.close()

            f = zf.open("numbers.npy", "w", force_zip64=True)
            assert self.numbers.dtype == np.float32
            np.save(f, self.numbers.flatten())
            f.close()

    def with_new_token_mapper(self, token_mapper: TokenMapper, new_name: str = None) -> 'Glove':
        """
        Returns a new Glove that shares memory with this Glove that uses a different token mapper.
        :param token_mapper: The new token mapper to use
        :param new_name: (Optional) If given, we will write this new name to this glove
        :return:
        """
        return Glove(
            token_mapper,
            new_name if new_name else self.glove_name,
            self.embedding_dim,
            self.vocab,
            self.numbers,
            numbers_is_zero_padded=True
        )

    def lookup_word(self, word: str) -> int:
        """
        This searches through word indices to arrive at the index for the correct word index quickly.

        :param word: the word to search for
        :return: the index of the word
        """
        mapped_index = self.token_mapper.map_token(word)
        if mapped_index >= 0:
            return mapped_index
        if word in self.vocab_dict:
            index = self.vocab_dict[word]
            return index + self.token_mapper.mapped_output_size()
        else:
            return self.token_mapper.map_unk(word)

    def lookup_unk(self, word: str) -> int:
        """
        Returns the unk embedding index for any word
        :param word: The word
        :return: An integer index
        """
        return self.token_mapper.map_unk(word)

    def get_word_at_index(self, index: int) -> str:
        """
        This retrieves the word at the given index in our database

        :param index:
        :return:
        """
        if index < self.token_mapper.mapped_output_size():
            return self.token_mapper.debug_token(index)
        index -= self.token_mapper.mapped_output_size()
        return self.vocab[index]

    def words_to_indices(self, words: Iterable[str]) -> List[int]:
        """
        Returns a list of integer indices corresponding to each word in words. Unknown words will be replaced with UNK
        :param words: (list(string)) words
        :return: (list(int)) indices
        """
        return [self.lookup_word(token) for token in words]

    def indices_to_words(self, indices: Sequence[int]) -> List[str]:
        """
        Converts a list of indices to their corresponding words in Glove
        :param indices: (list(int)) indices
        :return: (list(string)) words
        """
        return [self.get_word_at_index(index) for index in indices]

    def get_embedding_at_index(self, index: int) -> np.ndarray:
        """
        This gets an embedding at a given index in our structures.

        :param index: the word index to lookup
        :return: the embedding
        """
        if index < self.token_mapper.mapped_output_size():
            return np.zeros((self.embedding_dim,))
        else:
            return self.numbers[index - self.token_mapper.mapped_output_size() + 1]

    def __len__(self) -> int:
        return self.vocab_size + self.token_mapper.mapped_output_size()

    def __iter__(self):
        return zip(self.vocab, range(self.vocab_size))

    def __contains__(self, key: Union[int, str]) -> bool:
        if isinstance(key, int):
            return 0 <= key < len(self)
        elif isinstance(key, str):
            mapped_index = self.token_mapper.map_token(key)
            if mapped_index >= 0:
                return True
            return key in self.vocab_dict
        else:
            raise ValueError(
                f"Tried to check containment of a {type(key)} which isn't an int or string")

    def __getitem__(self, key):
        if type(key) == int:
            return self.get_word_at_index(key)
        if type(key) == str:
            return self.lookup_word(self.normalize_word(key))

    def __call__(self, key):
        if type(key) == int:
            return self.get_embedding_at_index(key)
        if type(key) == str:
            return self.get_embedding_at_index(self.lookup_word(self.normalize_word(key)))

    def tokens(self) -> Sequence[str]:
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        return self.vocab

    @staticmethod
    def normalize_word(token: str):
        return unicodedata.normalize('NFD', token)


class TunedBinaryGlove:
    """
    The point of this class is to wrap a Glove, and allow certain custom words to be overwritten with new 
    embeddings without any disruption downstream.
    """

    def __init__(self, glove: Glove):
        self.glove = glove
        self.custom_words_vectors: Dict[int, np.ndarray] = {}
        self.custom_words_indices: Dict[str, int] = {}
        self.custom_words_back: Dict[int, str] = {}
        self.custom_unk: Optional[np.ndarray] = None

    def set_custom_word(self, word: str, vector: np.ndarray):
        """
        This adds a custom word to the dataset, which will overwrite any words previously listed
        
        :param word: the word to add
        :param vector: the vector to associate with this word
        """
        if word not in self.custom_words_indices:
            custom_index = len(self.custom_words_vectors) + len(self.glove)
            self.custom_words_indices[word] = custom_index
        else:
            custom_index = self.custom_words_indices[word]
        self.custom_words_vectors[custom_index] = vector
        self.custom_words_back[custom_index] = word

    def set_custom_unk(self, vector: np.ndarray):
        """
        If we want to override the UNK behavior with a new value, this is it.
        
        :param vector: 
        """
        self.custom_unk = vector

    def lookup_word(self, word: str) -> int:
        """
        This searches through word indices to arrive at the index for the correct word index quickly.

        :param word: the word to search for
        :return: the index of the word
        """
        if word in self.custom_words_indices:
            return self.custom_words_indices[word]
        index = self.glove.lookup_word(word)
        return index

    def get_word_at_index(self, index: int) -> str:
        """
        This retrieves the word at the given index in our database

        :param index:
        :return:
        """
        if index in self.custom_words_back:
            return self.custom_words_back[index]
        return self.glove.get_word_at_index(index)

    def words_to_indices(self, words: Sequence[str]) -> List[int]:
        """
        Returns a list of integer indices corresponding to each word in words. Unknown words will be replaced with UNK
        :param words: (list(string)) words
        :return: (list(int)) indices
        """
        return [self.lookup_word(token) for token in words]

    def indices_to_words(self, indices: Sequence[int]) -> List[str]:
        """
        Converts a list of indices to their corresponding words in Glove
        :param indices: (list(int)) indices
        :return: (list(string)) words
        """
        return [self.get_word_at_index(index) for index in indices]

    def get_embedding_at_index(self, index: int) -> np.ndarray:
        """
        This gets an embedding at a given index in our structures.

        :param index: the word index to lookup
        :return: the embedding
        """
        if index in self.custom_words_vectors:
            return self.custom_words_vectors[index]
        if self.glove.token_mapper.is_unk(index) and self.custom_unk is not None:
            return self.custom_unk
        return self.glove.get_embedding_at_index(index)

    def __len__(self) -> int:
        return len(self.glove) + len(self.custom_words_indices)

    def __iter__(self):
        return self.glove.__iter__()

    def __contains__(self, key) -> bool:
        if key in self.custom_words_indices:
            return True
        return self.glove.__contains__(key)

    def __getitem__(self, key):
        if type(key) == int:
            return self.get_word_at_index(key)
        if type(key) == str:
            return self.lookup_word(self.glove.normalize_word(key))
        return self.glove.__getitem__(key)

    def tokens(self) -> Sequence[str]:
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        return list(self.glove.vocab) + list(self.custom_words_indices.keys())


__all__ = ['Glove', 'TunedBinaryGlove']
