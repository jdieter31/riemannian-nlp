"""
Converts a stream of tokens into one of word pieces


"""
import json
import sys
from typing import Tuple, Dict, List, Counter
from zipfile import ZipFile


class WordPiecer:
    EOW = "</w>"  # How to pretty print EOWs
    UNK = "<?>"  # How to pretty print unknown characters.
    UNK_EOW = "<?></w>"  # How to pretty print unknown characters.

    def __init__(self, pieces: List[str], merges: List[Tuple[str, str]]):
        """
        :param pieces: A sequence of word pieces.
        :param merges: Combinations of word pieces that can be merged with the compression algorithm.
        """
        # We look up indices a lot, so interning strings is natural.
        self._pieces = [sys.intern(p) for p in pieces]
        self._indices = {p: i for i, p in enumerate(pieces)}
        self._merges = [(self._indices[x], self._indices[y], self._indices[x + y]) for x, y in
                        merges]
        self._merge_indices = {(x, y): i for i, (x, y, _) in enumerate(self._merges)}

        if self.UNK not in self._pieces:
            self._indices[self.UNK] = len(self._pieces)
            self._pieces.append(self.UNK)

        if self.UNK_EOW not in self._pieces:
            self._indices[self.UNK_EOW] = len(self._pieces)
            self._pieces.append(self.UNK_EOW)

    def __len__(self):
        """
        :return: Size of the vocabulary produced by WordPiecer.
        """
        return len(self._pieces)

    def __repr__(self):
        return f"<WordPiecer: {len(self._pieces)} pieces and {len(self._merges)} merges>"

    def _compress(self, indices: List[int]) -> bool:
        """
        Attempt to compress the given set of indices in place.
        :param indices:
        :return: true if we succeeded at compressing the indices
        """
        # Find the top-ranked merge to complete:
        NONE = len(self._merges) + 1

        merge_idx = min((self._merge_indices.get((indices[i], indices[i + 1]), NONE) for i in
                         range(len(indices) - 1)),
                        default=NONE)

        if merge_idx == NONE:
            return False

        (x, y, z) = self._merges[merge_idx]

        # Apply the chosen merge by deleting anything that matches its values
        for i, _ in enumerate(indices):
            if i == len(indices) - 1:
                break

            if indices[i] == x and indices[i + 1] == y:
                # pop i twice
                indices.pop(i)
                indices.pop(i)
                indices.insert(i, z)
        return True

    def encode_string(self, part: str, append_eow: bool = True) -> List[int]:
        """
        Encodes a single part using the merge algorithm.
        :param part: An arbitrary string.
        :param append_eow: if true, append '</w>' to the last token.
        :return: A sequence of piece indices
        """
        # 0. Handle empty string.
        if not part:
            return []

        # 1. Map part into a sequence of primary pieces indices that will be compressed below.
        indices: List[int] = []
        for char in part[:-1]:
            indices.append(self._indices.get(char, self._indices[self.UNK]))
        char = part[-1]
        if append_eow:
            indices.append(self._indices.get(char + self.EOW, self._indices[self.UNK_EOW]))
        else:
            unk_eow = self.UNK_EOW if char.endswith(self.EOW) else self.UNK
            indices.append(self._indices.get(char, self._indices[unk_eow]))

        # 2. Compress the indices in order of the merge indices
        while self._compress(indices):
            pass

        return indices

    def encode(self, tokens: List[str]) -> List[int]:
        ret = []
        for token in tokens:
            ret.extend(self.encode_string(token))
        return ret

    def decode(self, indices: List[int], strip_eow: bool = True) -> List[str]:
        """
        Convert given word piece indices into a set of tokens
        :param indices: a list of word piece indices
        :param strip_eow: remove the </w> tags
        :return: a list of tokens that correspond to @indices
        """
        ret = []
        word = ""
        for idx in indices:
            piece = self._pieces[idx]
            if piece.endswith(self.EOW):
                word += piece[:-len(self.EOW)] if strip_eow else piece
                ret.append(word)
                word = ""
            else:
                word += piece

        # NOTE: this is potentially incorrect because we might add a new word even without the </w> tag.
        if word:
            ret.append(word)

        return ret

    def to_string(self, indices: List[int]) -> str:
        """
        Converts a sequence of indices in to a readable string.
        :param indices:
        :return: A string corresponding to @indices.

        Example: 1 2 3 -> the
        """
        return " ".join(self.decode(indices, strip_eow=False))

    @classmethod
    def learn(cls, tokens_: List[str], character_threshold: int = 5,
              max_vocab: int = 100) -> 'WordPiecer':
        """
        Learns a word piecer from a sequence of tokens.
        :param tokens_: a sequence of strings to learn from. By default, all tokens are appended with EOW tags.
        :param character_threshold: the threshold on the number of times a character should have been seen to be added.
                                    a positive number has a regularizing effect.
        :param max_vocab: The maximum number of elements to add to the vocabulary before quitting.
        :return: A new word piecer that can be used for these purposes.
        """
        pieces: List[str] = [cls.UNK, cls.UNK_EOW]
        indices: Dict[str, int] = {p: i for i, p in enumerate(pieces)}
        merges: List[Tuple[str, str]] = []

        # 0. Minify the number of tokens actually considered by compressing this list of tokens by count.
        tokens = Counter[str](tokens_)

        # 1. Start by building a list of character to use.
        # 1.1 Compute character statistics to replace rare characters with UNK tokens.
        piece_stats = Counter[str]()
        # Start by adding characters as pieces.
        for token, freq in tokens.items():
            for char in token[:-1]:
                piece_stats[char] += freq
            piece_stats[token[-1] + cls.EOW] += freq

        # 1.2 Add these characters to the pieces
        for char, freq in piece_stats.most_common():
            if freq > character_threshold:
                indices[char] = len(pieces)
                pieces.append(char)

        assert len(
            pieces) < max_vocab, f"max_vocab ({max_vocab}) is too small to fit the {len(pieces)} characters"

        # 1.3 Replace sequences with integers indices.
        data: List[Tuple[List[int], int]] = []

        UNK_IDX, UNK_EOW_IDX = indices[cls.UNK], indices[cls.UNK_EOW]
        for token, freq in tokens.items():
            datum = [indices.get(char, UNK_IDX) for char in token[:-1]]
            datum.append(indices.get(token[-1] + cls.EOW, UNK_EOW_IDX))
            data.append((datum, freq))

        # 2. Merge bigrams to construct a vocabulary.
        merge_stats = Counter[Tuple[int, int]]()
        # 2.1 Build statistics for merges.
        for datum, freq in data:
            for i in range(len(datum) - 1):
                merge_stats[datum[i], datum[i + 1]] += freq

        # 2.2 While the vocabulary size can take it, keep merging tokens.
        while len(pieces) < max_vocab and merge_stats:
            # Get the most common merge
            ((x, y), _), = merge_stats.most_common(1)

            # Add this merge
            merges.append((pieces[x], pieces[y]))
            if pieces[x] + pieces[y] not in indices:
                indices[pieces[x] + pieces[y]] = len(indices)
                pieces.append(pieces[x] + pieces[y])
            z = indices[pieces[x] + pieces[y]]

            # Apply this merge to all of the data and update its statistics.
            for datum, freq in data:
                for i, _ in enumerate(datum):
                    # Handle corner case.
                    if i == len(datum) - 1:
                        break

                    # Aha, this is a candidate merge
                    if datum[i] == x and datum[i + 1] == y:
                        datum.pop(i)
                        datum.pop(i)
                        datum.insert(i, z)

                        # Update statistics
                        if i < len(datum) - 1:
                            merge_stats[y, datum[i + 1]] -= freq
                            merge_stats[z, datum[i + 1]] += freq

            # We will never see x, y again so remove it.
            merge_stats.pop((x, y))

        return cls(pieces, merges)

    def to_file(self, zf: ZipFile):
        zf.writestr("encoder/pieces.txt", "\n".join(self._pieces))
        zf.writestr("encoder/merges.txt", "\n".join([
            f"{self._pieces[x]}\t{self._pieces[y]}" for (x, y, _) in self._merges
        ]))

    __instance = None

    @classmethod
    def canonical(cls) -> 'WordPiecer':

        if cls.__instance is None:
            with ZipFile(props.auto.WORD_PIECER_PATH) as zf:
                cls.__instance = cls.from_file(zf)
        return cls.__instance

    @classmethod
    def from_file(cls, zf: ZipFile) -> 'WordPiecer':
        def as_tuple(ls: List[str]):
            return ls[0], ls[1]

        pieces = zf.read("encoder/pieces.txt").decode("utf-8").split("\n")
        merges = [as_tuple(line.split("\t")) for line in
                  zf.read("encoder/merges.txt").decode("utf-8").split("\n")]
        return cls(pieces, merges)

    @classmethod
    def from_openai(cls, encoder_path: str, merges_path: str) -> 'WordPiecer':
        with open(encoder_path) as f:
            encoder = json.load(f)
            pieces, idxs = zip(*sorted(encoder.items(), key=lambda kv: kv[-1]))
            # Make sure they're in sorted order.
            assert idxs[0] == 0 and idxs[-1] == len(pieces) - 1

        with open(merges_path) as f:
            merges = [(x, y) for line in f.readlines() for x, y in line.strip().split() if
                      not line.startswith("#")]

        return cls(pieces, merges)


__all__ = ['WordPiecer']
