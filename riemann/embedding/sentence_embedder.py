from abc import ABC, abstractmethod
from typing import Set

import numpy as np

from .word_embedding import Glove
from .core_nlp import SimpleSentence
from .term_frequencies import TermFrequencies


class SentenceEmbedder(ABC):
    @abstractmethod
    def embed(self, sentence: SimpleSentence, **kwargs) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        pass


class GloveSentenceEmbedder(SentenceEmbedder):
    POS_WHITELIST = {
        "JJ",
        "JJR",
        "JJS",
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        # "PRP",
        # "PRP$",
        # "RB",
        # "RBR",
        # "RBS",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        # "WDT",
        # "WP",
        # "WRB",
    }
    LEMMA_BLACKLIST = {"there", "do", "be", "have", "use", "need", "want", "like", "such"}

    def __init__(
            self,
            glove: Glove,
            pos_whitelist: Set[str] = None,
            lemma_blacklist: Set[str] = None,
            term_frequencies: TermFrequencies = None,
            lemmatize: bool=True,
    ):
        self.glove = glove
        self.pos_whitelist = pos_whitelist if pos_whitelist is not None else GloveSentenceEmbedder.POS_WHITELIST
        self.lemma_blacklist = lemma_blacklist if lemma_blacklist is not None else GloveSentenceEmbedder.LEMMA_BLACKLIST
        self.lemmatize = lemmatize

        self.term_frequencies = term_frequencies

    @classmethod
    def canonical(cls) -> 'GloveSentenceEmbedder':
        return cls(
            glove=Glove.canonical(),
            term_frequencies=TermFrequencies.canonical()
        )

    @property
    def dim(self):
        return self.glove.embedding_dim

    def weight(self, token):
        """
        SIF reweighting as described in A Simple But Tough-to-Beat Baseline for Sentence Embeddings (Arora 2017)
        :param token:
        :return:
        """
        a = 0.001
        if self.term_frequencies is not None and token in self.term_frequencies:
            return a / (a + self.term_frequencies[token] / float(self.term_frequencies.total_count))
        else:
            return 1.0
        pass

    def include_in_embedding(self, lemma: str = None, pos: str = None):
        should_include = True
        should_include &= ((self.lemma_blacklist is None) or (lemma not in self.lemma_blacklist))
        should_include &= ((self.pos_whitelist is None) or (pos is None) or pos == '' or (pos in self.pos_whitelist))
        return should_include

    def embed(
            self,
            sentence: SimpleSentence,
            use_filter: bool = True,
            l2_normalize: bool = True,
            ignore_case: bool = True,
            verbose: bool = False,
            username: str = None,
            knol_id: str = None,
            **kwargs
    ) -> np.ndarray:
        """
        Embeds a sentence using its tokens, lemmas, and parts of speech
        :param sentence: A sentence loaded from corenlp
        :param use_filter: If we should filter out certain parts of speech or lemmas
        :param l2_normalize: If true, we will return only unit vectors (unless it is all 0)
        :param ignore_case: If true, we will lower case every token and lemma before finding the embedding vector
        :param verbose: If verbose, log warnings.
        :param username: The username of the user adding the question.
                         This is purely for debugging, and is safe to ignore.
        :param knol_id: The ID of the Knol we're adding. This is purely for debugging, and is safe to ignore.

        :return: A vector corresponding to the embedded question.
        """
        vec = np.zeros(self.glove.embedding_dim, dtype=np.float32)
        valid_tokens = 0
        for token, lemma, pos in zip(sentence.original_texts(), sentence.lemmas(), sentence.pos()):
            if ignore_case:
                lemma = lemma.lower()
                token = token.lower()
            if lemma == "" or pos == "":
                if verbose:
                    print("Got an empty POS tag and/or Lemma. Setting lemma to be token to compensate. username=" +
                         str(username) + "; knol_id=" + str(knol_id))
                lemma = token
            if not use_filter or self.include_in_embedding(lemma, pos):
                idx = self.glove.lookup_word(lemma if self.lemmatize else token)
                if idx >= self.glove.token_mapper.mapped_output_size():
                    valid_tokens += 1
                    weight = self.weight(lemma if self.lemmatize else token)
                    vec += weight * self.glove.get_embedding_at_index(idx)

        if valid_tokens > 0:
            if l2_normalize:
                return self._normalize(vec)
            else:
                return vec / valid_tokens
        elif use_filter:
            # back off to a non filtered embedding if no valid tokens were used and we have a 0 vector
            return self.embed(sentence, use_filter=False, l2_normalize=l2_normalize, ignore_case=ignore_case,
                              verbose=verbose, username=username, knol_id=knol_id)
        else:
            return vec

    @classmethod
    def _normalize(cls, vec: np.ndarray, epsilon=1e-12) -> np.ndarray:
        """
        Normalize a vector
        :param vec: A one dimensional array
        :return:
        """
        length = np.power(np.sum(np.power(vec, 2)), 0.5)
        return vec / (length + epsilon)

    @classmethod
    def mock(cls):
        return cls(glove=Glove.mock(), term_frequencies=TermFrequencies.mock())


__all__ = ['SentenceEmbedder', 'GloveSentenceEmbedder']
