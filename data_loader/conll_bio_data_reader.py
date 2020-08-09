from typing import Dict, List, Iterable, Tuple
import itertools
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from transformers import AutoTokenizer

from data_loader.base_data_reader import BaseDataReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _convert_tags_to_wordpiece_tags(tags: List[str], offsets: List[int]) -> List[str]:
    """
    Converts a series of BIO tags to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.
    This is only used if you pass a `bert_model_name` to the dataset reader below.
    Parameters
    ----------
    tags : `List[str]`
        The BIO formatted tags to convert to BIO tags for wordpieces
    offsets : `List[int]`
        The wordpiece offsets.
    Returns
    -------
    The new BIO tags.
    """
    new_tags = []
    j = 0
    for i, offset in enumerate(offsets):
        tag = tags[i]
        is_o = tag == "O"
        is_start = True
        while j < offset:
            if is_o:
                new_tags.append("O")

            elif tag.startswith("I"):
                new_tags.append(tag)

            elif is_start and tag.startswith("B"):
                new_tags.append(tag)
                is_start = False

            elif tag.startswith("B"):
                _, label = tag.split("-", 1)
                new_tags.append("I-" + label)
            j += 1

    # Add O tags for cls and sep tokens.
    return ["O"] + new_tags + ["O"]


def _is_divider(line: str) -> bool:
    return line.strip() == ''


class ConllBIODataReader(BaseDataReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD POS-TAG CHUNK-TAG

    with a blank line indicating the end of each sentence
    and converts it into a ``Dataset`` suitable for sequence tagging.

    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values corresponding to the ``tag_label``
    values will get loaded into the ``"tags"`` ``SequenceLabelField``.
    And if you specify any ``feature_labels`` (you probably shouldn't),
    the corresponding values will get loaded into their own ``SequenceLabelField`` s.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``chunk``)
        Specify `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels: ``Sequence[str]``, optional (default=``()``)
        These labels will be loaded as features into the corresponding instance fields:
        ``pos`` -> ``pos_tags`` or ``chunk`` -> ``chunk_tags``.
        Each will have its own namespace: ``pos_tags`` or ``chunk_tags``.
        If you want to use one of the tags as a `feature` in your model, it should be
        specified here.
    coding_scheme: ``str``, optional (default=``BIO``)
        Specifies the coding scheme for ``chunk_labels``.
        Valid options are ``BIO`` and ``BIOUL``.  The ``BIO`` default maintains
        the original BIO scheme in the CoNLL 2000 chunking data.
        In the BIO scheme, B is a token starting a span, I is a token continuing a span, and
        O is a token outside of a span.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the chosen ``tag_label``.
    """

    def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
        """
        Convert a list of tokens to wordpiece tokens and offsets, as well as adding
        BERT CLS and SEP tokens to the begining and end of the sentence.
        A slight oddity with this function is that it also returns the wordpiece offsets
        corresponding to the _start_ of words as well as the end.
        We need both of these offsets (or at least, it's easiest to use both), because we need
        to convert the labels to tags using the end_offsets. However, when we are decoding a
        BIO sequence inside the SRL model itself, it's important that we use the start_offsets,
        because otherwise we might select an ill-formed BIO sequence from the BIO sequence on top of
        wordpieces (this happens in the case that a word is split into multiple word pieces,
        and then we take the last tag of the word, which might correspond to, e.g, I-V, which
        would not be allowed as it is not preceeded by a B tag).
        For example:
        `annotate` will be bert tokenized as ["anno", "##tate"].
        If this is tagged as [B-V, I-V] as it should be, we need to select the
        _first_ wordpiece label to be the label for the token, because otherwise
        we may end up with invalid tag sequences (we cannot start a new tag with an I).
        Returns
        -------
        wordpieces : List[str]
            The BERT wordpieces from the words in the sentence.
        end_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]`
            results in the end wordpiece of each word being chosen.
        start_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in start_offsets]`
            results in the start wordpiece of each word being chosen.
        """
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()
            word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)
        if len(word_piece_tokens) > 510:
            word_piece_tokens = word_piece_tokens[:510]
        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]

        return wordpieces, end_offsets, start_offsets

    def __init__(self, data_dir, batch_size, shuffle, small_data, embedder_type) -> None:
        self._token_indexers = {'bert': SingleIdTokenIndexer()}
        path = {
            "scibert": "embeddings/scibert_scivocab_uncased",
            "biobert": "embeddings/biobert_large",
            "bert": "embeddings/bert"
        }[embedder_type]
        self.bert_tokenizer = AutoTokenizer.from_pretrained(path)
        self.lowercase_input = True
        super().__init__(data_dir, batch_size, shuffle, small_data,
                         "train.txt", "dev.txt", "test.txt")

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    lines = list(lines)
                    if len(lines) == 0 or lines[0].startswith('-DOCSTART-'):
                        continue
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens, _, _, labels = fields
                    yield self.text_to_instance(tokens, labels)

    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         labels: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input(tokens)
        text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                               token_indexers=self._token_indexers)
        wordpiece_tags = _convert_tags_to_wordpiece_tags(labels, offsets)

        instance_fields: Dict[str, Field] = {'text': text_field,
                                             "metadata": MetadataField({"words": tokens}),
                                             'ner_bio': SequenceLabelField(wordpiece_tags, text_field,
                                                                           "ner_bio_labels")}
        # Add "feature labels" to instance
        return Instance(instance_fields)
