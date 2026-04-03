import itertools
import functools
import logging
from typing import Collection, Sequence

import torch
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MultiLabelBinarizer

from .base import Transform
from ..types import ExampleMultiLabel, BatchMultiLabel

log = logging.getLogger(__name__)


class OneHotEncode(Transform):
    """One-hot encode a categorical variable.

    Attributes:
        onehot_encoder: the wrapped encoder instance
    """

    one_hot_encoder: OneHotEncoder

    def __init__(self, categories: Collection[str], **kwargs):
        categories = [list(categories)]
        self.one_hot_encoder = OneHotEncoder(
            categories=categories,
            sparse_output=False,
            **kwargs,
        )

    def __call__(self, labels: Sequence[str]):
        string_array = np.array(labels).reshape(-1, 1)
        encoded = self.one_hot_encoder.fit_transform(string_array)
        return torch.Tensor(encoded)

    def __repr__(self):
        _base = super().__repr__()
        categories = ", ".join(self.one_hot_encoder.categories[0])
        return _base.format(categories)


class LabelEncode(Transform):
    """Label encode categorical variables.

    Attributes:
        ordinal_encoder: sklearn.preprocessing.OrdinalEncoder
    """

    ordinal_encoder: OrdinalEncoder

    def __init__(self, values: Sequence[str]):
        categories = [np.array(values)]
        self.ordinal_encoder = OrdinalEncoder(categories=categories)

    def __call__(self, labels: Sequence[str]):
        string_array = np.array(labels).reshape(-1, 1)
        return torch.Tensor(self.ordinal_encoder.fit_transform(string_array))

    def __repr__(self):
        _base = super().__repr__()
        categories = ", ".join(self.ordinal_encoder.categories[0])
        return _base.format(categories)


class MultiLabelEncode(Transform):
    """Transforms a sequence of labels into a binary vector.

    Attributes:
        label_binarizer: the wrapped binarizer instance

    Raises:
        ValueError: if any of the labels are not found in the encoder classes
    """

    label_binarizer: MultiLabelBinarizer

    def __init__(self, classes: Collection[str]):
        self.label_binarizer = MultiLabelBinarizer(
            classes=list(classes), sparse_output=False
        )

    @functools.cached_property
    def classes(self):
        return set(self.label_binarizer.classes)

    def __call__(self, labels: ExampleMultiLabel | BatchMultiLabel) -> torch.Tensor:
        # If labels is a single example, convert it to a batch
        if not labels or isinstance(labels[0], str):
            labels = [labels]
        self._check_inputs(labels)
        encoded = self.label_binarizer.fit_transform(labels)
        return torch.from_numpy(encoded)

    def _check_inputs(self, labels: BatchMultiLabel):
        unique_labels = set(itertools.chain.from_iterable(labels))
        if not unique_labels <= self.classes:
            missing_labels = unique_labels - self.classes
            raise ValueError(
                f"Labels {missing_labels} not found in the encoder classes {self.classes}"
            )

    def __repr__(self):
        _base = super().__repr__()
        classes = ", ".join(self.label_binarizer.classes)
        return _base.format(classes)


class PerturbationEmbeddingEncode(Transform):
    """Encode perturbations using pre-computed molecule embeddings.

    Maps perturbation labels to continuous embedding vectors via a lookup
    dictionary. For combination perturbations (multiple labels), the
    embeddings are summed.

    Attributes:
        embedding_dict: mapping from perturbation name to embedding vector
        embedding_dim: dimensionality of the embedding vectors
    """

    def __init__(
        self,
        embedding_dict: dict[str, np.ndarray],
        control_label: str = "control",
    ):
        self.embedding_dict = embedding_dict
        self.control_label = control_label
        first_emb = next(iter(embedding_dict.values()))
        self.embedding_dim = first_emb.shape[0]
        self._zero = np.zeros(self.embedding_dim, dtype=first_emb.dtype)

    def _encode_single(self, labels: Sequence[str]) -> np.ndarray:
        """Encode a single cell's perturbation label(s) to an embedding."""
        if not labels or (len(labels) == 1 and labels[0].strip() == self.control_label):
            return self._zero.copy()

        embs = []
        for label in labels:
            label_clean = label.strip()
            if label_clean == self.control_label:
                continue
            if label_clean in self.embedding_dict:
                embs.append(self.embedding_dict[label_clean])
            elif label in self.embedding_dict:
                embs.append(self.embedding_dict[label])
            else:
                log.warning("No embedding found for perturbation '%s', using zeros", label)
                embs.append(self._zero)
        return np.sum(embs, axis=0) if embs else self._zero.copy()

    def __call__(
        self, labels: ExampleMultiLabel | BatchMultiLabel
    ) -> torch.Tensor:
        if not labels or isinstance(labels[0], str):
            labels = [labels]
        encoded = np.stack([self._encode_single(l) for l in labels])
        return torch.from_numpy(encoded)

    def __repr__(self):
        _base = super().__repr__()
        return _base.format(f"dim={self.embedding_dim}, n_perts={len(self.embedding_dict)}")
