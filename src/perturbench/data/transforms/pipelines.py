import numpy as np

from .base import Dispatch, Compose
from .encoders import OneHotEncode, MultiLabelEncode, PerturbationEmbeddingEncode
from .ops import ToDense, ToFloat, MapApply


class SingleCellPipeline(Dispatch):
    """Single cell transform pipeline."""

    def __init__(
        self,
        perturbation_uniques: set[str],
        covariate_uniques: dict[str:set],
    ) -> None:
        # Set up covariates transform
        covariate_transform = {
            key: Compose([OneHotEncode(uniques), ToFloat()])
            for key, uniques in covariate_uniques.items()
        }
        # Initialize the pipeline
        super().__init__(
            perturbations=Compose(
                [
                    MultiLabelEncode(perturbation_uniques),
                    ToFloat(),
                ]
            ),
            gene_expression=ToDense(),
            covariates=MapApply(covariate_transform),
        )


class MoleculeEmbeddingPipeline(Dispatch):
    """Transform pipeline that encodes perturbations using molecule embeddings.

    Uses pre-computed molecule embeddings (e.g. ECFP fingerprints or LPM
    embeddings) instead of one-hot encoding for the perturbation field.
    """

    def __init__(
        self,
        perturbation_embedding_dict: dict[str, np.ndarray],
        covariate_uniques: dict[str, set],
        perturbation_control_value: str = "control",
    ) -> None:
        covariate_transform = {
            key: Compose([OneHotEncode(uniques), ToFloat()])
            for key, uniques in covariate_uniques.items()
        }
        super().__init__(
            perturbations=Compose(
                [
                    PerturbationEmbeddingEncode(
                        perturbation_embedding_dict,
                        control_label=perturbation_control_value,
                    ),
                    ToFloat(),
                ]
            ),
            gene_expression=ToDense(),
            covariates=MapApply(covariate_transform),
        )
