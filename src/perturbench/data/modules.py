import logging
import warnings
from pathlib import Path
from collections.abc import Callable, Mapping
import gc

import numpy as np
import pandas as pd
from omegaconf import DictConfig
import lightning as L
import scanpy as sc
from torch.utils.data import DataLoader
from perturbench.modelcore.utils import (
    instantiate_with_context,
)

from .datasets import (
    CounterfactualWithReference,
    SingleCellPerturbation,
    SingleCellPerturbationWithControls,
)
from .utils import batch_dataloader
from .collate import noop_collate
import perturbench.data.datasplitter as datasplitter

log = logging.getLogger(__name__)


class AnnDataLitModule(L.LightningDataModule):
    """AnnData Data Module for Perturbation Prediction Models."""

    def __init__(
        self,
        datapath: Path,
        add_controls: str,
        perturbation_key: str,
        perturbation_control_value: str,
        batch_size: int,
        perturbation_combination_delimiter: str | None,
        covariate_keys: list[str] | None = None,
        splitter: DictConfig | None = None,
        num_workers: int = 0,
        num_val_workers: int | None = None,
        num_test_workers: int | None = None,
        transform: DictConfig | None = None,
        collate: Mapping[str, Callable] | None = None,
        batch_sample: bool = True,
        evaluation: DictConfig | None = None,
        use_counts: bool = False,
        embedding_key: str | None = None,
        perturbation_embedding_path: str | None = None,
        perturbation_embedding_column: str | None = None,
        perturbation_embedding_name_column: str = "original_pert_name",
        perturbation_embedding_dataset: str | None = None,
        perturbation_embedding_name_remap: dict[str, str] | None = None,
        perturbation_subset_path: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_val_workers = (
            num_val_workers if num_val_workers is not None else num_workers
        )
        self.num_test_workers = (
            num_test_workers if num_test_workers is not None else num_workers
        )
        self.batch_sample = batch_sample
        self.split = None  ## Data split as a pandas series

        # Select appropriate dataset class and common arguments
        if add_controls:
            dataset_class = SingleCellPerturbationWithControls
        else:
            dataset_class = SingleCellPerturbation

        data_kwargs = {
            "perturbation_key": perturbation_key,
            "perturbation_combination_delimiter": perturbation_combination_delimiter,
            "covariate_keys": covariate_keys,
            "perturbation_control_value": perturbation_control_value,
            "embedding_key": embedding_key,
        }

        # Load AnnData object
        adata = sc.read_h5ad(datapath)
        if use_counts:
            adata.X = adata.layers["counts"]
        adata.raw = None
        if "counts" in adata.layers:
            del adata.layers["counts"]

        # Load molecule embeddings if specified, and filter to shared perturbations
        self._perturbation_embedding_dict = None
        self._perturbation_feature_dim = None
        if perturbation_embedding_path is not None:
            self._perturbation_embedding_dict = self._load_perturbation_embeddings(
                path=perturbation_embedding_path,
                column=perturbation_embedding_column,
                name_column=perturbation_embedding_name_column,
                dataset_filter=perturbation_embedding_dataset,
                perturbation_control_value=perturbation_control_value,
                name_remap=perturbation_embedding_name_remap,
            )
            first_emb = next(iter(self._perturbation_embedding_dict.values()))
            self._perturbation_feature_dim = first_emb.shape[0]

            # Filter adata to perturbations present in the embedding dict
            embedding_names = set(self._perturbation_embedding_dict.keys())
            pert_col = adata.obs[perturbation_key]
            keep_mask = pert_col.isin(embedding_names) | (
                pert_col == perturbation_control_value
            )
            n_before = adata.n_obs
            adata = adata[keep_mask].copy()
            n_dropped = n_before - adata.n_obs
            if n_dropped > 0:
                log.info(
                    "Filtered %d cells with perturbations missing from embeddings "
                    "(%d → %d cells)",
                    n_dropped, n_before, adata.n_obs,
                )

            log.info(
                "Loaded %d perturbation embeddings (dim=%d) from %s [%s]",
                len(self._perturbation_embedding_dict),
                self._perturbation_feature_dim,
                perturbation_embedding_path,
                perturbation_embedding_column,
            )

        # Filter to a pre-defined subset of perturbations (for fair comparisons)
        if perturbation_subset_path is not None:
            subset = set(
                line.strip()
                for line in Path(perturbation_subset_path).read_text().splitlines()
                if line.strip()
            )
            pert_col = adata.obs[perturbation_key]
            keep_mask = pert_col.isin(subset) | (
                pert_col == perturbation_control_value
            )
            n_before = adata.n_obs
            adata = adata[keep_mask].copy()
            log.info(
                "Restricted to %d perturbation subset from %s (%d → %d cells)",
                len(subset), perturbation_subset_path, n_before, adata.n_obs,
            )

        # Split intro train, val, test datasets
        if splitter is not None:
            # AnnData Split
            split_dict = datasplitter.PerturbationDataSplitter.split_dataset(
                splitter_config=splitter,
                obs_dataframe=adata.obs,
                perturbation_key=perturbation_key,
                perturbation_combination_delimiter=perturbation_combination_delimiter,
                perturbation_control_value=perturbation_control_value,
            )

            train_adata = adata[split_dict["train"]]
            val_adata = adata[split_dict["val"]]
            test_adata = adata[split_dict["test"]]

        else:
            train_adata = adata
            val_adata = None
            test_adata = None

        # Create datasets
        self.train_dataset, train_context = dataset_class.from_anndata(
            train_adata, **data_kwargs
        )

        self.val_dataset, val_context = (
            dataset_class.from_anndata(val_adata, **data_kwargs)
            if val_adata is not None
            else (None, None)
        )

        if evaluation.split_value_to_evaluate == "train":
            test_adata = train_adata
        elif evaluation.split_value_to_evaluate == "val":
            test_adata = val_adata
        self.test_dataset, test_context = (
            CounterfactualWithReference.from_anndata(
                test_adata,
                seed=seed,
                max_control_cells_per_covariate=evaluation.max_control_cells_per_covariate,
                **data_kwargs,
            )
            if test_adata is not None
            else (None, None)
        )
        self.train_context = train_context
        self.evaluation = evaluation

        if self._perturbation_embedding_dict is not None:
            train_context["perturbation_embedding_dict"] = self._perturbation_embedding_dict
            train_context["perturbation_control_value"] = perturbation_control_value

        # Verify that train, val, test datasets have the same perturbations and covariates
        self._verify_splits(train_context, val_context, test_context)

        # Build transform context. Keep default behavior unchanged, and only
        # expand one-hot perturbation classes when val/test contain perturbations
        # not present in train (unseen perturbation task).
        transform_context = train_context
        train_perturbation_uniques = list(train_context["perturbation_uniques"])
        merged_perturbation_uniques = list(train_perturbation_uniques)
        seen_perturbations = set(train_perturbation_uniques)
        for split_context in (val_context, test_context):
            if split_context is None:
                continue
            for perturbation in split_context["perturbation_uniques"]:
                if perturbation not in seen_perturbations:
                    seen_perturbations.add(perturbation)
                    merged_perturbation_uniques.append(perturbation)
        if len(merged_perturbation_uniques) != len(train_perturbation_uniques):
            log.info(
                "Unseen perturbation split detected. Expanding perturbation "
                "encoder classes from %d (train) to %d (train+val+test).",
                len(train_perturbation_uniques),
                len(merged_perturbation_uniques),
            )
            transform_context = dict(train_context)
            transform_context["perturbation_uniques"] = merged_perturbation_uniques

        # Build an example/batch transform pipeline from transform context
        if transform is not None:
            transform_pipeline = instantiate_with_context(transform, transform_context)

            # Set transform pipeline for each dataset
            self.train_dataset.transform = transform_pipeline
            self.val_dataset.transform = transform_pipeline
            self.test_dataset.transform = transform_pipeline

        # Build example collation function
        if self.batch_sample is True:
            self.example_collate_fn = noop_collate()
        else:
            self.example_collate_fn = collate

        if self._perturbation_feature_dim is not None:
            self.num_perturbations = self._perturbation_feature_dim
        else:
            self.num_perturbations = len(transform_context["perturbation_uniques"])

        # Cleanup
        del adata, train_adata
        if val_adata is not None:
            del val_adata
        if test_adata is not None:
            del test_adata
        gc.collect()

    @property
    def num_genes(self) -> int:
        """Number of genes in the dataset."""
        return len(self.train_dataset.gene_names)

    @property
    def embedding_width(self) -> int | None:
        """Width of the embeddings."""
        if self.train_dataset.embeddings is None:
            return None
        else:
            return self.train_dataset.embeddings.shape[1]

    @staticmethod
    def _load_perturbation_embeddings(
        path: str,
        column: str,
        name_column: str,
        dataset_filter: str | None,
        perturbation_control_value: str,
        name_remap: dict[str, str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Load pre-computed perturbation embeddings from a pickle file.

        Args:
            path: Path to the pickle file containing a DataFrame with
                perturbation embeddings.
            column: Column name containing the embedding vectors
                (e.g. 'LPM_emb' or 'ECFP:2').
            name_column: Column containing the perturbation name to use as
                the lookup key (should match names in the dataset's
                perturbation_key column).
            dataset_filter: If set, filter the DataFrame to rows matching
                this value in the 'dataset' column.
            perturbation_control_value: The control label in the dataset
                (used to remap DMSO/vehicle entries).
            name_remap: Optional mapping from embedding names to dataset
                names (e.g. {'(+)-JQ1': 'JQ1'} when the dataset remapped
                names during curation).

        Returns:
            A dict mapping perturbation name to embedding vector.
        """
        df = pd.read_pickle(path)
        if dataset_filter is not None and "dataset" in df.columns:
            df = df[df["dataset"] == dataset_filter]

        if name_remap is None:
            name_remap = {}

        embedding_dict: dict[str, np.ndarray] = {}
        for _, row in df.iterrows():
            emb = row[column]
            if emb is None or (isinstance(emb, float) and np.isnan(emb)):
                continue
            original_name = str(row[name_column]).strip()
            # Map control synonyms
            if original_name == perturbation_control_value or original_name in (
                "DMSO", "dmso", "vehicle",
            ):
                continue
            emb_array = np.asarray(emb, dtype=np.float32)
            # Apply name remapping to match dataset names
            mapped_name = name_remap.get(original_name, original_name)
            # Store under both original and remapped names so lookups
            # work regardless of which variant appears in the dataset
            for name in {original_name, mapped_name}:
                if name in embedding_dict:
                    embedding_dict[name] = (embedding_dict[name] + emb_array) / 2.0
                else:
                    embedding_dict[name] = emb_array

        if not embedding_dict:
            raise ValueError(
                f"No valid embeddings found in {path} "
                f"(column={column}, dataset={dataset_filter})"
            )
        return embedding_dict

    @staticmethod
    def _verify_splits(train_info: dict, val_info: dict | None, test_info: dict | None):
        for split, info in [("val", val_info), ("test", test_info)]:
            if info is not None:
                if not set(train_info["perturbation_uniques"]) >= set(
                    info["perturbation_uniques"]
                ):
                    unseen = set(info["perturbation_uniques"]) - set(
                        train_info["perturbation_uniques"]
                    )
                    warnings.warn(
                        f"{split} dataset contains {len(unseen)} perturbations "
                        f"not in train (unseen perturbation task)."
                    )
                if set(train_info["perturbation_uniques"]) != set(
                    info["perturbation_uniques"]
                ):
                    warnings.warn(
                        f"{split} dataset is missing perturbations from train dataset."
                    )

                if not set(train_info["covariate_uniques"]) >= set(
                    info["covariate_uniques"]
                ):
                    raise RuntimeError(
                        f"Train dataset must contain all covariates in {split} dataset."
                    )
                if set(train_info["covariate_uniques"]) != set(
                    info["covariate_uniques"]
                ):
                    warnings.warn(
                        f"{split} dataset is missing covariates from train dataset."
                    )

    def train_dataloader(self) -> DataLoader:
        if self.batch_sample:
            return batch_dataloader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.example_collate_fn,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.example_collate_fn,
                shuffle=True,
            )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None
        else:
            if self.batch_sample:
                return batch_dataloader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_val_workers,
                    collate_fn=self.example_collate_fn,
                )
            else:
                return DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_val_workers,
                    collate_fn=self.example_collate_fn,
                    shuffle=False,
                )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_dataset is None:
            return None
        else:
            return batch_dataloader(
                self.test_dataset,
                batch_size=self.evaluation.chunk_size,
                num_workers=self.num_test_workers,
                shuffle=False,
                collate_fn=self.example_collate_fn,
            )
