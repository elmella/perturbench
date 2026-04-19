"""
Copyright (C) 2024  <anonymized authors of NeurIPS submission #1306>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np

from ..nn.mlp import MLP, MaskNet
from .base import PerturbationModel
from perturbench.data.types import Batch

log = logging.getLogger(__name__)


class LatentAdditive(PerturbationModel):
    """
    A latent additive model for predicting perturbation effects
    """

    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        n_layers: int = 2,
        encoder_width: int = 128,
        latent_dim: int = 32,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
        dropout: float | None = None,
        softplus_output: bool = True,
        sparse_additive_mechanism: bool = False,
        inject_covariates_encoder: bool = False,
        inject_covariates_decoder: bool = False,
        n_total_covariates: int | None = None,
        learnable_embeddings: bool = False,
        freeze_embeddings: bool = False,
        perturbation_embedding_path: str | None = None,
        perturbation_embedding_column: str | None = None,
        perturbation_embedding_name_column: str = "original_pert_name",
        perturbation_embedding_dataset: str | None = None,
        datamodule: L.LightningDataModule | None = None,
    ) -> None:
        """
        The constructor for the LatentAdditive class.

        Args:
            n_genes: Number of genes to use for prediction
            n_perts: Number of perturbations in the dataset
                (not including controls)
            n_layers: Number of layers in the encoder/decoder
            encoder_width: Width of the hidden layers in the encoder/decoder
            latent_dim: Dimension of the latent space
            lr: Learning rate
            wd: Weight decay
            lr_scheduler_freq: How often the learning rate scheduler checks
                val_loss
            lr_scheduler_interval: Whether the learning rate scheduler checks
                every epoch or step
            lr_scheduler_patience: Learning rate scheduler patience
            lr_scheduler_factor: Factor by which to reduce learning rate when
                learning rate scheduler triggers
            dropout: Dropout rate or None for no dropout.
            softplus_output: Whether to apply a softplus activation to the
                output of the decoder to enforce non-negativity
            inject_covariates_encoder: Whether to condition the encoder on
                covariates
            inject_covariates_decoder: Whether to condition the decoder on
                covariates
            datamodule: The datamodule used to train the model
        """
        super(LatentAdditive, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
        )

        self.save_hyperparameters(ignore=["datamodule"])
        if n_genes is not None:
            self.n_genes = n_genes
        if n_perts is not None:
            self.n_perts = n_perts

        if inject_covariates_encoder or inject_covariates_decoder:
            if datamodule is None or datamodule.train_context is None:
                raise ValueError(
                    "If inject_covariates is True, datamodule must be provided"
                )
            n_total_covariates = np.sum(
                [
                    len(unique_covs)
                    for unique_covs in datamodule.train_context[
                        "covariate_uniques"
                    ].values()
                ]
            )

        encoder_input_dim = (
            self.n_input_features + n_total_covariates
            if inject_covariates_encoder
            else self.n_input_features
        )
        decoder_input_dim = (
            latent_dim + n_total_covariates if inject_covariates_decoder else latent_dim
        )

        self.gene_encoder = MLP(
            encoder_input_dim, encoder_width, latent_dim, n_layers, dropout
        )
        self.decoder = MLP(
            decoder_input_dim, encoder_width, self.n_genes, n_layers, dropout
        )

        # Build learnable perturbation embedding if pretrained embeddings are available
        self.learnable_embeddings = learnable_embeddings
        self.pert_embedding = None
        pert_encoder_input_dim = self.n_perts

        if learnable_embeddings and perturbation_embedding_path is not None and datamodule is not None:
            import pandas as pd
            emb_dict = self._load_embedding_dict(
                perturbation_embedding_path,
                perturbation_embedding_column,
                perturbation_embedding_name_column,
                perturbation_embedding_dataset,
            )
            # Use full perturbation list (train + unseen) to match one-hot indices
            all_pert_names = getattr(datamodule, 'all_perturbation_names',
                                     datamodule.train_context["perturbation_uniques"])
            first_emb = next(iter(emb_dict.values()))
            emb_dim = first_emb.shape[0]
            emb_matrix = torch.zeros(self.n_perts, emb_dim)
            matched = 0
            for i, name in enumerate(all_pert_names):
                if i < self.n_perts and name in emb_dict:
                    emb_matrix[i] = torch.from_numpy(
                        np.asarray(emb_dict[name], dtype=np.float32)
                    )
                    matched += 1
            self.pert_embedding = nn.Embedding.from_pretrained(
                emb_matrix, freeze=freeze_embeddings
            )
            pert_encoder_input_dim = emb_dim
            log.info(
                "Initialized %d/%d learnable perturbation embeddings "
                "(dim=%d, freeze=%s) from %s [%s]",
                matched, len(all_pert_names), emb_dim, freeze_embeddings,
                perturbation_embedding_path, perturbation_embedding_column,
            )

        self.pert_encoder = MLP(
            pert_encoder_input_dim, encoder_width, latent_dim, n_layers, dropout
        )

        if sparse_additive_mechanism:
            self.mask_encoder = MaskNet(
                self.n_perts, encoder_width, latent_dim, n_layers
            )

        self.dropout = dropout
        self.softplus_output = softplus_output
        self.sparse_additive_mechanism = sparse_additive_mechanism
        self.inject_covariates_encoder = inject_covariates_encoder
        self.inject_covariates_decoder = inject_covariates_decoder

    @staticmethod
    def _load_embedding_dict(path, column, name_column, dataset_filter):
        """Load pretrained embeddings from a pickle file into a dict."""
        import pandas as pd
        df = pd.read_pickle(path)
        if dataset_filter is not None and "dataset" in df.columns:
            df = df[df["dataset"] == dataset_filter]
        emb_dict = {}
        for _, row in df.iterrows():
            emb = row[column]
            if emb is None or (isinstance(emb, float) and np.isnan(emb)):
                continue
            name = str(row[name_column]).strip()
            emb_dict[name] = np.asarray(emb, dtype=np.float32)
        return emb_dict

    def forward(
        self,
        control_input: torch.Tensor,
        perturbation: torch.Tensor,
        covariates: dict[str, torch.Tensor],
    ):
        if self.inject_covariates_encoder or self.inject_covariates_decoder:
            merged_covariates = torch.cat(
                [cov.squeeze() for cov in covariates.values()], dim=1
            )

        if self.inject_covariates_encoder:
            control_input = torch.cat([control_input, merged_covariates], dim=1)

        latent_control = self.gene_encoder(control_input)

        if self.pert_embedding is not None:
            # Convert one-hot perturbation to index, look up learnable embedding
            pert_indices = perturbation.argmax(dim=-1)
            pert_input = self.pert_embedding(pert_indices)
        else:
            pert_input = perturbation

        latent_perturbation = self.pert_encoder(pert_input)

        if self.sparse_additive_mechanism:
            mask = self.mask_encoder(perturbation)
            latent_perturbation = mask * latent_perturbation

        latent_perturbed = latent_control + latent_perturbation

        if self.inject_covariates_decoder:
            latent_perturbed = torch.cat([latent_perturbed, merged_covariates], dim=1)
        predicted_perturbed_expression = self.decoder(latent_perturbed)

        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression

    def training_step(self, batch: Batch, batch_idx: int):
        (
            observed_perturbed_expression,
            control_expression,
            perturbation,
            covariates,
            embeddings,
        ) = self.unpack_batch(batch)

        if embeddings is not None:
            control_input = embeddings
        else:
            control_input = control_expression

        predicted_perturbed_expression = self.forward(
            control_input, perturbation, covariates
        )
        loss = F.mse_loss(predicted_perturbed_expression, observed_perturbed_expression)
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        (
            observed_perturbed_expression,
            control_expression,
            perturbation,
            covariates,
            embeddings,
        ) = self.unpack_batch(batch)

        if embeddings is not None:
            control_input = embeddings
        else:
            control_input = control_expression

        predicted_perturbed_expression = self.forward(
            control_input, perturbation, covariates
        )
        val_loss = F.mse_loss(
            predicted_perturbed_expression, observed_perturbed_expression
        )
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        return val_loss

    def predict(self, batch: Batch):
        if batch.embeddings is not None:
            control_input = batch.embeddings.squeeze()
        else:
            control_input = batch.gene_expression.squeeze()

        perturbation = batch.perturbations.squeeze().to(self.device)
        covariates = {k: v.to(self.device) for k, v in batch.covariates.items()}

        predicted_perturbed_expression = self.forward(
            control_input,
            perturbation,
            covariates,
        )
        return predicted_perturbed_expression
