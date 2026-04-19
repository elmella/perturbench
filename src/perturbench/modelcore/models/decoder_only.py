"""
BSD 3-Clause License

Copyright (c) 2024, <anonymized authors of NeurIPS submission #1306>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np

from ..nn.mlp import MLP
from .base import PerturbationModel
from perturbench.data.types import Batch


class DecoderOnly(PerturbationModel):
    """
    A latent additive model for predicting perturbation effects
    """

    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        n_layers=2,
        encoder_width=128,
        softplus_output=True,
        use_covariates=True,
        use_perturbations=True,
        learnable_embeddings: bool = False,
        freeze_embeddings: bool = False,
        perturbation_embedding_path: str | None = None,
        perturbation_embedding_column: str | None = None,
        perturbation_embedding_name_column: str = "original_pert_name",
        perturbation_embedding_dataset: str | None = None,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
        datamodule: L.LightningDataModule | None = None,
    ) -> None:
        """
        The constructor for the DecoderOnly class.

        Args:
            n_genes (int): Number of genes to use for prediction
            n_perts (int): Number of perturbations in the dataset (not including controls)
            n_layers (int): Number of layers in the encoder/decoder
            lr (float): Learning rate
            wd (float): Weight decay
            lr_scheduler_freq (int): How often the learning rate scheduler checks val_loss
            lr_scheduler_patience (int): Learning rate scheduler patience
            lr_scheduler_factor (float): Factor by which to reduce learning rate when learning rate scheduler triggers
            softplus_output (bool): Whether to apply a softplus activation to the output of the decoder to enforce non-negativity
        """

        super(DecoderOnly, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
        )
        self.save_hyperparameters(ignore=["datamodule"])

        if not (use_covariates or use_perturbations):
            raise ValueError(
                "'use_covariates' and 'use_perturbations' can not both be false. Either covariates or perturbations have to be used."
            )

        if n_genes is not None:
            self.n_genes = n_genes
        if n_perts is not None:
            self.n_perts = n_perts

        n_total_covariates = (
            np.sum(
                [
                    len(unique_covs)
                    for unique_covs in datamodule.train_context[
                        "covariate_uniques"
                    ].values()
                ]
            )
            if use_covariates
            else 0
        )

        # Learnable perturbation embeddings
        self.pert_embedding = None
        pert_dim = self.n_perts if use_perturbations else 0

        if learnable_embeddings and perturbation_embedding_path is not None and datamodule is not None and use_perturbations:
            import pandas as pd
            df = pd.read_pickle(perturbation_embedding_path)
            if perturbation_embedding_dataset is not None and "dataset" in df.columns:
                df = df[df["dataset"] == perturbation_embedding_dataset]
            emb_dict = {}
            for _, row in df.iterrows():
                emb = row[perturbation_embedding_column]
                if emb is not None and not (isinstance(emb, float) and np.isnan(emb)):
                    emb_dict[str(row[perturbation_embedding_name_column]).strip()] = np.asarray(emb, dtype=np.float32)

            all_pert_names = getattr(datamodule, "all_perturbation_names",
                                     datamodule.train_context["perturbation_uniques"])
            emb_dim = next(iter(emb_dict.values())).shape[0]
            emb_matrix = torch.zeros(self.n_perts, emb_dim)
            for i, name in enumerate(all_pert_names):
                if name in emb_dict:
                    emb_matrix[i] = torch.from_numpy(emb_dict[name])
            self.pert_embedding = nn.Embedding.from_pretrained(
                emb_matrix, freeze=freeze_embeddings
            )
            pert_dim = emb_dim

        decoder_input_dim = n_total_covariates + pert_dim

        self.decoder = MLP(decoder_input_dim, encoder_width, self.n_genes, n_layers)
        self.softplus_output = softplus_output
        self.use_covariates = use_covariates
        self.use_perturbations = use_perturbations

    def forward(
        self,
        control_expression: torch.Tensor,
        perturbation: torch.Tensor,
        covariates: dict[str, torch.Tensor],
    ):
        if self.pert_embedding is not None:
            pert_indices = perturbation.argmax(dim=-1)
            pert_input = self.pert_embedding(pert_indices)
        else:
            pert_input = perturbation

        if self.use_covariates and self.use_perturbations:
            embedding = torch.cat([cov.squeeze() for cov in covariates.values()], dim=1)
            embedding = torch.cat([pert_input, embedding], dim=1)
        elif self.use_covariates:
            embedding = torch.cat([cov.squeeze() for cov in covariates.values()], dim=1)
        elif self.use_perturbations:
            embedding = pert_input

        predicted_perturbed_expression = self.decoder(embedding)

        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression

    def training_step(self, batch: Batch, batch_idx: int):
        (
            observed_perturbed_expression,
            control_expression,
            perturbation,
            covariates,
            _,
        ) = self.unpack_batch(batch)
        predicted_perturbed_expression = self.forward(
            control_expression, perturbation, covariates
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
            _,
        ) = self.unpack_batch(batch)
        predicted_perturbed_expression = self.forward(
            control_expression, perturbation, covariates
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

    def predict(self, batch):
        control_expression = batch.gene_expression.squeeze().to(self.device)
        perturbation = batch.perturbations.squeeze().to(self.device)
        covariates = {k: v.to(self.device) for k, v in batch.covariates.items()}

        predicted_perturbed_expression = self.forward(
            control_expression,
            perturbation,
            covariates,
        )
        return predicted_perturbed_expression
