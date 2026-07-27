"""Tests for the concatenation architecture (issue #36 Phase 2).

All tests use ``from_foundation=False`` for both the auxiliary encoder and
the concatenation model, so no network download or cached CheMeleon
checkpoint is required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from chemprop.data import BatchMolGraph, MoleculeDatapoint, MoleculeDataset

from moal.auxiliary_encoder import AuxiliaryEncoderModule
from moal.concatenation_model import (
    ConcatenationChemPropLightningModule,
    build_concatenation_features,
    concatenation_feature_dim,
)
from moal.config import AuxiliaryModelConfig
from moal.types import CensoringType, LabelRecord, QueryType

_EMBEDDING_DIM = 16


@pytest.fixture
def aux_encoder() -> AuxiliaryEncoderModule:
    """Small random-init auxiliary encoder with two tasks."""
    config = AuxiliaryModelConfig(
        from_foundation=False,
        message_hidden_dim=_EMBEDDING_DIM,
        ffn_hidden_dim=16,
        depth=1,
    )
    return AuxiliaryEncoderModule(task_names=["log2fc_1um", "pic50"], config=config)


def _fast_model(concat_feature_dim: int, **overrides) -> ConcatenationChemPropLightningModule:
    defaults = {
        "from_foundation": False,
        "message_hidden_dim": 16,
        "ffn_hidden_dim": 16,
        "depth": 1,
        "freeze_epochs": 0,
    }
    defaults.update(overrides)
    return ConcatenationChemPropLightningModule(concat_feature_dim=concat_feature_dim, **defaults)


def _records() -> list[LabelRecord]:
    smiles = ["CCO", "CCN", "CCC", "c1ccccc1"]
    readouts = [{"log2fc_1um": 2.1}, {}, {"pic50": 6.4}, {}]
    records = []
    for smi, readout in zip(smiles, readouts, strict=True):
        records.append(
            LabelRecord(
                smiles=smi,
                canonical_smiles=smi,
                value=6.0,
                upper_bound=6.0,
                censoring_type=CensoringType.EXACT,
                fidelity=QueryType.DOSE_RESPONSE,
                cost=10.0,
                iteration=0,
                raw_ps_readouts=readout,
            )
        )
    return records


class TestConcatenationFeatureDim:
    """Tests for concatenation_feature_dim's arithmetic."""

    def test_matches_2n_plus_embedding_plus_1(self):
        """The formula must be 2 * n_tasks + embedding_dim + 1."""
        assert concatenation_feature_dim(n_tasks=3, embedding_dim=10) == 2 * 3 + 10 + 1


class TestBuildConcatenationFeatures:
    """Tests for build_concatenation_features: observed vs embedding routing and shape."""

    def test_observed_readout_also_gets_embedding(self, aux_encoder):
        """A compound with a readout must populate readout/mask AND the embedding block, with flag=1."""
        features = build_concatenation_features(["CCO"], [{"log2fc_1um": 2.5}], aux_encoder)
        n_tasks = 2

        readout_block = features[0, :n_tasks]
        mask_block = features[0, n_tasks : 2 * n_tasks]
        embedding_block = features[0, 2 * n_tasks : 2 * n_tasks + _EMBEDDING_DIM]
        flag = features[0, -1]

        assert readout_block[0] == 2.5
        assert list(mask_block) == [1.0, 0.0]
        assert not np.all(embedding_block == 0.0)
        assert flag == 1.0

    def test_missing_readout_uses_embedding_only(self, aux_encoder):
        """A compound with an empty readout dict must leave the readout/mask block zero, populate the embedding block, and flag=0."""
        features = build_concatenation_features(["CCO"], [{}], aux_encoder)
        n_tasks = 2

        readout_mask_block = features[0, : 2 * n_tasks]
        embedding_block = features[0, 2 * n_tasks : 2 * n_tasks + _EMBEDDING_DIM]
        flag = features[0, -1]

        assert np.all(readout_mask_block == 0.0)
        assert not np.all(embedding_block == 0.0)
        assert flag == 0.0

    def test_use_observed_readout_false_zeroes_readout_block_but_keeps_embedding(self, aux_encoder):
        """With use_observed_readout=False, every compound is embedding-only regardless of its own readout data."""
        with_readout = build_concatenation_features(
            ["CCO"], [{"log2fc_1um": 2.5}], aux_encoder, use_observed_readout=False
        )
        without_readout = build_concatenation_features(
            ["CCO"], [{}], aux_encoder, use_observed_readout=False
        )
        n_tasks = 2

        assert np.all(with_readout[0, : 2 * n_tasks] == 0.0)
        assert with_readout[0, -1] == 0.0
        np.testing.assert_allclose(
            with_readout[0, 2 * n_tasks : 2 * n_tasks + _EMBEDDING_DIM],
            without_readout[0, 2 * n_tasks : 2 * n_tasks + _EMBEDDING_DIM],
        )

    def test_output_shape_matches_concatenation_feature_dim(self, aux_encoder):
        """Output width must equal concatenation_feature_dim(n_tasks, embedding_dim)."""
        features = build_concatenation_features(
            ["CCO", "CCN", "CCC"], [{"log2fc_1um": 1.0}, {}, {"pic50": 5.0}], aux_encoder
        )

        assert features.shape == (3, concatenation_feature_dim(2, _EMBEDDING_DIM))

    def test_mismatched_lengths_raises(self, aux_encoder):
        """canonical_smiles and readouts of different lengths must raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            build_concatenation_features(["CCO", "CCN"], [{}], aux_encoder)


class TestConcatenationChemPropLightningModule:
    """Tests for training and prediction through the concatenation architecture."""

    def test_forward_output_shape(self, aux_encoder):
        """A forward pass must return one scalar prediction per input molecule."""
        feat_dim = concatenation_feature_dim(2, _EMBEDDING_DIM)
        model = _fast_model(feat_dim)
        dataset = MoleculeDataset([MoleculeDatapoint.from_smi(s) for s in ["CCO", "CCN"]])
        bmg = BatchMolGraph([dataset[i].mg for i in range(len(dataset))])
        x_d = torch.zeros(2, feat_dim)

        preds = model(bmg, x_d)

        assert preds.shape == (2,)

    def test_refit_and_predict_smiles_round_trip(self, aux_encoder):
        """refit() must train without error and predict_smiles() must return one prediction per input SMILES."""
        feat_dim = concatenation_feature_dim(2, _EMBEDDING_DIM)
        model = _fast_model(feat_dim)
        records = _records()

        model.refit(
            records,
            aux_encoder=aux_encoder,
            max_epochs=1,
            datamodule_kwargs={"val_fraction": 0.25, "seed": 1},
        )

        smiles = [r.canonical_smiles for r in records]
        readouts = [r.raw_ps_readouts for r in records]
        preds = model.predict_smiles(smiles, readouts, aux_encoder)

        assert preds.shape == (len(records),)
        assert np.all(np.isfinite(preds))

    def test_predict_smiles_chunks_correctly_across_batch_boundary(self, aux_encoder):
        """predict_smiles must produce one prediction per SMILES even when batch_size splits the input into multiple chunks."""
        feat_dim = concatenation_feature_dim(2, _EMBEDDING_DIM)
        model = _fast_model(feat_dim)
        smiles = ["CCO", "CCN", "CCC", "CCCC", "CCCCC"]
        readouts = [{"log2fc_1um": float(i)} for i in range(len(smiles))]

        preds = model.predict_smiles(smiles, readouts, aux_encoder, batch_size=2)

        assert preds.shape == (5,)

    def test_refit_rejects_non_drc_records(self, aux_encoder):
        """refit() must reject any record whose fidelity is not DOSE_RESPONSE."""
        feat_dim = concatenation_feature_dim(2, _EMBEDDING_DIM)
        model = _fast_model(feat_dim)
        ps_record = LabelRecord(
            smiles="CCO",
            canonical_smiles="CCO",
            value=5.0,
            upper_bound=11.0,
            censoring_type=CensoringType.INTERVAL,
            fidelity=QueryType.PRIMARY_SCREEN,
            cost=1.0,
            iteration=0,
        )

        with pytest.raises(ValueError, match="DOSE_RESPONSE"):
            model.refit([ps_record], aux_encoder=aux_encoder, max_epochs=1)
