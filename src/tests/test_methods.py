from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from agentad.methods import (  # noqa: E402
    AERCA,
    AERCAConfig,
    CARLA,
    CARLAConfig,
    CrossAD,
    CrossADConfig,
    DADA,
    DADAConfig,
    KANAD,
    KANADConfig,
    Left,
    LeftConfig,
    PaAno,
    PaAnoConfig,
    ScatterAD,
    ScatterADConfig,
    inject_anomalies,
)
from agentad.methods._utils import overlap_average  # noqa: E402


def assert_finite_shape(tensor, shape):
    assert tensor.shape == shape
    assert torch.isfinite(tensor).all()


def test_patch_scores_are_averaged_over_covered_points():
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(
        overlap_average(scores, patch_length=2, output_length=4),
        torch.tensor([[1.0, 1.5, 2.5, 3.0]]),
    )
    assert torch.equal(
        overlap_average(
            torch.tensor([[1.0, 2.0]]),
            patch_length=2,
            output_length=4,
            stride=2,
        ),
        torch.tensor([[1.0, 1.0, 2.0, 2.0]]),
    )


def test_aerca_losses_scores_causality_and_root_causes():
    model = AERCA(
        AERCAConfig(
            input_features=3,
            window=2,
            hidden_dim=8,
            hidden_layers=2,
        )
    )
    series = torch.randn(2, 12, 3)
    output = model(series)
    assert_finite_shape(output.reconstruction, (2, 8, 3))
    assert_finite_shape(output.encoder_coefficients, (2, 10, 2, 3, 3))
    losses = model.loss(series)
    assert losses.total.ndim == 0 and torch.isfinite(losses.total)
    losses.total.backward()
    assert_finite_shape(model.score(series), (2, 12))
    assert_finite_shape(model.causal_graph(series), (2, 3, 3))
    model.calibrate_root_causes(series)
    assert_finite_shape(model.root_cause_score(series), (2, 12, 3))


def test_carla_two_stage_losses_neighbor_mining_and_scores():
    config = CARLAConfig(
        input_features=2,
        window_length=8,
        mid_channels=4,
        projection_dim=6,
        clusters=3,
        cluster_heads=2,
    )
    model = CARLA(config)
    anchors = torch.randn(8, 8, 2)
    positives = anchors.roll(1, dims=0)
    injected = inject_anomalies(
        anchors,
        generator=torch.Generator().manual_seed(11),
    )
    assert_finite_shape(injected, anchors.shape)
    assert not torch.equal(injected, anchors)
    pretext = model.pretext_loss(anchors, positives, injected)
    classification = model.classification_loss(
        anchors,
        anchors.roll(1, dims=0),
        anchors.roll(4, dims=0),
    )
    total = pretext + classification.total
    assert total.ndim == 0 and torch.isfinite(total)
    total.backward()

    model.eval()
    nearest, furthest = model.mine_neighbors(
        anchors,
        k=2,
        query_chunk_size=3,
        bank_chunk_size=4,
    )
    assert nearest.shape == furthest.shape == (8, 2)
    rows = torch.arange(8)[:, None]
    assert not (nearest == rows).any()
    assert not (furthest == rows).any()
    model.calibrate_normal_clusters(anchors)
    assert_finite_shape(model.score(anchors), (8,))
    assert_finite_shape(model.score_series(torch.randn(2, 12, 2)), (2, 12))


def test_left_losses_prototype_update_and_score():
    config = LeftConfig(
        input_features=2,
        sequence_length=32,
        n_fft=16,
        hop_length=4,
        window_length=16,
        scale_factors=(4, 2, 1),
        patch_length=4,
        model_dim=16,
        encoder_layers=1,
        heads=4,
        feedforward_dim=32,
        fusion_layers=1,
        dropout=0,
        prototypes=4,
        confidence_threshold=0,
        memory_warmup_steps=0,
        memory_ramp_steps=1,
        score_smoothing=3,
    )
    model = Left(config)
    series = torch.randn(2, 32, 2)
    output = model(series)
    losses = model.loss_from_output(output)
    assert losses.total.ndim == 0 and torch.isfinite(losses.total)
    losses.total.backward()
    prototypes_before = model.time_prototypes.prototypes.detach().clone()
    model.update_prototypes(series, output)
    assert not torch.equal(prototypes_before, model.time_prototypes.prototypes)

    model.eval()
    assert_finite_shape(model.score(series), (2, 32))


def test_crossad_loss_and_score_shapes():
    config = CrossADConfig(
        sequence_length=24,
        patch_length=4,
        scale_kernels=(4, 2),
        top_frequencies=3,
        query_count=3,
        query_length=2,
        context_size=4,
        encoder_layers=1,
        decoder_layers=1,
        extractor_layers=1,
        heads=2,
        model_dim=8,
        attention_dropout=0,
        projection_dropout=0,
        feedforward_dropout=0,
    )
    model = CrossAD(config)
    series = torch.randn(2, 24, 3)

    model.train()
    loss = model.loss(series)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()

    model.eval()
    assert_finite_shape(model.score(series), (2, 24))


def test_dada_masked_reconstructions_score_and_checkpoint_conversion(tmp_path):
    config = DADAConfig(
        sequence_length=12,
        hidden_dim=8,
        representation_dim=8,
        bottleneck_dims=(2, 4),
        experts_per_input=1,
        patch_length=3,
        encoder_depth=1,
        copies=4,
        representation_dropout=0,
        bottleneck_dropout=0,
    )
    model = DADA(config).eval()
    series = torch.randn(2, 12, 2)
    generator = torch.Generator().manual_seed(7)
    output = model(series, generator=generator)
    assert_finite_shape(output.reconstructions, (4, 2, 12, 2))
    assert output.balance_loss.ndim == 0
    assert_finite_shape(
        model.score(series, generator=torch.Generator().manual_seed(7)),
        (2, 12),
    )
    checkpoint = tmp_path / "dada.bin"
    torch.save(
        {f"model.{key}": value for key, value in model.state_dict().items()},
        checkpoint,
    )
    restored = DADA(config)
    restored.load_reference_checkpoint(checkpoint)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_kanad_forecast_pairs_loss_and_score():
    model = KANAD(KANADConfig(window=8, order=2))
    series = torch.randn(3, 12, 2)
    windows, targets = model.pairs(series)
    assert windows.shape == (12, 8, 2)
    assert targets.shape == (12, 2)
    loss = model.loss(windows, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert_finite_shape(model.score(windows, targets), (12,))


def test_paano_training_memory_and_point_scores():
    config = PaAnoConfig(
        input_features=2,
        patch_length=8,
        convolution_widths=(8, 6),
        kernel_sizes=(3, 3),
        projection_dim=6,
        positive_radius=1,
        random_negatives=2,
        top_k=2,
    )
    model = PaAno(config)
    patches = torch.randn(16, 2, 8)
    losses = model.training_loss(
        patches,
        torch.arange(4, 12),
        iteration=0,
        total_iterations=20,
        generator=torch.Generator().manual_seed(3),
    )
    assert losses.total.ndim == 0 and torch.isfinite(losses.total)
    losses.total.backward()

    model.eval()
    model.build_memory_bank(patches)
    assert_finite_shape(model.score(torch.randn(2, 12, 2)), (2, 12))


def test_scatterad_losses_target_update_and_score():
    config = ScatterADConfig(
        input_features=3,
        hidden_dim=8,
        graph_layers=1,
        heads=2,
        temporal_dropout=0,
        attention_dropout=0,
        projection_dropout=0,
    )
    model = ScatterAD(config)
    series = torch.randn(2, 10, 3)
    losses = model.loss(series)
    assert losses.total.ndim == 0 and torch.isfinite(losses.total)
    losses.total.backward()
    model.update_target()

    model.eval()
    assert_finite_shape(model.score(series), (2, 10))
