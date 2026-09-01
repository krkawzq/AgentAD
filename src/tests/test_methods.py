from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from agentad.methods import (  # noqa: E402
    AERCA,
    AERCAConfig,
    AxonAD,
    AxonADConfig,
    BASELINES,
    CARLA,
    CARLAConfig,
    CARLAPretextLightningModule,
    ClusterLocalOutlierFactorConfig,
    ConnectivityOutlierFactorConfig,
    CopulaConfig,
    CrossAD,
    CrossADConfig,
    DADA,
    DADAConfig,
    ExtendedIsolationForestConfig,
    FourierConfig,
    HistogramConfig,
    IsolationForestConfig,
    KANAD,
    KANADConfig,
    KMeansDistanceConfig,
    Left,
    LeftConfig,
    LocalOutlierFactorConfig,
    LocalPolynomialConfig,
    MMPADConfig,
    MatrixProfileConfig,
    MinimumCovarianceDeterminantConfig,
    NearestNeighborsConfig,
    OneClassSVMConfig,
    PaAno,
    PaAnoConfig,
    PrincipalComponentConfig,
    RobustPCAConfig,
    ScatterAD,
    ScatterADConfig,
    SpectralResidualConfig,
    StatisticalFeaturesConfig,
    StreamingMatrixProfileConfig,
    TimeRCD,
    TimeRCDConfig,
    TSPulseConfig,
    TSPulseFineTune,
    TSPulseZeroShot,
    XLSTMAD,
    XLSTMADConfig,
    build_reference,
    cluster_local_outlier_factor_score,
    connectivity_outlier_factor_score,
    copula_score,
    extended_isolation_forest_score,
    fourier_score,
    histogram_score,
    inject_anomalies,
    isolation_forest_score,
    kmeans_distance_score,
    local_outlier_factor_score,
    local_polynomial_score,
    matrix_profile_score,
    minimum_covariance_determinant_score,
    mmpad_score,
    nearest_neighbors_score,
    one_class_svm_score,
    principal_component_score,
    robust_pca_score,
    spectral_residual_score,
    statistical_features_score,
    streaming_matrix_profile_score,
)
from agentad.methods.MMPAD.algorithm import _resolved_length  # noqa: E402
from agentad.methods.TimeRCD import load_official_checkpoint  # noqa: E402
from agentad.methods.TimeRCD.model import _bounded_inference_window  # noqa: E402
from agentad.methods._utils import (  # noqa: E402
    evaluation_mode,
    overlap_average,
    topk_cosine_distance,
)
from agentad.methods.baseline._common import zscore  # noqa: E402


def assert_finite_shape(tensor, shape):
    assert tensor.shape == shape
    assert torch.isfinite(tensor).all()


def test_original_config_factories_cover_published_variants():
    assert set(AERCAConfig.original_configs()) == {
        "linear",
        "lorenz96",
        "lotka_volterra",
        "msds",
        "nonlinear",
        "swat",
    }
    assert set(CARLAConfig.original_configs()) == {
        "kpi",
        "msl",
        "smap",
        "smd",
        "swat",
        "yahoo",
    }
    assert len(CrossADConfig.original_configs()) == 15
    assert len(LeftConfig.original_configs()) == 7
    assert len(MMPADConfig.original_configs(subsequence_length=16)) == 5
    assert set(DADAConfig.original_configs()) == {"pretrained", "cli"}
    assert set(KANADConfig.original_configs()) == {"default"}
    assert set(ScatterADConfig.original_configs()) == {
        "MSL",
        "SWaT",
        "PSM",
        "WADI",
        "NIPS_TS_Water",
        "NIPS_TS_Swan",
    }
    assert set(TimeRCDConfig.original_configs(input_features=3)) == {
        "pretraining",
        "pretrained_univariate",
        "pretrained_multivariate",
    }
    assert set(TSPulseConfig.original_configs(input_features=2)) == {
        "default",
        "tsb_zero_shot",
        "tsb_finetune",
    }
    assert set(XLSTMADConfig.original_configs(input_features=2)) == {
        "default",
        "example",
    }
    assert set(BASELINES) == {
        "ClusterLocalOutlierFactor",
        "ConnectivityOutlierFactor",
        "Copula",
        "ExtendedIsolationForest",
        "Fourier",
        "Histogram",
        "IsolationForest",
        "KMeansDistance",
        "LocalOutlierFactor",
        "LocalPolynomial",
        "MatrixProfile",
        "MinimumCovarianceDeterminant",
        "NearestNeighbors",
        "OneClassSVM",
        "PrincipalComponent",
        "RobustPCA",
        "SpectralResidual",
        "StatisticalFeatures",
        "StreamingMatrixProfile",
    }


def test_evaluation_mode_restores_mixed_submodule_states():
    module = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.BatchNorm1d(2))
    module.train()
    module[0].eval()
    with evaluation_mode(module):
        assert not any(child.training for child in module.modules())
    assert module.training
    assert not module[0].training
    assert module[1].training


def test_time_rcd_multivariate_window_respects_attention_token_cap():
    config = replace(
        TimeRCDConfig.original_pretrained_multivariate(input_features=256),
        inference_window=15_000,
        max_attention_tokens=4096,
    )
    assert _bounded_inference_window(config, 100_000, 256) == 256
    assert _bounded_inference_window(config, 100_000, 2) == 15_000
    assert _bounded_inference_window(config, 1_000, 1) == 1_000


def test_baseline_zscore_maps_numerically_constant_slices_to_zero():
    values = torch.tensor([[1e-38, 2e-38, 3e-38]], dtype=torch.float32)
    normalized = zscore(values, dim=1, ddof=1)
    assert torch.equal(normalized, torch.zeros_like(normalized))


def test_chunked_topk_cosine_matches_dense_reference():
    queries = torch.randn(7, 5)
    bank = torch.randn(11, 5)
    expected = 1 - (
        torch.nn.functional.normalize(queries, dim=1)
        @ torch.nn.functional.normalize(bank, dim=1).T
    ).topk(3, dim=1).values.mean(1)
    actual = topk_cosine_distance(
        queries,
        bank,
        k=3,
        query_chunk_size=2,
        bank_chunk_size=4,
    )
    assert torch.allclose(actual, expected)
    assert topk_cosine_distance(queries[:0], bank, k=1).shape == (0,)


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
    with pytest.raises(ValueError, match="uncovered"):
        overlap_average(
            torch.tensor([[1.0, 2.0]]),
            patch_length=2,
            output_length=5,
            stride=2,
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
    losses = model.compute_loss(series)
    assert losses.total.ndim == 0 and torch.isfinite(losses.total)
    losses.total.backward()
    assert_finite_shape(model.score(series), (2, 12))
    assert_finite_shape(model.causal_graph(series), (2, 3, 3))
    model.calibrate_root_causes(series)
    assert_finite_shape(model.root_cause_score(series), (2, 12, 3))
    # Smoothness regularizes across adjacent lags, so order-one models have
    # no smoothness term, exactly like the original implementation.
    single = AERCA(
        AERCAConfig(input_features=3, window=1, hidden_dim=8, hidden_layers=2)
    )
    single_losses = single.compute_loss(torch.randn(2, 12, 3))
    assert float(single_losses.encoder_smoothness) == 0.0
    assert float(single_losses.decoder_smoothness) == 0.0
    assert torch.isfinite(single_losses.total)


def test_carla_two_stage_losses_neighbor_mining_and_scores():
    # The exact-index assertions below need a seeded projection space:
    # unseeded inits occasionally collapse every pairwise similarity onto
    # the same value, where chunked and dense topk break ties differently.
    torch.manual_seed(20)
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
    pretext = model.compute_pretext_loss(anchors, positives, injected)
    classification = model.compute_classification_loss(
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
    # Mining happens in the pretext projection space.
    features = model.project(anchors)
    similarity = features @ features.T
    similarity.fill_diagonal_(-torch.inf)
    assert torch.equal(nearest, similarity.topk(2, dim=1).indices)
    similarity.fill_diagonal_(torch.inf)
    assert torch.equal(furthest, similarity.topk(2, dim=1, largest=False).indices)
    model.calibrate_normal_clusters(anchors)
    assert_finite_shape(model.window_score(anchors), (8,))
    assert_finite_shape(model.score(torch.randn(2, 12, 2)), (2, 12))


def test_carla_pretext_neighbor_positives():
    config = CARLAConfig(
        input_features=2,
        window_length=8,
        mid_channels=4,
        projection_dim=6,
        clusters=3,
        cluster_heads=2,
    )
    module = CARLAPretextLightningModule(config)
    pool = torch.randn(20, 8, 2)
    anchors = pool[[15, 5, 19]]
    batch = {
        "anchors": anchors,
        "window_pool": pool,
        "anchor_indices": torch.tensor([15, 5, 19]),
    }
    positives = module._positives(batch, anchors)
    assert positives.shape == anchors.shape
    # Windows past the first ten draw one of their ten predecessors.
    matches = (pool == positives[0].unsqueeze(0)).all(-1).all(-1)
    assert matches[5:15].any() and not matches[:5].any()
    # The first ten windows fall back to noise around the anchor.
    assert not torch.equal(positives[1], anchors[1])
    # Batches without a pool keep the additive-noise fallback.
    noisy = module._positives({"anchors": anchors}, anchors)
    assert noisy.shape == anchors.shape and not torch.equal(noisy, anchors)


def test_mmpad_infers_subsequence_length_from_autocorrelation():
    config = MMPADConfig()
    t = torch.arange(1500, dtype=torch.float64)
    # A decaying periodic signal makes the first ACF peak the strongest one.
    signal = torch.sin(2 * torch.pi * t / 128) * torch.exp(-t / 2000)
    series = signal.unsqueeze(0).unsqueeze(-1).float()
    assert _resolved_length(config, series) == 128
    # Series shorter than the original's 401-point ACF window take the
    # fallback period, clamped to the series length.
    assert _resolved_length(config, torch.randn(1, 300, 1)) == 125


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
        update_error_quantile=1,
        update_disagreement_quantile=1,
        memory_warmup_steps=0,
        memory_ramp_steps=1,
        score_smoothing=3,
    )
    model = Left(config)
    series = torch.randn(2, 32, 2)
    output = model(series)
    # Real training order: update prototypes between forward and backward so
    # the test locks the autograd-safety of the in-place .data assignment.
    prototypes_before = model.time_prototypes.prototypes.detach().clone()
    model.update_prototypes(series, output)
    assert not torch.equal(prototypes_before, model.time_prototypes.prototypes)
    losses = model.compute_loss_from_output(series, output)
    assert losses.total.ndim == 0 and torch.isfinite(losses.total)
    losses.total.backward()

    model.eval()
    assert_finite_shape(model.score(series), (2, 32))


def test_mmpad_self_join_and_fitted_reference_scores():
    config = MMPADConfig(
        subsequence_length=4,
        dimensions=1,
        neighbors=1,
        query_chunk_size=3,
    )
    series = torch.randn(2, 16, 2)
    assert_finite_shape(mmpad_score(series, config), (2, 16))
    reference = build_reference(torch.randn(1, 12, 2), config)
    assert_finite_shape(mmpad_score(series, config, reference), (2, 16))


def test_tspulse_finetune_loss_and_zero_shot_score():
    config = TSPulseConfig(
        input_features=2,
        context_length=16,
        patch_length=4,
        model_dim=8,
        decoder_model_dim=8,
        layers=1,
        decoder_layers=1,
        expansion_factor=2,
        register_tokens=2,
        dropout=0,
        aggregation_length=8,
        smoothing_window=3,
        mask_type="user",
        forecast_length=1,
    )
    finetuned = TSPulseFineTune(config)
    windows = torch.randn(2, 16, 2)
    losses = finetuned.compute_loss(
        windows,
        future_values=torch.randn(2, 1, 2),
        generator=torch.Generator().manual_seed(5),
    )
    assert losses.total.ndim == 0 and torch.isfinite(losses.total)
    losses.total.backward()
    with pytest.raises(ValueError, match="hide at least one"):
        finetuned.compute_loss(windows, observed_mask=torch.ones_like(windows))

    with pytest.raises(ValueError, match="from_pretrained"):
        TSPulseZeroShot(config)
    assert_finite_shape(finetuned.score(torch.randn(1, 22, 2), batch_size=3), (1, 22))


def test_time_rcd_loss_score_and_checkpoint_conversion(tmp_path):
    config = TimeRCDConfig(
        input_features=2,
        model_dim=16,
        projection_dim=8,
        patch_length=4,
        layers=1,
        heads=4,
        dropout=0,
        inference_window=8,
    )
    model = TimeRCD(config)
    series = torch.randn(2, 12, 2)
    losses = model.compute_loss(
        series,
        labels=torch.zeros(2, 12),
        masked_points=torch.arange(12)[None].expand(2, -1) < 4,
        generator=torch.Generator().manual_seed(2),
    )
    assert losses.total.ndim == 0 and torch.isfinite(losses.total)
    losses.total.backward()
    with pytest.raises(ValueError, match="at least one time point"):
        model.compute_loss(series, masked_points=torch.zeros(2, 12, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one valid"):
        model(series, torch.zeros(2, 12, dtype=torch.bool))
    model.eval()
    assert_finite_shape(model.score(series, batch_size=2), (2, 12))
    checkpoint = tmp_path / "time_rcd.pth"
    torch.save(
        {
            "model_state_dict": {
                f"module.{key}": value for key, value in model.state_dict().items()
            }
        },
        checkpoint,
    )
    restored = TimeRCD(config)
    restored.load_checkpoint(checkpoint)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_xlstmad_reconstruction_and_aligned_score():
    model = XLSTMAD(
        XLSTMADConfig(
            input_features=2,
            window_length=8,
            embedding_dim=20,
            blocks=2,
            scalar_memory_blocks=(0,),
            heads=4,
            scalar_kernel=3,
            matrix_kernel=3,
            scalar_backend="vanilla",
        )
    )
    windows = torch.randn(3, 8, 2)
    losses = model.compute_loss(windows)
    assert losses.total.ndim == 0 and torch.isfinite(losses.total)
    losses.total.backward()
    model.eval()
    assert_finite_shape(model.score(torch.randn(2, 12, 2), batch_size=3), (2, 12))


def test_axonad_loss_target_update_calibration_and_score():
    model = AxonAD(
        AxonADConfig(
            input_features=2,
            window_length=8,
            model_dim=8,
            heads=2,
            tail_length=2,
            query_dilations=(1, 2),
            dropout=0,
        )
    )
    windows = torch.randn(3, 8, 2)
    losses = model.compute_loss(windows)
    assert losses.total.ndim == 0 and torch.isfinite(losses.total)
    losses.total.backward()
    model.update_target()
    series = torch.randn(2, 12, 2)
    model.calibrate(series, batch_size=3)
    assert_finite_shape(model.score(series, batch_size=3), (2, 12))


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
    loss = model.compute_loss(series)
    assert loss.total.ndim == 0 and torch.isfinite(loss.total)
    loss.total.backward()

    context_before = model.context_memory.context.clone()
    assert_finite_shape(model.score(series), (2, 24))
    assert model.training
    assert torch.equal(context_before, model.context_memory.context)


def test_crossad_batch_norm_handles_single_token_training_case():
    model = CrossAD(
        CrossADConfig(
            sequence_length=2,
            patch_length=2,
            scale_kernels=(2,),
            top_frequencies=1,
            query_count=1,
            query_length=1,
            context_size=2,
            encoder_layers=1,
            decoder_layers=1,
            extractor_layers=1,
            heads=1,
            model_dim=4,
            normalization="batch",
            attention_dropout=0,
            projection_dropout=0,
            feedforward_dropout=0,
        )
    )
    losses = model.compute_loss(torch.randn(1, 2, 1))
    assert torch.isfinite(losses.total)
    losses.total.backward()


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
    model.train()
    training_output = model(series, generator=torch.Generator().manual_seed(9))
    (training_output.reconstructions.mean() + training_output.balance_loss).backward()
    assert model.adaptive_bottleneck.w_gate.grad is not None
    assert torch.isfinite(model.adaptive_bottleneck.w_gate.grad).all()


def test_kanad_forecast_pairs_loss_and_score():
    model = KANAD(KANADConfig(window=8, order=2))
    series = torch.randn(3, 12, 2)
    windows, targets = model.pairs(series)
    assert windows.shape == (12, 8, 2)
    assert targets.shape == (12, 2)
    loss = model.compute_loss(windows, targets)
    assert loss.total.ndim == 0 and torch.isfinite(loss.total)
    assert_finite_shape(model.window_score(windows, targets), (12,))
    assert_finite_shape(model.score(series), (3, 12))
    assert model.training


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
    losses = model.compute_loss(
        patches,
        torch.arange(4, 12),
        iteration=0,
        total_iterations=20,
        generator=torch.Generator().manual_seed(3),
    )
    assert losses.total.ndim == 0 and torch.isfinite(losses.total)
    losses.total.backward()

    model.eval()
    memory = model.build_memory_bank(patches)
    # The coreset keeps one representative per MiniBatchKMeans cluster; with
    # 16 patches that is min(500, 16 - 1) = 15 clusters.
    assert memory.shape[0] == 15
    assert_finite_shape(model.score(torch.randn(2, 12, 2)), (2, 12))


def test_mmpad_flat_channels_are_excluded_like_the_original():
    generator = torch.Generator().manual_seed(4)
    series = torch.randn(1, 16, 2, generator=generator)
    flat = series.clone()
    flat[..., 1] = 5.0
    config = MMPADConfig(subsequence_length=4, dimensions=1, neighbors=1)
    # A flat channel is invalid and must not contribute a zero correlation;
    # the score equals the single-varying-channel score.
    assert torch.allclose(
        mmpad_score(flat, config), mmpad_score(series[..., :1], config)
    )


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
    losses = model.compute_loss(series)
    assert losses.total.ndim == 0 and torch.isfinite(losses.total)
    losses.total.backward()
    model.update_target()

    model.eval()
    assert_finite_shape(model.score(series), (2, 10))


PRETRAINED_DIR = Path(__file__).resolve().parents[2] / "pretrained"


@pytest.mark.skipif(
    not (PRETRAINED_DIR / "dada" / "pytorch_model.bin").exists(),
    reason="official checkpoints not downloaded (scripts/download/download_pretrained.py)",
)
def test_dada_and_time_rcd_official_checkpoints_load_and_score():
    dada = DADA.from_official_checkpoint(PRETRAINED_DIR / "dada" / "pytorch_model.bin")
    generator = torch.Generator().manual_seed(7)
    assert_finite_shape(
        dada.score(torch.randn(1, 100, 3), generator=generator), (1, 100)
    )

    uni = load_official_checkpoint(
        PRETRAINED_DIR / "time_rcd" / "uni" / "pretrain_checkpoint_best_uni.pth"
    )
    assert_finite_shape(uni.score(torch.randn(1, 320, 1)), (1, 320))

    multi = load_official_checkpoint(
        PRETRAINED_DIR / "time_rcd" / "multi" / "pretrain_checkpoint_best_multi.pth",
        input_features=8,
    )
    assert_finite_shape(multi.score(torch.randn(1, 320, 8)), (1, 320))


@pytest.mark.skipif(
    not (PRETRAINED_DIR / "tspulse" / "model.safetensors").exists(),
    reason="official checkpoints not downloaded (scripts/download/download_pretrained.py)",
)
def test_tspulse_zero_shot_official_snapshot_scores():
    pytest.importorskip("tsfm_public")
    config = TSPulseConfig(input_features=1, context_length=512, patch_length=8)
    zero_shot = TSPulseZeroShot.from_pretrained(
        config, model_name_or_path=str(PRETRAINED_DIR / "tspulse")
    )
    assert_finite_shape(zero_shot.score(torch.randn(1, 600, 1)), (1, 600))


def test_spectral_residual_flags_spike_and_zeroes_constant_series():
    torch.manual_seed(0)
    t = torch.arange(256, dtype=torch.float32)
    values = torch.sin(t * 0.15) + 0.05 * torch.sin(t * 1.3) + 0.02 * torch.randn(256)
    values[120:124] += 3.0
    scores = spectral_residual_score(values[None, :, None], SpectralResidualConfig())
    assert_finite_shape(scores, (1, 256))
    assert scores[0, 120:124].max() > scores[0, 20:100].max()
    constant = torch.full((1, 64, 1), 3.0)
    assert torch.equal(
        spectral_residual_score(constant, SpectralResidualConfig()),
        torch.zeros(1, 64),
    )


def test_fourier_flags_injected_region_with_zscored_magnitude():
    t = torch.arange(300, dtype=torch.float32)
    series = torch.sin(t * 0.2)[None, :, None].clone()
    series[:, 150:160, 0] += 4.0
    scores = fourier_score(series, FourierConfig())
    assert_finite_shape(scores, (1, 300))
    # Region scores are mean |z| of selected candidates, so any formed
    # region exceeds the detection threshold.
    assert scores[:, 150:160].max() > FourierConfig().outlier_threshold
    assert scores[:, 150:160].max() > scores[:, 20:100].max()


def test_matrix_profile_matches_bruteforce_reference():
    torch.manual_seed(3)
    series = torch.randn(1, 120, 1)
    window, zone = 12, 3
    scores = matrix_profile_score(series, MatrixProfileConfig(window=window))

    values = series[0, :, 0]
    count = values.numel() - window + 1
    raw = values.unfold(0, window, 1)
    normalized = (raw - raw.mean(1, keepdim=True)) / raw.std(
        1, keepdim=True, correction=0
    ).clamp_min(1e-12)
    profile = []
    for i in range(count):
        best = torch.inf
        for j in range(count):
            if abs(i - j) <= zone:
                continue
            correlation = float(normalized[i] @ normalized[j]) / window
            best = min(best, (2 * window * (1 - correlation)) ** 0.5)
        profile.append(best)
    profile = torch.tensor(profile)
    expected = torch.full((values.numel(),), float(profile.min()))
    expected[window // 2 : window // 2 + count] = profile
    assert torch.allclose(scores[0], expected, atol=1e-4)


def test_streaming_matrix_profile_scores_only_against_the_past():
    torch.manual_seed(4)
    pattern = torch.randn(20)
    values = torch.cat(
        (
            torch.randn(60),
            pattern,
            torch.randn(40),
            pattern,
            torch.randn(30),
        )
    )
    series = values[None, :, None]
    config = StreamingMatrixProfileConfig(warmup=60, window=20)
    scores = streaming_matrix_profile_score(series, config)
    assert_finite_shape(scores, (1, 170))
    # Window starts below the warmup are zeroed and centered padding maps
    # point t to window start t - window//2, so points up to 69 stay zero.
    assert scores[:, :70].abs().max() == 0
    # The repeated pattern finds its earlier twin as a left neighbor...
    assert scores[0, 130] < 0.5
    # ...while the novel tail has no close past match.
    assert scores[0, 150] > scores[0, 130]


def test_local_polynomial_scores_context_residual_and_zeroes_initial_segment():
    t = torch.arange(400, dtype=torch.float32)
    series = torch.sin(t * 0.1)[None, :, None].clone()
    series[:, 200:206, 0] += 3.0
    config = LocalPolynomialConfig(degree=2, window=63)
    scores = local_polynomial_score(series, config)
    assert_finite_shape(scores, (1, 400))
    # initial = min(500, 40) -> first scored window starts at 63.
    assert scores[:, :63].abs().max() == 0
    assert scores[:, 200:206].max() > scores[:, 100:150].max()


def test_robust_pca_recovers_sparse_corruption_of_low_rank_series():
    torch.manual_seed(5)
    low_rank = torch.randn(200, 1) @ torch.randn(1, 3)
    matrix = low_rank + 0.01 * torch.randn(200, 3)
    matrix[150:154] += 5.0
    scores = robust_pca_score(matrix[None], RobustPCAConfig())
    assert_finite_shape(scores, (1, 200))
    assert scores[0, 150:154].min() > scores[0, :150].max()


def test_statistical_features_flags_anomaly_and_zeroes_constant_series():
    torch.manual_seed(6)
    t = torch.arange(600, dtype=torch.float32)
    values = torch.sin(t * 0.05) + 0.05 * torch.randn(600)
    values[300:310] += 2.0
    scores = statistical_features_score(
        values[None, :, None], StatisticalFeaturesConfig()
    )
    assert_finite_shape(scores, (1, 600))
    assert scores[0, 300:310].max() > scores[0, 100:200].max()
    constant = torch.full((1, 100, 1), 2.0)
    assert torch.equal(
        statistical_features_score(constant, StatisticalFeaturesConfig()),
        torch.zeros(1, 100),
    )


def test_nearest_neighbors_matches_dense_reference_and_flags_point_outlier():
    torch.manual_seed(7)
    series = torch.randn(1, 60, 1)
    config = NearestNeighborsConfig(window=1, neighbors=4, method="mean")
    scores = nearest_neighbors_score(series, config)

    values = series[0, :, 0]
    standardized = (values - values.mean()) / values.std(correction=0)
    distance = torch.cdist(standardized[:, None], standardized[:, None])
    distance.fill_diagonal_(torch.inf)
    nearest = distance.sort(dim=1).values[:, :4]
    assert torch.allclose(scores[0], nearest.mean(dim=1), atol=1e-5)

    corrupted = series.clone()
    corrupted[0, 30, 0] += 10.0
    outlier_scores = nearest_neighbors_score(
        corrupted, NearestNeighborsConfig(window=1, neighbors=4)
    )
    assert outlier_scores[0, 30] == outlier_scores.max()


def test_local_outlier_factor_matches_manual_reference():
    torch.manual_seed(8)
    series = torch.randn(1, 40, 1)
    config = LocalOutlierFactorConfig(window=1, neighbors=3, normalize=False)
    scores = local_outlier_factor_score(series, config)

    values = series[0, :, 0]
    count = values.numel()
    distance = torch.cdist(values[:, None], values[:, None])
    distance.fill_diagonal_(torch.inf)
    neighbors = distance.argsort(dim=1)[:, :3]
    k_distance = distance.gather(1, neighbors)[:, -1]
    lrd = []
    for i in range(count):
        reach = (
            sum(
                max(float(distance[i, j]), float(k_distance[j]))
                for j in neighbors[i].tolist()
            )
            / 3
        )
        lrd.append(1.0 / (reach + 1e-10))
    lrd = torch.tensor(lrd)
    expected = torch.tensor([lrd[neighbors[i]].mean() / lrd[i] for i in range(count)])
    assert torch.allclose(scores[0], expected, atol=1e-4)


def test_connectivity_outlier_factor_matches_manual_reference():
    torch.manual_seed(9)
    series = torch.randn(1, 30, 2)
    config = ConnectivityOutlierFactorConfig(neighbors=4)
    scores = connectivity_outlier_factor_score(series, config)

    values = series[0]
    count, k = values.shape[0], 4
    distance = torch.cdist(values, values)
    path = distance.argsort(dim=1)
    chaining = []
    for i in range(count):
        costs = []
        for j in range(k):
            costs.append(
                min(float(distance[path[i, j + 1], path[i, m]]) for m in range(j + 1))
            )
        weight = sum(
            2 * (k + 1 - (h + 1)) / ((k + 1) * k) * cost for h, cost in enumerate(costs)
        )
        chaining.append(weight)
    expected = torch.tensor(
        [
            chaining[i] * k / sum(chaining[j] for j in path[i, 1 : k + 1])
            for i in range(count)
        ]
    )
    assert torch.allclose(scores[0], expected, atol=1e-4)


def test_copula_matches_empirical_copula_reference():
    torch.manual_seed(10)
    series = torch.randn(1, 50, 2) * torch.tensor([1.0, 3.0]) + torch.tensor(
        [0.5, -1.0]
    )
    scores = copula_score(series, CopulaConfig(normalize=False))

    values = series[0].to(torch.float64)
    count = values.shape[0]

    def ecdf(column):
        order = column.argsort()
        probabilities = torch.arange(1, count + 1, dtype=torch.float64) / count
        sorted_column = column[order]
        for i in range(count - 2, -1, -1):
            if sorted_column[i] == sorted_column[i + 1]:
                probabilities[i] = probabilities[i + 1]
        output = torch.empty_like(probabilities)
        output[order] = probabilities
        return output

    columns = [ecdf(values[:, c]) for c in range(values.shape[1])]
    left = -torch.log(torch.stack(columns, dim=1))
    right = -torch.log(
        torch.stack([ecdf(-values[:, c]) for c in range(values.shape[1])], dim=1)
    )
    centered = values - values.mean(0)
    moment2 = centered.square().mean(0)
    moment3 = centered.pow(3).mean(0)
    skewness = torch.where(
        moment2 > 0, moment3 / moment2.pow(1.5), torch.zeros_like(moment3)
    ).sign()
    u_skew = -left * (skewness - 1).sign() + right * (skewness + 1).sign()
    expected = torch.maximum(u_skew, (left + right) / 2).sum(dim=1)
    assert torch.allclose(scores[0], expected.to(scores.dtype), atol=1e-4)


def test_histogram_matches_manual_density_reference():
    torch.manual_seed(11)
    series = torch.randn(1, 80, 1)
    config = HistogramConfig(window=1, bins=5)
    scores = histogram_score(series, config)

    values = series[0, :, 0]
    standardized = (values - values.mean()) / values.std(correction=0)
    count = standardized.numel()
    bins, alpha = config.bins, config.alpha
    edges = torch.linspace(standardized.min(), standardized.max(), bins + 1)
    widths = edges.diff()

    # Values on the lower edge fall into bin 0, on the upper edge into the
    # last bin, interior values into the bin below the edge they hit.
    assignment = []
    for value in standardized:
        index = int(torch.searchsorted(edges, value, right=False))
        assignment.append(min(max(index - 1, 0), bins - 1))
    counts = torch.zeros(bins)
    for slot in assignment:
        counts[slot] += 1
    density = counts / (count * widths)
    bin_scores = torch.log2(density + alpha)
    expected = torch.tensor([-bin_scores[slot] for slot in assignment])
    assert torch.allclose(scores[0], expected, atol=1e-5)


def test_cluster_local_outlier_factor_flags_cluster_outliers_and_zeroes_constant_series():
    torch.manual_seed(12)
    t = torch.arange(240, dtype=torch.float32)
    values = torch.sin(t * 0.05) + 0.05 * torch.randn(240)
    values[200:210] += 8.0
    series = values[None, :, None]
    scores = cluster_local_outlier_factor_score(
        series, ClusterLocalOutlierFactorConfig()
    )
    assert_finite_shape(scores, (1, 240))
    # The distant block forms a small cluster scored by its distance to the
    # nearest large-cluster center.
    assert scores[0, 200:210].min() > scores[0, :100].max()
    constant = torch.full((1, 100, 1), 2.0)
    assert torch.equal(
        cluster_local_outlier_factor_score(constant, ClusterLocalOutlierFactorConfig()),
        torch.zeros(1, 100),
    )


def test_isolation_forest_flags_windowed_outliers():
    torch.manual_seed(13)
    t = torch.arange(300, dtype=torch.float32)
    values = torch.sin(t * 0.08) + 0.05 * torch.randn(300)
    values[150:160] += 6.0
    series = values[None, :, None]
    config = IsolationForestConfig(window=8, trees=50, sample_size=128, seed=3)
    scores = isolation_forest_score(series, config)
    assert_finite_shape(scores, (1, 300))
    # Windows covering the spike isolate near the root and score close to one.
    assert scores[0, 150:160].max() > scores[0, 20:100].max()


def test_extended_isolation_forest_flags_point_outliers():
    torch.manual_seed(14)
    t = torch.arange(300, dtype=torch.float32)
    values = torch.sin(t * 0.08) + 0.05 * torch.randn(300)
    values[150:160] += 6.0
    series = values[None, :, None]
    config = ExtendedIsolationForestConfig(trees=50, sample_size=128, seed=3)
    scores = extended_isolation_forest_score(series, config)
    assert_finite_shape(scores, (1, 300))
    assert scores[0, 150:160].max() > scores[0, 20:100].max()


def test_kmeans_distance_flags_windowed_outliers_and_matches_mean_reference():
    torch.manual_seed(15)
    t = torch.arange(240, dtype=torch.float32)
    values = torch.sin(t * 0.01) + 0.001 * torch.randn(240)
    values[200:204] += 8.0
    series = values[None, :, None]
    config = KMeansDistanceConfig(window=8, clusters=3, seed=1)
    scores = kmeans_distance_score(series, config)
    assert_finite_shape(scores, (1, 240))
    # Windows covering the spike sit far from every ramp-shaped centroid.
    assert scores[0, 200:204].max() > scores[0, 20:100].max()
    # Window 1 keeps raw values and a single cluster converges to the mean,
    # so the score reduces to the deviation from the mean.
    point = values.clone()
    point[100] += 5.0
    reference = kmeans_distance_score(
        point[None, :, None], KMeansDistanceConfig(window=1, clusters=1, seed=1)
    )
    assert torch.allclose(reference[0], (point - point.mean()).abs(), atol=1e-5)


def test_minimum_covariance_determinant_flags_direction_outliers():
    torch.manual_seed(16)
    base = torch.randn(200, 1)
    matrix = torch.cat(
        (base, base + 0.05 * torch.randn(200, 1), base + 0.05 * torch.randn(200, 1)),
        dim=1,
    )
    matrix[150:154, 0] += 8.0
    scores = minimum_covariance_determinant_score(
        matrix[None], MinimumCovarianceDeterminantConfig()
    )
    assert_finite_shape(scores, (1, 200))
    # Rows spiking a single channel point away from the correlated cloud.
    assert scores[0, 150:154].min() > scores[0, :150].max()
    constant = torch.full((1, 100, 3), 2.0)
    assert torch.equal(
        minimum_covariance_determinant_score(
            constant, MinimumCovarianceDeterminantConfig()
        ),
        torch.zeros(1, 100),
    )


def test_one_class_svm_flags_boundary_violating_windows():
    torch.manual_seed(17)
    t = torch.arange(300, dtype=torch.float32)
    values = torch.sin(t * 0.08) + 0.05 * torch.randn(300)
    values[150:160] += 6.0
    series = values[None, :, None]
    config = OneClassSVMConfig(window=8, seed=1)
    scores = one_class_svm_score(series, config)
    assert_finite_shape(scores, (1, 300))
    # Windows beyond the fitted boundary take a positive margin.
    assert scores[0, 150:160].max() > 0
    assert scores[0, 150:160].max() > scores[0, 20:100].max()


def test_principal_component_matches_manual_reference():
    torch.manual_seed(18)
    series = torch.randn(1, 60, 2)
    config = PrincipalComponentConfig(window=1, normalize=False)
    scores = principal_component_score(series, config)

    values = series[0]
    mean = values.mean(dim=0)
    scale = values.std(dim=0, correction=0)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    standardized = (values - mean) / scale
    keep = standardized.abs().any(dim=0)
    standardized = standardized[:, keep]
    centered = standardized - standardized.mean(dim=0)
    _, _, vectors = torch.linalg.svd(centered, full_matrices=False)
    variance = centered.square().sum(dim=0) / (values.shape[0] - 1)
    shares = (variance / variance.sum()).clamp_min(1e-12)
    # Every fitted eigenvector contributes its distance weighted by the
    # inverse explained-variance share.
    expected = (torch.cdist(standardized, vectors) / shares).sum(dim=1)
    assert torch.allclose(scores[0], expected, atol=1e-5)
