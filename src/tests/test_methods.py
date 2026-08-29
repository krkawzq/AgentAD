from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from agentad.methods import (  # noqa: E402
    METHODS,
    AERCA,
    AERCAConfig,
    AERCALightningModule,
    AxonAD,
    AxonADConfig,
    AxonADLightningModule,
    CARLA,
    CARLAConfig,
    CARLAClassificationLightningModule,
    CARLAPretextLightningModule,
    CrossAD,
    CrossADConfig,
    CrossADLightningModule,
    DADA,
    DADAConfig,
    DADALightningModule,
    KANAD,
    KANADConfig,
    KANADLightningModule,
    Left,
    LeftConfig,
    LeftLightningModule,
    MMPAD,
    MMPADConfig,
    MMPADLightningModule,
    PaAno,
    PaAnoConfig,
    PaAnoLightningModule,
    ScatterAD,
    ScatterADConfig,
    ScatterADLightningModule,
    TimeRCD,
    TimeRCDConfig,
    TimeRCDLightningModule,
    TSPulseConfig,
    TSPulseFineTune,
    TSPulseLightningModule,
    TSPulseZeroShot,
    XLSTMAD,
    XLSTMADConfig,
    XLSTMADLightningModule,
    inject_anomalies,
)
from agentad.methods._utils import (  # noqa: E402
    evaluation_mode,
    overlap_average,
    topk_cosine_distance,
)


def assert_finite_shape(tensor, shape):
    assert tensor.shape == shape
    assert torch.isfinite(tensor).all()


def test_method_registry_exposes_detectors_but_not_agent_frameworks():
    assert set(METHODS) == {
        "AERCA",
        "AxonAD",
        "CARLA",
        "CrossAD",
        "DADA",
        "KAN-AD",
        "Left",
        "MMPAD",
        "PaAno",
        "ScatterAD",
        "Time-RCD",
        "TSPulse-FT",
        "TSPulse-ZS",
        "xLSTMAD",
    }


def test_every_method_exposes_an_independent_lightning_module():
    import lightning as L

    modules = (
        AERCALightningModule,
        AxonADLightningModule,
        CARLAPretextLightningModule,
        CARLAClassificationLightningModule,
        CrossADLightningModule,
        DADALightningModule,
        KANADLightningModule,
        LeftLightningModule,
        MMPADLightningModule,
        PaAnoLightningModule,
        ScatterADLightningModule,
        TimeRCDLightningModule,
        TSPulseLightningModule,
        XLSTMADLightningModule,
    )
    assert all(issubclass(module, L.LightningModule) for module in modules)
    assert len({module.__module__ for module in modules}) == 13


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


def test_evaluation_mode_restores_mixed_submodule_states():
    module = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.BatchNorm1d(2))
    module.train()
    module[0].eval()
    with evaluation_mode(module):
        assert not any(child.training for child in module.modules())
    assert module.training
    assert not module[0].training
    assert module[1].training


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
    model = MMPAD(MMPADConfig())
    t = torch.arange(1500, dtype=torch.float64)
    # A decaying periodic signal makes the first ACF peak the strongest one.
    signal = torch.sin(2 * torch.pi * t / 128) * torch.exp(-t / 2000)
    series = signal.unsqueeze(0).unsqueeze(-1).float()
    assert model._resolved_length(series) == 128
    # Series shorter than the original's 401-point ACF window take the
    # fallback period, clamped to the series length.
    assert model._resolved_length(torch.randn(1, 300, 1)) == 125


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
    model = MMPAD(
        MMPADConfig(
            subsequence_length=4,
            dimensions=1,
            neighbors=1,
            query_chunk_size=3,
        )
    )
    series = torch.randn(2, 16, 2)
    assert_finite_shape(model.score(series), (2, 16))
    model.fit(torch.randn(1, 12, 2))
    assert_finite_shape(model.score(series), (2, 16))


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
    assert_finite_shape(
        finetuned.score(torch.randn(1, 22, 2), batch_size=3), (1, 22)
    )


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
    model = MMPAD(
        MMPADConfig(subsequence_length=4, dimensions=1, neighbors=1),
    )
    # A flat channel is invalid and must not contribute a zero correlation;
    # the score equals the single-varying-channel score.
    assert torch.allclose(model.score(flat), model.score(series[..., :1]))


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
