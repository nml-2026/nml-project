from __future__ import annotations

from torch import nn

from ..config import RunConfig
from .baselines import JointAutoregressiveGRUModel, JointGRUForecastModel, NotebookExactModel
from .graph import DynamicGraphAutoregressiveModel, DynamicGraphDirectModel, StaticGraphDirectModel
from .graph_transformer import GraphTransformerForecastModel
from .ssm import GraphSSMDirectModel, SocialSSMForecastModel


def build_model(config: RunConfig) -> nn.Module:
    experiment = config.experiment
    model_cfg = config.model
    model_name = experiment.model_name
    common_kwargs = {
        "input_dim": config.input_dim,
        "num_entities": experiment.num_entities,
        "horizon_size": experiment.horizon_size,
        "hidden_dim": model_cfg.hidden_dim,
        "message_dim": model_cfg.message_dim,
        "num_layers": model_cfg.num_layers,
        "dropout": model_cfg.dropout,
    }
    if model_name == "e0_reference":
        return NotebookExactModel(
            input_dim=config.input_dim,
            hidden_dim=model_cfg.hidden_dim,
            horizon_size=experiment.horizon_size,
            context_size=experiment.context_size,
        )
    if model_name == "e1_joint_gru":
        return JointGRUForecastModel(
            input_dim=config.input_dim,
            num_entities=experiment.num_entities,
            context_size=experiment.context_size,
            horizon_size=experiment.horizon_size,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
        )
    if model_name == "e1_joint_autoreg_gru":
        return JointAutoregressiveGRUModel(
            input_dim=config.input_dim,
            num_entities=experiment.num_entities,
            horizon_size=experiment.horizon_size,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
            use_velocity=config.features.use_velocity,
        )
    if model_name == "e1_social_ssm":
        return SocialSSMForecastModel(
            input_dim=config.input_dim,
            num_entities=experiment.num_entities,
            horizon_size=experiment.horizon_size,
            hidden_dim=model_cfg.hidden_dim,
            dropout=model_cfg.dropout,
            use_semantic_flags=model_cfg.use_semantic_flags,
            use_entity_embeddings=model_cfg.use_entity_embeddings,
            entity_embedding_dim=model_cfg.entity_embedding_dim,
            temporal_ssm_layers=model_cfg.temporal_ssm_layers,
            social_ssm_layers=model_cfg.social_ssm_layers,
            decoder_ssm_layers=model_cfg.decoder_ssm_layers,
            ssm_conv_kernel=model_cfg.ssm_conv_kernel,
            separate_ball_head=model_cfg.separate_ball_head,
        )
    if model_name == "e2_static_graph":
        return StaticGraphDirectModel(
            **common_kwargs,
            use_semantic_flags=model_cfg.use_semantic_flags,
            message_passing_type=model_cfg.message_passing_type,
            use_relation_types=model_cfg.use_relation_types,
            use_entity_embeddings=model_cfg.use_entity_embeddings,
            entity_embedding_dim=model_cfg.entity_embedding_dim,
            edge_radius=model_cfg.edge_radius,
        )
    if model_name in {"e3_dynamic_graph", "e3_graph_ssm"}:
        dynamic_kwargs = dict(
            **common_kwargs,
            k_neighbors=model_cfg.k_neighbors,
            use_semantic_flags=model_cfg.use_semantic_flags,
            dynamic_edges=model_cfg.dynamic_edges,
            use_velocity=config.features.use_velocity,
            message_passing_type=model_cfg.message_passing_type,
            separate_ball_head=model_cfg.separate_ball_head,
            use_relation_types=model_cfg.use_relation_types,
            use_entity_embeddings=model_cfg.use_entity_embeddings,
            entity_embedding_dim=model_cfg.entity_embedding_dim,
            edge_radius=model_cfg.edge_radius,
        )
    if model_name == "e3_dynamic_graph":
        if model_cfg.decoder_type == "direct":
            return DynamicGraphDirectModel(
                **dynamic_kwargs,
                direct_decoder_heads=model_cfg.direct_decoder_heads,
            )
        return DynamicGraphAutoregressiveModel(**dynamic_kwargs)
    if model_name == "e3_graph_ssm":
        return GraphSSMDirectModel(
            **dynamic_kwargs,
            temporal_ssm_layers=model_cfg.temporal_ssm_layers,
            social_ssm_layers=model_cfg.social_ssm_layers,
            decoder_ssm_layers=model_cfg.decoder_ssm_layers,
            ssm_conv_kernel=model_cfg.ssm_conv_kernel,
        )
    if model_name == "e5_graph_transformer":
        return GraphTransformerForecastModel(
            input_dim=config.input_dim,
            num_entities=experiment.num_entities,
            context_size=experiment.context_size,
            horizon_size=experiment.horizon_size,
            hidden_dim=model_cfg.hidden_dim,
            dropout=model_cfg.dropout,
            k_neighbors=model_cfg.k_neighbors,
            use_semantic_flags=model_cfg.use_semantic_flags,
            dynamic_edges=model_cfg.dynamic_edges,
            separate_ball_head=model_cfg.separate_ball_head,
            use_entity_embeddings=model_cfg.use_entity_embeddings,
            entity_embedding_dim=model_cfg.entity_embedding_dim,
            edge_radius=model_cfg.edge_radius,
            transformer_heads=model_cfg.transformer_heads,
            spatial_layers=model_cfg.spatial_layers,
            temporal_layers=model_cfg.temporal_layers,
            decoder_layers=model_cfg.decoder_layers,
            transformer_ffn_multiplier=model_cfg.transformer_ffn_multiplier,
            graph_ablation_mode=model_cfg.graph_ablation_mode,
            graph_prep_backend=model_cfg.graph_prep_backend,
        )
    raise ValueError(f"Unknown model name {model_name!r}.")
