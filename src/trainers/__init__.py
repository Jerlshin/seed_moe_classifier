"""Hydra entry points for the two training stages (paper Sections 4 and 5).

Both modules expose a ``main`` decorated with ``@hydra.main``. They are launched
as modules rather than imported, so this package deliberately does not import
them at module load time -- doing so would trigger Hydra's config search on any
``import src.trainers``.

    python -m src.trainers.contrastive_pretrain experiment=pretrain_swinv2_dino
    python -m src.trainers.moe_finetune        experiment=finetune_hierarchical_moe
"""

__all__: list[str] = []
