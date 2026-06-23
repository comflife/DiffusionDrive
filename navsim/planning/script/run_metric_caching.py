import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from navsim.planning.metric_caching.caching import cache_data
from navsim.planning.script.builders.worker_pool_builder import build_worker

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_metric_caching"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Build metric caches for NAVSIM scenes.
    :param cfg: omegaconf dictionary
    """
    cache_path = Path(cfg.cache.cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Metric cache path: {cache_path}")
    logger.info(f"Navsim log path: {cfg.navsim_log_path}")

    worker = build_worker(cfg)
    cache_data(cfg, worker)


if __name__ == "__main__":
    main()
