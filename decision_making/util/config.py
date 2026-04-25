from datetime import datetime
from pathlib import Path
from typing import Any

from util.logger import logger
import yaml


class ConfigParser:
    """Manages configuration loading and validation."""

    def __init__(self, args):
        """Initialize the configuration manager."""
        self.config_path = Path(args.config)
        self.trading_date = args.trading_date
        self.config = self.load_config()

    def load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        cfg = {}
        try:
            with self.config_path.open(encoding="utf-8") as f:
                logger.info(f"Loading configuration from {self.config_path}")
                cfg = yaml.safe_load(f)
        except FileNotFoundError as err:
            raise ValueError(f"Configuration file not found: {self.config_path}") from err
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}") from e

        cfg["trading_date"] = datetime.strptime(self.trading_date, "%Y-%m-%d")
        cfg["planner_mode"] = cfg.get("planner_mode", False)

        return cfg

    def get_config(self) -> dict[str, Any]:
        """Get the configuration."""
        return self.config
