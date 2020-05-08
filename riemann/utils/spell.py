"""
This module contains utility functions for use with Spell (https://spell.run).
"""
# Spell is only imported in the Spell python environment. We use it to log metrics to Spell.
import os
import logging
from datetime import datetime
from typing import Union, Optional, Any, Dict
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

try:
    from spell.metrics import send_metric as send_spell_metric  # type: ignore
    logger.info("Logging metrics with spell")
except ImportError:
    def send_spell_metric(name: str, value: Union[str, int, float], index: Optional[int] = None):
        pass


class ProxyWriter(SummaryWriter):
    """
    Proxies `torch.utils.tensorboard.SummaryWriter` to also send metrics to Spell.
    See `torch.utils.tensorboard.SummaryWriter` for documentation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #: The last time we sent a log to Spell for a tag. Used to prevent it from throttling us
        self._last_logged: Dict[str, datetime] = {}

    def add_scalar(self, tag: Any, scalar_value: Any, global_step: Optional[Any] = None,
                   walltime: Optional[Any] = None) -> None:
        # Inherits documentation from SummaryWriter

        super().add_scalar(tag, scalar_value, global_step, walltime)

        # Prepend tags with log_dir to differentiate train and eval metrics
        log_dir = os.path.basename(self.get_logdir())
        tag_ = f"{log_dir}/{tag}"

        # Only send logs to Spell every second to prevent throttling
        if (datetime.now() - self._last_logged.get(tag_, datetime.now())).seconds > 1:
            send_spell_metric(tag_, scalar_value, global_step)
            self._last_logged[tag_] = datetime.now()
