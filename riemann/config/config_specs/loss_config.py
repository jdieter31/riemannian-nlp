from ..config import ConfigDict
from ..manifold_initialization_config import ManifoldInitializationConfig

CONFIG_NAME = "loss"


class LossConfig(ConfigDict):
    """
    Configuration for loss functions
    """
    #: The margin of the neighborhood loss function.
    margin: float = 0.3

    #: Number of random samples to use when measuring isometry
    random_isometry_samples: int = 30
    random_isometry_initialization: ManifoldInitializationConfig = ManifoldInitializationConfig(
            default_params=[-1.0, 1.0]
        )

    #: If true, we'll use the conformality regularizer
    use_conformality_regularizer: bool = True
    #: Confirmality ranges between 1 and 0; a value of 1 is equivalent to being isometric, while
    #: a value of 0 is unbounded conformality; an intermediate value is bounded.
    conformality: float = 1.0
