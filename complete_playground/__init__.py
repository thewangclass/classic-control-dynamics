"""Root '__init___' of the complete_playground module setting the '__all__' of complete_playground modules"""

from complete_playground.core import (
    Env
)
from complete_playground.spaces.space import Space
from complete_playground.envs.registration import (
    make,
    spec,
    register,
    registry,
    pprint_registry,    
)
from complete_playground import envs, spaces, rl_training_scripts, utils, error, logger

__all__ = [
    # core classes
    "Env",
    "Space",

    # registration

    # module folders
    "envs",
    "spaces",
    "utils",
    "error",
    "logger",
]
__version__ = "0.00.0"