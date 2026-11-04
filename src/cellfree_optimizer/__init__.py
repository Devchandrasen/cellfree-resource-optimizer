"""Cell-free wireless resource allocation."""

from .config import NetworkConfig
from .network import Network, generate_network

__version__ = "0.1.0"
__all__ = ["Network", "NetworkConfig", "generate_network", "optimize", "validate_solution"]


def optimize(network, method="joint", seconds=60.0, seed=0):
    """Optimize one network with a fresh, independently charged deadline."""
    from .optimizer import optimize as implementation

    return implementation(network, method=method, seconds=seconds, seed=seed)


def validate_solution(network, solution, tolerance=2e-6):
    """Check every physical and allocation constraint without repairing the input."""
    from .validation import validate_solution as implementation

    return implementation(network, solution, tolerance=tolerance)
