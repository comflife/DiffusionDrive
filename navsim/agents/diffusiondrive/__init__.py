"""
DiffusionDrive Agent for NavSim.

Includes:
- TransfuserAgent: Original DiffusionDrive with diffusion-based trajectory generation
- TransfuserAgentAR: Discrete Autoregressive version with token-based trajectory generation
"""

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

# Optional imports - may fail if dependencies are missing
try:
    from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
except ImportError:
    TransfuserAgent = None

try:
    from navsim.agents.diffusiondrive.transfuser_agent_ar import TransfuserAgentAR
except ImportError:
    TransfuserAgentAR = None

__all__ = [
    "TransfuserAgent",
    "TransfuserAgentAR",
    "TransfuserConfig",
]
