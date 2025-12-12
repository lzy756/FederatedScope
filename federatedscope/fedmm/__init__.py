"""
FedMM integration for FederatedScope.

This package registers the dataset loader, model builder and trainer that
implement the FedMM optimizer and the DANN-style backbone used in the original
paper.
"""

from federatedscope.fedmm import data  # noqa: F401
from federatedscope.fedmm import model  # noqa: F401
from federatedscope.fedmm import trainer  # noqa: F401
