# coding: utf-8

from riemann.config.config_loader import initialize_config
initialize_config()
from riemann.data.data_loader import get_training_data
g = get_training_data()
from riemann.config.graph_sampling_config import GraphSamplingConfig
iter = g.get_neighbor_iterator(GraphSamplingConfig())

