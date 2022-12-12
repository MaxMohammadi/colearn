import argparse
import os
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights, ColearnModel, ModelFormat, convert_model_to_onnx
from colearn.training import initial_result, collective_learning_round, set_equal_weights
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results
from colearn_other.fraud_dataset import fraud_preprocessing