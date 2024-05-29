from src.utilities.program_args import parse_program_args
from src.utilities.constants import SENTENCE_TRANSFORMERS
from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings
from src.models.str_siamese_mlp import STRSiameseMLP

"""Hyperparameters for Siamese MLP: 
Architecture: shared layer, common layer, activation function, dropout
Transformer;
Optimizer: type, learning rate;
Early stopping: type, patience;
Training: epochs, batch size; """
