# Module to find word analogies

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Sequence

def cosine_distance(x: NDArray, y: NDArray) -> NDArray:
    """Computes the cosine distance between all sets of two vectors

    Args:
        x (NDArray, shape [n, d]): First d-dimensional set of vectors
        y (NDArray, shape [m, d]): Second d-dimensional set of vectors

    Returns:
        NDArray, shape [n, m]: Matrix of cosine distances
    """
    x_norm = x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-12)
    y_norm = y / (np.linalg.norm(y, ord=2, axis=-1, keepdims=True) + 1e-12)
    return 1 - (x_norm @ y_norm.T)

def get_analogies(embedding_matrix: NDArray, triplet: Tuple[str, str, str],
                          token_to_idx: Dict[str, int], idx_to_token: Dict[int, str],
                          num_candidates: int=5) -> Sequence[str]:
    """ For a triplet a, b, d finds the token whose embedding is closest to u_a - u_b + u_d

    Args:
        embedding_matrix (NDArray, shape [vocabulary_size, embedding_dim]): Token embeddings
        triplet (Tuple[str, str, str]): The tokens a, b, d
        token_to_idx (Dict[str, int]): Mapping from token to indices in the embedding matrix
        idx_to_token (Dict[int, str]): Mapping from index in the embedding matrix to token
        num_candidates (int, default: 5): How many tokens to suggest, i.e. returns the `num_candidates` most similar tokens

    Returns:
        Sequence[str]: The `num_candidates` most similar tokens with respect to the embedding
    """
    a, b, d = triplet
    
    """ 
    Computes:
        result (NDArray, shape [embedding_dim]): The embedding vector u_a - u_b + u_d
    """
    ###########################
    # YOUR CODE HERE
    ###########################

    w_a = embedding_matrix[token_to_idx[a],:]
    w_b = embedding_matrix[token_to_idx[b],:]
    w_d = embedding_matrix[token_to_idx[d],:]
    result = w_a - w_b + w_d
    
    distances = cosine_distance(result[None, :], embedding_matrix) # 1, vocabulary_size
    candidates = [idx_to_token[idx] for idx in np.argsort(distances.flatten())]
    candidates = [token for token in candidates if token not in [a, b, d]][:num_candidates]
    return candidates
    
    