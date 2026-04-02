# mc_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, Optional
import numpy as np
from itertools import product


# ============================================================
# Generic finite-state machinery
# ============================================================

@dataclass
class FiniteMarkovModel:
    """
    Generic container for a finite Markov chain on an enumerated state space.

    Required fields:
      - states: array (n, d_state) or (n,) object-like
      - pi: stationary distribution as array (n,), sums to 1
      - P: transition matrix as array (n, n), row-stochastic
    """
    states: np.ndarray
    pi: np.ndarray
    P: np.ndarray

    def sanity_check(self, atol: float = 1e-12) -> None:
        n = self.P.shape[0]
        if self.P.shape != (n, n):
            raise ValueError("P must be square (n,n).")
        if self.pi.shape != (n,):
            raise ValueError("pi must be shape (n,).")
        if np.any(self.P < -atol):
            raise ValueError("P has negative entries beyond tolerance.")
        if not np.allclose(self.P.sum(axis=1), 1.0, atol=atol):
            raise ValueError("Rows of P do not sum to 1.")
        if np.any(self.pi < -atol):
            raise ValueError("pi has negative entries beyond tolerance.")
        if not np.allclose(self.pi.sum(), 1.0, atol=atol):
            raise ValueError("pi does not sum to 1.")


def stable_softmax_logweights(logw: np.ndarray) -> np.ndarray:
    """
    Convert log-weights to a probability vector stably:
      pi_i ∝ exp(logw_i)
    """
    logw = np.asarray(logw, dtype=np.float64)
    m = logw.max()
    w = np.exp(logw - m)
    return w / w.sum()


# ============================================================
# Curie–Weiss on hypercube: concrete model + Glauber(MH) kernel
# ============================================================

def hypercube_states(d: int) -> np.ndarray:
    """
    Enumerate X = {-1,+1}^d as an array of shape (2^d, d).
    Lexicographic ordering over product([-1,1], repeat=d).
    """
    return np.array(list(product([-1, 1], repeat=d)), dtype=np.int8)


def interaction_matrix_decay(d: int) -> np.ndarray:
    """
    Interaction matrix J with entries J_{ij} = 2^{-|i-j|}.
    Indices i,j in {0,...,d-1}.
    """
    idx = np.arange(d)
    dist = np.abs(idx[:, None] - idx[None, :])
    return 2.0 ** (-dist)


def hamiltonian_curie_weiss(states: np.ndarray, J: np.ndarray, h: float) -> np.ndarray:
    """
    H(x) = - x^T J x - h * sum_i x_i, vectorised over all states.
    states: (n,d), J: (d,d)
    returns H: (n,)
    """
    X = states.astype(np.float64)
    quad = np.einsum("ni,ij,nj->n", X, J, X)
    lin = X.sum(axis=1)
    return -(quad + h * lin)


def gibbs_pi_from_H(H: np.ndarray, T: float) -> np.ndarray:
    """
    pi(x) ∝ exp(-H(x)/T)
    """
    if T <= 0:
        raise ValueError("Temperature T must be > 0.")
    logw = -H / T
    return stable_softmax_logweights(logw)


def build_glauber_mh_P(states: np.ndarray, H: np.ndarray, T: float) -> np.ndarray:
    """
    Build full transition matrix for:
      - pick coordinate i uniformly from {1,...,d}
      - propose flipping x_i
      - accept with exp(-(H(y)-H(x))_+ / T)

    So for single-flip neighbor y of x:
      P(x,y) = (1/d) * exp(-max(H(y)-H(x),0)/T)
    and P(x,x) makes the row sum 1.
    """
    n, d = states.shape
    if H.shape != (n,):
        raise ValueError("H must have shape (n,).")

    # fast lookup: state -> index
    key: Dict[Tuple[int, ...], int] = {tuple(states[k].tolist()): k for k in range(n)}
    P = np.zeros((n, n), dtype=np.float64)

    for x_idx in range(n):
        x = states[x_idx]
        Hx = H[x_idx]
        row_sum = 0.0

        for i in range(d):
            y = x.copy()
            y[i] = -y[i]
            y_idx = key[tuple(y.tolist())]
            Hy = H[y_idx]

            delta = Hy - Hx
            acc = np.exp(-max(delta, 0.0) / T)
            p_xy = (1.0 / d) * acc

            P[x_idx, y_idx] = p_xy
            row_sum += p_xy

        P[x_idx, x_idx] = 1.0 - row_sum

    return P


def make_curie_weiss_hypercube_model(
    d: int,
    T: float,
    h: float,
    J: Optional[np.ndarray] = None,
) -> FiniteMarkovModel:
    """
    Factory: returns (states, pi, P) for the Curie–Weiss hypercube example.
    """
    states = hypercube_states(d)
    if J is None:
        J = interaction_matrix_decay(d)
    H = hamiltonian_curie_weiss(states, J, h)
    pi = gibbs_pi_from_H(H, T)
    P = build_glauber_mh_P(states, H, T)

    model = FiniteMarkovModel(states=states, pi=pi, P=P)
    model.sanity_check()
    return model


def random_cut_mask(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Return a random boolean mask of length n representing S.
    Ensures S is non-empty and not all of X.
    
    choose subset of {1,...,n-1} uniformly, except exclude all ones"""
    if rng is None:
        rng = np.random.default_rng()

    while True:
        rest = rng.integers(0, 1 << (n-1))
        if rest != (1 << (n-1)) - 1:
            break

    S = np.zeros(n, dtype=bool)
    S[0] = True
    for j in range(1, n):
        S[j] = bool((rest >> (j-1)) & 1)

    return S


def build_G_from_partition(pi: np.ndarray, S_mask: np.ndarray) -> np.ndarray:
    """
    Build the 2-block averaging kernel G_S for a partition X = S U S^c.
    G_S(x,·) = pi(·|S) if x in S
             = pi(·|S^c) if x in S^c
    """
    pi = np.asarray(pi, dtype=np.float64)
    S_mask = np.asarray(S_mask, dtype=bool)

    n = pi.shape[0]
    if pi.shape != (n,):
        raise ValueError("pi must be shape (n,).")
    if S_mask.shape != (n,):
        raise ValueError("S_mask must be shape (n,).")
    if not S_mask.any() or S_mask.all():
        raise ValueError("S must be non-empty and not equal to X.")

    piS = float(pi[S_mask].sum())
    piSc = float(pi[~S_mask].sum())
    if piS <= 0 or piSc <= 0:
        raise ValueError("Partition must have positive pi-mass on both sides.")

    G = np.zeros((n, n), dtype=np.float64)

    # rows corresponding to x in S
    for i in np.where(S_mask)[0]:
        G[i, S_mask] = pi[S_mask] / piS

    # rows corresponding to x in S^c
    for i in np.where(~S_mask)[0]:
        G[i, ~S_mask] = pi[~S_mask] / piSc

    # sanity checks
    if not np.allclose(G.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("G rows do not sum to 1.")
    if np.any(G < -1e-14):
        raise RuntimeError("G has negative entries.")

    return G




