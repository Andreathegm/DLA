import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pygame
from device import device

_ = pygame.init()

# ── Cache della gamma matrix ─────────────────────────────────────────────────
# Problema originale: compute_gamma_matrix veniva chiamata dentro compute_returns
# ad ogni episodio, riallocando e ricalcolando una matrice T×T ogni volta.
# Con T=500: 500×500 float32 = 1 MB allocato/deallocato 1000 volte = spreco puro.
#
# Soluzione: teniamo una cache {T: matrice} così la matrice viene costruita
# al più una volta per ogni lunghezza di episodio incontrata.
# ────────────────────────────────────────────────────────────────────────────
_gamma_matrix_cache: dict[tuple, torch.Tensor] = {}

def compute_gamma_matrix(T: int, gamma: float) -> torch.Tensor:
    key = (T, gamma)
    if key not in _gamma_matrix_cache:
        rows = torch.arange(T).unsqueeze(0)          # shape (1, T)
        cols = torch.arange(T).unsqueeze(1)           # shape (T, 1)
        mat  = torch.triu(float(gamma) ** (rows - cols))  # shape (T, T)
        # ── Perché .to(device) QUI è corretto ───────────────────────────────
        # La matrice è costante per tutta la vita del programma: la costruiamo
        # una volta e la teniamo già sul device giusto. Non c'è nessun
        # trasferimento ripetuto: è esattamente il caso d'uso ideale per GPU
        # (un'operazione @  matrice × vettore grande e stabile). Su CPU il
        # .to(device) è no-op (il tensore è già lì), quindi non fa danni.
        # ────────────────────────────────────────────────────────────────────
        _gamma_matrix_cache[key] = mat.to(device)
    return _gamma_matrix_cache[key]


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    # ── Perché .to(device) QUI è corretto ───────────────────────────────────
    # rewards arriva come lista Python (CPU-side per definizione). Dobbiamo
    # convertirla in tensore prima di fare la moltiplicazione con la gamma
    # matrix che è già su device. Questo è l'unico posto dove il dato
    # "nasce" lato CPU e deve salire sul device: ha senso farlo una volta sola
    # qui, non step by step dentro il loop di run_episode.
    # ────────────────────────────────────────────────────────────────────────
    r = torch.tensor(rewards, dtype=torch.float32).to(device)
    G = compute_gamma_matrix(len(rewards), gamma)   # già su device (dalla cache)
    return G @ r   # shape (T,)


def select_action(env, obs: torch.Tensor, policy) -> tuple[int, torch.Tensor]:
    # obs arriva già come tensore su device (lo mette run_episode), quindi
    # nessun .to() necessario qui.
    dist     = Categorical(policy(obs))
    action   = dist.sample()
    log_prob = dist.log_prob(action)   # già su device, niente .to()
    return (action.item(), log_prob.reshape(1))


def run_episode(env, policy, maxlen: int = 500):
    observations = []
    actions      = []
    log_probs    = []
    rewards      = []

    (obs, info) = env.reset()
    for _ in range(maxlen):
        # ── Perché .to(device) QUI è necessario ─────────────────────────────
        # obs arriva da env.step() come numpy array (CPU). Dobbiamo portarlo
        # su device per darlo alla policy. Non c'è alternativa: gym restituisce
        # sempre numpy. Tuttavia lo facciamo UNA VOLTA PER STEP, non più volte
        # come accadeva prima (log_prob aveva un .to() ridondante in select_action).
        # Con device=CPU questo .to() è un no-op istantaneo.
        # ────────────────────────────────────────────────────────────────────
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

        (action, log_prob) = select_action(env, obs_tensor, policy)
        observations.append(obs_tensor)
        actions.append(action)
        log_probs.append(log_prob)

        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break

    # torch.cat qui è già efficiente: concatena N tensori scalar→(1,) in un
    # unico vettore (T,) in un'unica operazione, tutto su device.
    return (observations, actions, torch.cat(log_probs), rewards)


def evaluate_policy(env, policy, M: int) -> tuple[float, float]:
    policy.eval()
    total_rewards = []
    episode_lengths = []
    
    with torch.no_grad():  # niente gradients durante valutazione
        for _ in range(M):
            (observations, actions, log_probs, rewards) = run_episode(env, policy)
            total_rewards.append(sum(rewards))
            episode_lengths.append(len(rewards))
    
    return (np.mean(total_rewards), np.mean(episode_lengths))