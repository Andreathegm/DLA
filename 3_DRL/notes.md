# Reinforce Alg. — Thoughts

## (A) Objective

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right], \quad \tau = (s_0, a_0, r_0, s_1, \ldots)$$

### Policy Gradient

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \underbrace{G_t}_{\text{return from timestamp } t}\right]$$

### trajectory probability

$$p(\tau | \theta) = \prod_{t=0}^{T} \pi_\theta(a_t | s_t) \cdot p(s_{t+1} | s_t, a_t)$$

Taking the log and separating into A and B term:

$$\log p(\tau | \theta) = \sum_{t=0}^{T} \log\!\left(\underbrace{\pi_\theta(a_t|s_t)}_{A} \cdot \underbrace{p(s_{t+1}|s_t, a_t)}_{B}\right) = \sum \log A + \sum \log B$$

deriving w.r.t  θ (B does not depend on θ):

$$\nabla_\theta \log p(\tau | \theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t)$$

---

## REINFORCE

Approximation with **one sample** (1 episode), because  in **(1)** we should have considered all trajectories.

$$\nabla_\theta J(\theta) \approx \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \quad \longrightarrow \text{high variance}$$

Loss :

$$\mathcal{L}(\theta) = -\sum_t \log \pi_\theta(a_t | s_t) \cdot G_t$$

---

## With Baseline

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t)(G_t - b(s_t))\right]$$

where $b(s_t)$ **does not depend on actions**.  
Dimostration that the expectation is zero :  

$$\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t)\right]$$

$$= \int b(s_t) \cdot \nabla_\theta \log \pi_\theta(\cdot) \cdot \pi_\theta(a|s_t)\, da$$

$$= \int b(s_t) \cdot \nabla_\theta \pi_\theta(a|s_t)\, da$$

$$= b(s_t) \cdot \nabla_\theta \underbrace{\int \pi_\theta(a|s_t)\, da}_{=1} = 0$$

### In the code we use

$$\tilde{G}_t = \frac{G_t - \mu_G}{\sigma_G + \varepsilon} \quad \text{e poi } \texttt{.mean()}$$

$$\nabla_\theta \mathcal{L} \approx \frac{1}{T} \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \tilde{G}_t$$

> so our baseline is  $\bar{G}_t$ or $\text{mean}(G_t)$