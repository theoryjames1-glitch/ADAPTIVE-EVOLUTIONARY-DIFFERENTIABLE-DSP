
# 🎛 AE-DDSP: The Core Idea

**Plant** = differentiable DSP system with parameters $\theta$ (oscillators, filters, envelopes, mixers).
**Controller** = two coupled mechanisms:

1. **Adaptive (gradient) control**: filtered gradients update $\theta$ with **adaptive hyperparameters** (learning rate $\alpha$, momentum $\mu$, etc.) tuned online from scalar feedback signals (loss, reward, variance, resonance measures).
2. **Evolutionary dither**: small, shaped, **stochastic perturbations** that probe nearby parameter space to escape flat/sticky regions and validate local improvements.

Everything runs in **discrete time** (DSP style), with bounded gains and stability monitors.

---

# 1) System Model (Discrete-Time Control)

State:

$$
x_t \triangleq \begin{bmatrix}\theta_t \\ h_t \\ \alpha_t \\ \mu_t \\ \sigma_t\end{bmatrix},
\quad
y_t \triangleq \begin{bmatrix}\ell_t \\ r_t\end{bmatrix}
$$

* $\theta_t \in \mathbb{R}^d$: DSP parameters (cutoff, Q, osc mix, env times, etc.)
* $h_t$: optimizer internal state (e.g., momentum/EMAs)
* $\alpha_t,\mu_t$: adaptive hyperparameters
* $\sigma_t$: evolutionary dither strength
* $\ell_t=L(\theta_t;\xi_t)$: task loss (often spectral/perceptual)
* $r_t$: optional reward (e.g., spectral match, resonance reward)

Signals (filtered feedback):

$$
\bar{\ell}_t = \mathrm{EMA}_\beta(\ell_t),\quad
v_t = \mathrm{VarWindow}(\ell_{t-k:t}),\quad
\Delta \ell_t = \bar{\ell}_{t-1}-\bar{\ell}_t,\quad
\Delta r_t = \bar{r}_t-\bar{r}_{t-1}.
$$

Plant dynamics (filtered gradient + dither):

$$
\begin{aligned}
h_{t+1} &= \Phi(h_t, g_t; \mu_t) \\
u_t &= U(h_{t+1}, g_t; \alpha_t) \\
\theta_{t+1} &= \theta_t - u_t + \sigma_t\,D(\theta_t)\,\eta_t
\end{aligned}
$$

* $g_t=\nabla_\theta \widehat{L}_t(\theta_t)$ (noisy grad)
* $\Phi$ (e.g., $h_{t+1}=\mu_t h_t+(1-\mu_t)g_t$)
* $U$ (e.g., $U=\alpha_t h_{t+1}$ or Adam-style)
* $D(\theta_t)$: **shaping** of dither (per-tensor RMS or preconditioner)
* $\eta_t\sim\mathcal{N}(0,I)$: zero-mean perturbation

---

# 2) Differentiable DSP Plant

Audio generation:

$$
x(t;\theta) \xrightarrow{\text{STFT}} X(f,\tau),
\quad \text{features}~\phi(X) \in \mathbb{R}^m
$$

* Modules (all differentiable): VCOs, waveshapers, VCFs (ladder approximations), VCAs, envelopes.
* Losses $L$: **feature-space** (log-mel, centroid, rolloff, flatness, peakiness), optionally **multi-resolution**.

**Any signal can be reward**: $r_t=f(\phi),$ including resonance metrics, stability, human/BCI coherence—composed via smooth, bounded functions (e.g., exponentials, sigmoids).

---

# 3) Adaptive Law (Scheduler as Envelope)

Hyperparameters updated from **scalar feedback** (no gradients through the scheduler):

$$
\begin{aligned}
\alpha_{t+1} &= \mathrm{clip}\Big(\alpha_t \cdot f_\alpha(\Delta \ell_t, v_t, \Delta r_t)\Big) \\
\mu_{t+1}     &= \mathrm{clip}\Big(\mu_t + f_\mu(\Delta \ell_t, v_t, \Delta r_t)\Big) \\
\sigma_{t+1} &= \mathrm{clip}\Big(\sigma_t \cdot f_\sigma(\text{plateau}_t, v_t)\Big)
\end{aligned}
$$

Typical, robust choices:

* **Trend/variance-aware LR**:

  $$
  f_\alpha=
  \begin{cases}
  u & \text{if } \Delta \ell_t>\varepsilon \ (\text{improving})\\
  d/(1+\lambda v_t) & \text{else (damp when noisy)}
  \end{cases}
  \quad u>1,\ d\in(0,1)
  $$
* **Momentum nudged by reward trend**:

  $$
  f_\mu=
  \begin{cases}
  +\delta_\mu & \text{if } \Delta r_t>0 \\
  -\kappa_\mu \mu_t & \text{otherwise}
  \end{cases}
  $$
* **Dither gating**:
  Increase $\sigma$ only on plateaus; reduce when $v_t$ high.

Optional **cosine envelope** overlay on $\alpha_t$ for gentle macro-scheduling.

---

# 4) Evolutionary Dither = Controlled Exploration

No parents, no genotype—just **signal-space probes**:

* Add small, shaped noise $\sigma_t D(\theta_t)\eta_t$ every step.
* **Sparse local tests** (optional): at selected steps, evaluate a few perturbed copies $\theta_t+\delta_i$; if some yield lower loss, **average-in** a small fraction:

  $$
  \theta_{t+1}\leftarrow (1-\rho)\theta_{t+1} + \rho \cdot \mathrm{argmin}_i~L(\theta_t+\delta_i)
  $$

  Update success rate → feeds $f_\sigma$.

This avoids long stagnation and helps cross narrow valleys without destabilizing the gradient loop.

---

# 5) Stability & Deep Resonance

**Bounded gains**: $\alpha\in[\alpha_{\min},\alpha_{\max}],\ \mu\in[0,1),\ \sigma \ll 1$.
**Variance damping**: reduce $\alpha$ when $v_t$ rises.
**Plateau detection**: boost $\sigma$ only when $\Delta\ell\approx 0$ for $p$ steps.
**Lyapunov heuristic**:

$$
\mathbb{E}[\ell_{t+1}-\ell_t\mid x_t] < 0
$$

enforced in practice by (i) LR damping on high variance, (ii) tiny dither except on plateaus, (iii) optional **risk penalty** (grad-norm, loudness, knob slew).

**Deep Resonance** target: keep updates in a **bounded, coherent oscillatory regime** (healthy “ringing”), not overdamped (stagnation) nor explosive (divergence). Use spectral/optimizer resonance rewards to steer.

---

# 6) Reward & Loss Design (feature space)

Composite reward (bounded, smooth):

$$
R_t = w_\text{aud}R^{\text{aud}} + w_\text{opt}R^{\text{opt}} + w_\text{pol}R^{\text{pol}} - \mathcal{P}_t
$$

* $R^{\text{aud}}$: log-mel similarity, centroid/rolloff targets, flatness, peakiness
* $R^{\text{opt}}$: stability (grad-norm under threshold), coherent oscillation (autocorr), small update norm
* $R^{\text{pol}}$: periodicity in knob trajectories + diversity of visited states
* $\mathcal{P}_t$: safety penalties (RMS loudness cap, slew-rate, KL to ref policy if used)

This can drive **RL updates** (advantage shaping) and/or **direct gradient control** (minimize $-R_t$).

---

# 7) Minimal Algorithm (Windowed AE-DDSP)

**Loop per window** (e.g., 0.25–1.0 s with hop overlap):

1. **Synthesize** audio $x$ using $\theta_t$.
2. **Encode** features $\phi(x)$ (multi-res STFT/log-mel + stats).
3. **Compute** loss $L$ and reward $R$ (composite).
4. **Filter** $(\bar{\ell}_t, v_t, \bar{r}_t)$; compute $\Delta \ell_t, \Delta r_t$.
5. **Scheduler**: update $(\alpha,\mu,\sigma)$.
6. **Gradient step**: $g_t=\nabla_\theta L$, $h_{t+1}=\Phi(h_t,g_t;\mu)$, $u_t=U(...)$.
7. **Dither**: $\theta_{t+1}=\theta_t-u_t+\sigma D(\theta_t)\eta_t$.
8. (Optional) **local tests** and small average-in if improved.
9. **Clamp** parameters to valid ranges (e.g., cutoff $>0$, Q in \[$q_{\min},q_{\max}$]).
10. **Log** resonance/stability metrics for adaptivity.

---

# 8) Practical Defaults

* **LR adaptivity**: $u\!\approx\!1.02,\ d\!\approx\!0.8,\ \lambda\!\in[1,10]$ for variance damping; $\alpha$ bounds around your known good LR (e.g., $[0.3, 3]\times\alpha_0$).
* **Momentum adaptivity**: $\delta_\mu\!\approx\!0.02,\ \kappa_\mu\!\approx\!0.1$.
* **Dither**: $\sigma_0 \sim 10^{-3}$ (relative RMS); raise on plateaus only; shape $D(\theta)$ with per-tensor RMS.
* **Features**: two FFT sizes (e.g., 1024 & 4096), hop 256, mel-64; stabilize with log and tiny $\epsilon$.
* **Regularizers**: loudness cap + knob slew; optional KL to a safe reference control policy.

---

# 9) Why AE-DDSP Works

* **Adaptive control** aligns update dynamics with task feedback and noise statistics (avoid both stall and blow-up).
* **Evolutionary dither** injects controlled exploration to escape flats & poor curvature—without disrupting stability.
* **Feature-space objectives** keep gradients perceptually grounded and low-dimensional.
* **Resonance regulation** (optimizer + audio) keeps the system in a productive, bounded oscillatory regime (the “sweet spot” for fast learning).
