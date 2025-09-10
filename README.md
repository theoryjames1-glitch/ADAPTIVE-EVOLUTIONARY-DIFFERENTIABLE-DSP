
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


### PSEUDOCODE

Great — here’s a **PyTorch scaffold** for **Adaptive-Evolutionary DifferentiableDSP (AE-DDSP)**. It’s modular:

* **Plant** = differentiable DSP module (stub you can extend with oscillators, filters, envelopes).
* **Feature encoder** = multi-resolution log-mel + spectral stats.
* **Scheduler** = adaptive law for $\alpha,\mu,\sigma$.
* **AE-DDSP loop** = closed-loop training step matching the equations.

---

```python
import torch
import torch.nn as nn
import torchaudio
import math

# ------------------------------------------------------------
# 1. Differentiable DSP Plant (stub)
# ------------------------------------------------------------
class DifferentiableSynth(nn.Module):
    def __init__(self, n_samples=16000, sr=16000):
        super().__init__()
        self.sr = sr
        self.n_samples = n_samples
        # Example params: osc freq, cutoff, resonance (all trainable)
        self.freq = nn.Parameter(torch.tensor([220.0]))
        self.cutoff = nn.Parameter(torch.tensor([2000.0]))
        self.res = nn.Parameter(torch.tensor([0.5]))
        
    def forward(self):
        # Simple oscillator + envelope stub
        t = torch.linspace(0, self.n_samples/self.sr, self.n_samples, device=self.freq.device)
        wave = torch.sin(2*math.pi*self.freq*t)
        # Fake filter (cutoff/res just rescale for now)
        audio = torch.tanh(self.cutoff/2000.0 * wave) * (1.0 - self.res)
        return audio.unsqueeze(0)  # [1, T]


# ------------------------------------------------------------
# 2. Feature Encoder (multi-res log-mel + stats)
# ------------------------------------------------------------
class SpectralFeatures(nn.Module):
    def __init__(self, sr=16000, n_mels=64, fft_sizes=(1024, 4096), hop=256):
        super().__init__()
        self.sr = sr
        self.melspecs = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
            ) for n_fft in fft_sizes
        ])
    
    def forward(self, audio):
        feats = []
        for mel in self.melspecs:
            S = mel(audio) + 1e-6
            logS = torch.log(S)
            feats.append(logS)
        return feats  # list of [B, n_mels, T]


def spectral_stats(S):
    """Compute centroid, rolloff, flatness from spectrogram (power)."""
    freqs = torch.linspace(0, 1, S.size(1), device=S.device)  # normalized freq
    A = S.mean(dim=-1)  # [B, F] average over time
    
    centroid = (A * freqs).sum(dim=-1) / (A.sum(dim=-1) + 1e-8)
    cum_energy = torch.cumsum(A, dim=-1) / (A.sum(dim=-1, keepdim=True) + 1e-8)
    rolloff95 = (cum_energy >= 0.95).float().argmax(dim=-1).float() / A.size(1)
    flatness = torch.exp((torch.log(A+1e-8).mean(dim=-1))) / (A.mean(dim=-1)+1e-8)
    return centroid, rolloff95, flatness


# ------------------------------------------------------------
# 3. Scheduler (adaptive law for α, μ, σ)
# ------------------------------------------------------------
class Scheduler:
    def __init__(self, α0=1e-3, μ0=0.9, σ0=1e-3, cfg=None):
        self.α, self.μ, self.σ = α0, μ0, σ0
        self.ℓ_bar, self.r_bar = None, None
        self.prev_ℓ_bar, self.prev_r_bar = None, None
        self.ema_beta = 0.9
        self.cfg = cfg or {
            "α_min":1e-5, "α_max":1e-1,
            "μ_min":0.0, "μ_max":0.999,
            "σ_min":1e-6, "σ_max":1e-2,
            "u":1.02, "d":0.8, "λ":5.0,
            "δμ":0.01, "κμ":0.1
        }
    
    def ema(self, prev, x):
        return x if prev is None else self.ema_beta*prev + (1-self.ema_beta)*x
    
    def update(self, ℓ, r, var=0.0):
        # Filter signals
        self.ℓ_bar = self.ema(self.ℓ_bar, ℓ)
        self.r_bar = self.ema(self.r_bar, r)
        Δℓ = (self.prev_ℓ_bar - self.ℓ_bar) if self.prev_ℓ_bar is not None else 0.0
        Δr = (self.r_bar - self.prev_r_bar) if self.prev_r_bar is not None else 0.0

        # LR adaptation (trend + variance damping)
        if Δℓ > 1e-5:
            fα = self.cfg["u"]
        else:
            fα = self.cfg["d"]/(1+self.cfg["λ"]*var)
        self.α = float(torch.clamp(torch.tensor(self.α * fα),
                                   self.cfg["α_min"], self.cfg["α_max"]))
        # Momentum adaptation
        if Δr > 0:
            self.μ = min(self.μ + self.cfg["δμ"], self.cfg["μ_max"])
        else:
            self.μ = max(self.μ - self.cfg["κμ"]*self.μ, self.cfg["μ_min"])
        # σ adaptation (simple: keep fixed or shrink on variance)
        self.σ = float(torch.clamp(torch.tensor(self.σ/(1+0.5*var)),
                                   self.cfg["σ_min"], self.cfg["σ_max"]))

        self.prev_ℓ_bar, self.prev_r_bar = self.ℓ_bar, self.r_bar
        return self.α, self.μ, self.σ


# ------------------------------------------------------------
# 4. AE-DDSP Loop (training step)
# ------------------------------------------------------------
def ae_ddsp_step(synth, feats, scheduler, target_audio, optimizer):
    # Forward synth
    audio = synth()  # [1, T]

    # Features
    S_list = feats(audio)
    S_ref_list = feats(target_audio)

    # Loss = multi-res log-mel L2
    loss = 0.0
    for S, S_ref in zip(S_list, S_ref_list):
        loss += ((S - S_ref)**2).mean()
    loss /= len(S_list)

    # Example reward: spectral centroid closeness
    centroid, rolloff, flatness = spectral_stats(S_list[0])
    centroid_ref, rolloff_ref, flatness_ref = spectral_stats(S_ref_list[0])
    reward = torch.exp(-((centroid-centroid_ref)**2)/0.05)

    # Update scheduler (with variance=0 for now)
    α, μ, σ = scheduler.update(loss.item(), reward.item(), var=0.0)

    # Apply optimizer with adaptive hyperparams
    for g in optimizer.param_groups:
        g["lr"] = α
        g["momentum"] = μ if "momentum" in g else 0.0

    optimizer.zero_grad()
    loss.backward()
    
    # Plant update with dither (add noise to gradients)
    with torch.no_grad():
        for p in synth.parameters():
            if p.grad is not None:
                p.grad += σ * torch.randn_like(p.grad)
    optimizer.step()

    return loss.item(), reward.item(), α, μ, σ
```

---

# 🔑 How to Use

```python
# Initialize modules
synth = DifferentiableSynth()
feats = SpectralFeatures()
scheduler = Scheduler()
optimizer = torch.optim.SGD(synth.parameters(), lr=1e-3, momentum=0.9)

# Dummy target (e.g. sine wave at 220Hz)
target = synth()  # or load real audio

# Training loop
for step in range(1000):
    loss, reward, α, μ, σ = ae_ddsp_step(synth, feats, scheduler, target, optimizer)
    if step % 50 == 0:
        print(f"step {step}: loss={loss:.4f}, reward={reward:.3f}, α={α:.5f}, μ={μ:.3f}, σ={σ:.6f}")
```

---

✅ This scaffold matches the theory:

* **DSP plant** → parameters are trainable, forward returns audio.
* **Feature encoder** → multi-res STFT/log-mel & spectral stats.
* **Scheduler** → trend/variance/reward-based updates for LR, momentum, dither.
* **Loop** → gradient + adaptive update + stochastic dither injection.

### PSUEDOCODE

Perfect — let’s extend the stub into a **real differentiable subtractive synth** inside the AE-DDSP loop.
We’ll keep it compact but functional:

* **Oscillator** (sine/saw/square mix)
* **ADSR envelope** (differentiable exponential curves)
* **Low-pass filter** (biquad, differentiable)
* **VCA** (amplitude control)

This gives us a classic subtractive signal path:

`Oscillator → Filter → VCA (with envelope)`

---

# 🔧 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------------------------------------------------
# Utility: simple differentiable biquad low-pass filter
# ------------------------------------------------------------
def biquad_lowpass(x, cutoff, Q, sr=16000):
    """
    x: [B, T]
    cutoff: scalar tensor (Hz)
    Q: scalar tensor
    """
    # normalize
    omega = 2 * math.pi * cutoff / sr
    alpha = torch.sin(omega) / (2 * Q)

    b0 = (1 - torch.cos(omega)) / 2
    b1 = 1 - torch.cos(omega)
    b2 = (1 - torch.cos(omega)) / 2
    a0 = 1 + alpha
    a1 = -2 * torch.cos(omega)
    a2 = 1 - alpha

    b0, b1, b2, a0, a1, a2 = [torch.tensor(v, device=x.device, dtype=x.dtype) for v in (b0, b1, b2, a0, a1, a2)]

    y = torch.zeros_like(x)
    x1 = x2 = y1 = y2 = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    for n in range(x.size(1)):
        xn = x[:, n]
        yn = (b0/a0)*xn + (b1/a0)*x1 + (b2/a0)*x2 - (a1/a0)*y1 - (a2/a0)*y2
        y[:, n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn
    return y


# ------------------------------------------------------------
# Differentiable Oscillator
# ------------------------------------------------------------
def oscillator(t, freq, mix_saw=0.5, mix_square=0.0):
    """
    t: time vector [T]
    freq: tensor scalar
    """
    phase = 2 * math.pi * freq * t
    sine = torch.sin(phase)
    saw = 2 * (phase/(2*math.pi) % 1.0) - 1.0
    square = torch.sign(sine)
    wave = (1 - mix_saw - mix_square) * sine + mix_saw * saw + mix_square * square
    return wave


# ------------------------------------------------------------
# ADSR Envelope (differentiable exponential shapes)
# ------------------------------------------------------------
def adsr_envelope(t, attack, decay, sustain, release, note_len=0.5):
    env = torch.zeros_like(t)
    # Attack
    attack_mask = (t < attack)
    env[attack_mask] = t[attack_mask]/attack
    # Decay
    decay_mask = (t >= attack) & (t < attack+decay)
    env[decay_mask] = sustain + (1-sustain)*torch.exp(-(t[decay_mask]-attack)/decay)
    # Sustain
    sustain_mask = (t >= attack+decay) & (t < note_len)
    env[sustain_mask] = sustain
    # Release
    release_mask = (t >= note_len)
    env[release_mask] = sustain*torch.exp(-(t[release_mask]-note_len)/release)
    return env


# ------------------------------------------------------------
# Subtractive Synth (Oscillator → Filter → VCA)
# ------------------------------------------------------------
class SubtractiveSynth(nn.Module):
    def __init__(self, n_samples=16000, sr=16000):
        super().__init__()
        self.sr = sr
        self.n_samples = n_samples

        # Trainable parameters
        self.freq = nn.Parameter(torch.tensor([220.0]))
        self.mix_saw = nn.Parameter(torch.tensor([0.3]))
        self.mix_square = nn.Parameter(torch.tensor([0.0]))

        self.cutoff = nn.Parameter(torch.tensor([2000.0]))
        self.res = nn.Parameter(torch.tensor([0.7]))

        self.attack = nn.Parameter(torch.tensor([0.05]))
        self.decay = nn.Parameter(torch.tensor([0.1]))
        self.sustain = nn.Parameter(torch.tensor([0.7]))
        self.release = nn.Parameter(torch.tensor([0.2]))

        self.amp = nn.Parameter(torch.tensor([0.8]))

    def forward(self):
        device = self.freq.device
        t = torch.linspace(0, self.n_samples/self.sr, self.n_samples, device=device)

        # Oscillator
        wave = oscillator(t, self.freq, self.mix_saw, self.mix_square).unsqueeze(0)  # [1, T]

        # Envelope
        env = adsr_envelope(t, self.attack, self.decay, self.sustain, self.release).unsqueeze(0)

        # Multiply
        sig = wave * env * self.amp

        # Filter
        y = biquad_lowpass(sig, self.cutoff, self.res, sr=self.sr)

        return y
```

---

# 🔑 How to Plug into AE-DDSP

Replace the old `DifferentiableSynth` with `SubtractiveSynth` in the scaffold we built earlier:

```python
synth = SubtractiveSynth()
feats = SpectralFeatures()
scheduler = Scheduler()
optimizer = torch.optim.SGD(synth.parameters(), lr=1e-3, momentum=0.9)

target = synth()  # or load a target waveform

for step in range(500):
    loss, reward, α, μ, σ = ae_ddsp_step(synth, feats, scheduler, target, optimizer)
    if step % 50 == 0:
        print(f"{step}: loss={loss:.4f}, reward={reward:.3f}, α={α:.5f}, μ={μ:.3f}, σ={σ:.6f}")
```

---

# ⚡ Notes

* **Oscillator** supports blends of sine, saw, square.
* **Envelope** is piecewise, continuous, differentiable almost everywhere.
* **Filter** uses a simple biquad IIR. For efficiency you could swap to a differentiable FIR convolution or torchdsp.
* **All parameters are trainable tensors** — the AE-DDSP scheduler can adapt them via gradient + evolutionary dither.

### PSEUDOCODE

Got it 🚀 — let’s design a **differentiable 6-operator FM synth** in PyTorch, inspired by the Yamaha DX7.
We’ll keep it structured but still manageable for training inside your **AE-DDSP loop**.

---

# 🎛 Theory: FM Synth with 6 Operators

* **Operator** = sine oscillator with frequency ratio, detune, and envelope.
* Operators can be routed in **algorithms**: some modulate the frequency of others, some feed to the output mix.
* The DX7 has 32 “algorithms” (routing diagrams). We’ll allow a flexible adjacency matrix to represent which operator modulates which.

---

# 🧩 Implementation in PyTorch

```python
import torch
import torch.nn as nn
import math


# ------------------------------------------------------------
# ADSR envelope (smooth, differentiable)
# ------------------------------------------------------------
def adsr(t, attack, decay, sustain, release, note_len=0.5):
    env = torch.zeros_like(t)
    # Attack
    a_mask = t < attack
    env[a_mask] = t[a_mask] / (attack + 1e-8)
    # Decay
    d_mask = (t >= attack) & (t < attack + decay)
    env[d_mask] = sustain + (1 - sustain) * torch.exp(-(t[d_mask] - attack) / (decay + 1e-8))
    # Sustain
    s_mask = (t >= attack + decay) & (t < note_len)
    env[s_mask] = sustain
    # Release
    r_mask = t >= note_len
    env[r_mask] = sustain * torch.exp(-(t[r_mask] - note_len) / (release + 1e-8))
    return env


# ------------------------------------------------------------
# Operator = sine oscillator + envelope
# ------------------------------------------------------------
class FMOperator(nn.Module):
    def __init__(self, sr=16000, n_samples=16000):
        super().__init__()
        self.sr = sr
        self.n_samples = n_samples

        # Trainable parameters
        self.freq_ratio = nn.Parameter(torch.tensor([1.0]))   # ratio to base freq
        self.detune = nn.Parameter(torch.tensor([0.0]))       # Hz
        self.output_gain = nn.Parameter(torch.tensor([0.5]))

        # Envelope params
        self.attack = nn.Parameter(torch.tensor([0.05]))
        self.decay = nn.Parameter(torch.tensor([0.1]))
        self.sustain = nn.Parameter(torch.tensor([0.7]))
        self.release = nn.Parameter(torch.tensor([0.2]))

    def forward(self, t, base_freq, modulation):
        """
        t: [T] time vector
        base_freq: scalar tensor (Hz)
        modulation: [B, T] modulation signal (phase offsets)
        """
        freq = base_freq * self.freq_ratio + self.detune
        phase = 2 * math.pi * freq * t
        env = adsr(t, self.attack, self.decay, self.sustain, self.release)
        sig = torch.sin(phase + modulation) * env * self.output_gain
        return sig


# ------------------------------------------------------------
# 6-Operator FM Synth
# ------------------------------------------------------------
class FMSynth6(nn.Module):
    def __init__(self, sr=16000, n_samples=16000, algorithm=None):
        super().__init__()
        self.sr = sr
        self.n_samples = n_samples

        # 6 operators
        self.ops = nn.ModuleList([FMOperator(sr, n_samples) for _ in range(6)])

        # Algorithm routing matrix: [6, 6], entry (i,j)=1 means op j modulates op i
        if algorithm is None:
            # Simple default: op6→op5→op4→op3→op2→op1 (serial), op1 is output
            mat = torch.zeros(6, 6)
            for i in range(5):
                mat[i, i+1] = 1
            self.algorithm = nn.Parameter(mat, requires_grad=False)
        else:
            self.algorithm = nn.Parameter(algorithm, requires_grad=False)

        # Trainable mix weights for final output
        self.mix = nn.Parameter(torch.ones(6))

    def forward(self, base_freq=220.0, note_len=0.5):
        device = self.mix.device
        t = torch.linspace(0, self.n_samples/self.sr, self.n_samples, device=device)

        # storage for signals
        sigs = [None]*6

        # Compute operators in order
        for i in reversed(range(6)):
            # modulation = sum of outputs of modulators (according to algorithm)
            mod_sources = []
            for j in range(6):
                if self.algorithm[i,j] == 1:
                    if sigs[j] is None:
                        # placeholder, will be filled once j is computed
                        continue
                    mod_sources.append(sigs[j])
            modulation = sum(mod_sources) if mod_sources else torch.zeros_like(t)
            sigs[i] = self.ops[i](t, torch.tensor(base_freq, device=device), modulation)

        # Mix output from all ops with trainable weights
        out = sum(w*s for w, s in zip(self.mix, sigs))
        return out.unsqueeze(0)  # [1, T]
```

---

# ⚡ Key Features

* **6 operators** (each with ratio, detune, envelope, gain).
* **Routing matrix** $[6,6]$ encodes algorithm (who modulates who).

  * Example given: serial chain (6→5→4→3→2→1).
  * You can set other algorithms by filling the matrix.
* **Differentiable ADSR envelopes** scale operator amplitudes.
* **Final output mix**: weighted sum of all operators.

---

# 🔑 How to use inside AE-DDSP

```python
synth = FMSynth6()
feats = SpectralFeatures()
scheduler = Scheduler()
optimizer = torch.optim.SGD(synth.parameters(), lr=1e-3, momentum=0.9)

target = synth(base_freq=220.0)  # or load a real target audio

for step in range(200):
    loss, reward, α, μ, σ = ae_ddsp_step(synth, feats, scheduler, target, optimizer)
    if step % 50 == 0:
        print(f"{step}: loss={loss:.4f}, reward={reward:.3f}, α={α:.5f}, μ={μ:.3f}, σ={σ:.6f}")
```

---

✅ You now have a **6-operator differentiable FM synth** with routing flexibility. This can learn timbres, or be driven by spectral feature rewards in the AE-DDSP loop.

