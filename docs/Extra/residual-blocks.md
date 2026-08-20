# Deep Dive: The Residual Block in ChessZero

> **Document Type:** Architectural Deep Dive & Extra Reference  
> **Related Chapters:** [Chapter 10 — The Neural Network](../10-the-neural-network.md) · [Chapter 4 — Deep RL & Function Approximation](../04-deep-rl-and-function-approximation.md) · [Chapter 17 — Scaling & Improvements](../17-scaling-and-improvements.md)  
> **Source Code:** [`src/chesszero/network.py`](../../src/chesszero/network.py)

---

## 1. Overview & Motivation

In **ChessZero**, the neural network $f_\theta(s) = (\mathbf{p}, v)$ serves as the guiding heuristic for Monte Carlo Tree Search (MCTS). Given an encoded chess board state $s \in \mathbb{R}^{19 \times 8 \times 8}$, the network must predict:
1. **Policy $\mathbf{p}$**: A distribution over $4{,}672$ candidate moves.
2. **Value $v$**: A scalar evaluation $v \in [-1, +1]$ of the board state from the perspective of the player to move.

The core backbone (or "tower") of `ChessNet` consists of a stack of **Residual Blocks** (`ResidualBlock`), an architectural paradigm first introduced by He et al. (2015) in *Deep Residual Learning for Image Recognition* and adopted by DeepMind in **AlphaGo Zero** (Silver et al., 2017) and **AlphaZero** (Silver et al., 2018).

```
                      ┌─────────────────────────────────────────┐
                      │             Encoded State               │
                      │             (B, 19, 8, 8)               │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │             Input Stem                  │
                      │ Conv 3x3 (19->128) + BN + ReLU          │
                      └────────────────────┬────────────────────┘
                                           │ (B, 128, 8, 8)
                                           ▼
                      ┌─────────────────────────────────────────┐
                   ┌─►│          Residual Block 1               │
                   │  └────────────────────┬────────────────────┘
                   │                       │ (B, 128, 8, 8)
                   │                       ▼
   Tower of        │  ┌─────────────────────────────────────────┐
   Residual Blocks │  │          Residual Block 2               │
   (e.g., N=6)     │  └────────────────────┬────────────────────┘
                   │                       │
                   │                      ...
                   │                       ▼
                   │  ┌─────────────────────────────────────────┐
                   └─►│          Residual Block N               │
                      └────────────────────┬────────────────────┘
                                           │ (B, 128, 8, 8)
                                           ├────────────────────────┐
                                           ▼                        ▼
                      ┌─────────────────────────┐ ┌─────────────────────────┐
                      │       Policy Head       │ │       Value Head        │
                      │ Conv 1x1 (128->2) + BN  │ │ Conv 1x1 (128->1) + BN  │
                      │ FC (128 -> 4672 logits) │ │ FC (64 -> 256 -> 1)     │
                      └─────────────────────────┘ └─────────────────────────┘
```

This document provides a thorough explanation of how the `ResidualBlock` operates mathematically and computationally, followed by an analysis of why this specific building block was chosen for chess evaluation.

---

## 2. Anatomy and Mechanics of `ResidualBlock`

### 2.1 The Code Implementation

In [`src/chesszero/network.py`](../../src/chesszero/network.py), the `ResidualBlock` is implemented as follows:

```python
class ResidualBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)
```

### 2.2 Step-by-Step Forward Pass

Let $x \in \mathbb{R}^{B \times C \times 8 \times 8}$ be the input tensor to the block, where $B$ is the batch size and $C = \text{filters}$ (e.g., $128$):

```
          Input x (B, C, 8, 8)
            │               │
            │ (identity)    │
            │               ▼
            │     ┌───────────────────┐
            │     │  Conv2d (3x3, P1) │  ──> (B, C, 8, 8)
            │     └─────────┬─────────┘
            │               ▼
            │     ┌───────────────────┐
            │     │   BatchNorm2d     │  ──> (B, C, 8, 8)
            │     └─────────┬─────────┘
            │               ▼
            │     ┌───────────────────┐
            │     │       ReLU        │  ──> (B, C, 8, 8)
            │     └─────────┬─────────┘
            │               ▼
            │     ┌───────────────────┐
            │     │  Conv2d (3x3, P1) │  ──> (B, C, 8, 8)
            │     └─────────┬─────────┘
            │               ▼
            │     ┌───────────────────┐
            │     │   BatchNorm2d     │  ──> (B, C, 8, 8)
            │     └─────────┬─────────┘
            │               │
            ▼               ▼
           [+] <────────────┘  (Element-wise Addition: F(x) + x)
            │
            ▼
     ┌─────────────┐
     │    ReLU     │
     └──────┬──────┘
            │
            ▼
      Output y (B, C, 8, 8)
```

1. **Identity Preservation (`residual = x`)**:
   The input tensor $x$ is stored as a reference before any transformations occur.
2. **First Convolution Layer (`conv1`)**:
   A 2D convolution with a $3 \times 3$ kernel, stride 1, and `padding=1`. Because `padding=1`, spatial dimensions remain strictly $8 \times 8$. `bias=False` is set because the succeeding Batch Normalization includes a learnable affine shift ($\beta$), rendering a convolutional bias redundant.
3. **First Batch Normalization (`bn1`)**:
   Normalizes feature activations per channel across the mini-batch:
   $$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta$$
   This stabilizes the distribution of activations, smoothing the optimization landscape and mitigating internal covariate shift.
4. **First Activation Function (`F.relu`)**:
   Applies the standard non-linear Rectified Linear Unit function $\text{ReLU}(z) = \max(0, z)$.
5. **Second Convolution Layer (`conv2`)**:
   A second $3 \times 3$ convolution with `padding=1` and `bias=False`, mapping $C \to C$ feature maps while preserving $8 \times 8$ geometry.
6. **Second Batch Normalization (`bn2`)**:
   Normalizes the unactivated feature maps produced by `conv2`. Crucially, **no ReLU is applied here yet**.
7. **Residual Skip Connection & Final Activation (`F.relu(x + residual)`)**:
   The transformation output $\mathcal{F}(x)$ is added element-wise to the original input $x$. The final $\text{ReLU}$ is applied to the combined sum:
   $$y = \text{ReLU}(\mathcal{F}(x) + x)$$

---

## 3. Mathematical Foundations: Why Residual Connections Work

### 3.1 The Residual Learning Formulation

In a standard (plain) convolutional feed-forward network, each stacked layer is tasked with directly learning an underlying mapping:
$$H(x)$$
As network depth increases, optimizing $H(x)$ directly becomes notoriously difficult due to vanishing gradients, exploding gradients, and the **degradation problem** (where deeper networks exhibit higher training errors than their shallower counterparts, despite possessing greater expressive capacity).

The core insight of ResNets is to reformulate the objective. Instead of fitting $H(x)$, we parameterize the stacked layers to learn a **residual mapping**:
$$\mathcal{F}(x) \triangleq H(x) - x$$
The original mapping is then recovered via:
$$H(x) = \mathcal{F}(x) + x$$

Why is this advantageous?
- **Identity as an easy default**: If an identity mapping is optimal (e.g., if additional depth does not benefit a particular feature extraction stage), driving the weights in $\mathcal{F}(x)$ toward zero ($\mathcal{F}(x) \to 0$) is drastically easier for gradient descent than learning an identity operator $H(x) = x$ using multi-layer nonlinear transformations.
- **Perturbative refinements**: The network is trained to compute incremental *modifications* or updates ($\Delta x$) to an existing representation rather than synthesizing a complete representation from scratch at every block.

### 3.2 Gradient Highway Formulation (Derivation)

Consider a chain of $L$ residual blocks without loss of generality. In the forward pass (ignoring non-linearities for clarity of gradient propagation):
$$x_{l+1} = x_l + \mathcal{F}(x_l, \mathcal{W}_l)$$
Recursively expanding this relation from block $l$ to a deeper block $L$ yields:
$$x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i)$$

During backpropagation, let $\mathcal{E}$ denote the scalar training loss. By the chain rule of calculus:
$$\frac{\partial \mathcal{E}}{\partial x_l} = \frac{\partial \mathcal{E}}{\partial x_L} \frac{\partial x_L}{\partial x_l} = \frac{\partial \mathcal{E}}{\partial x_L} \left( \mathbf{I} + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i) \right)$$

This derivation reveals the crucial property of residual networks:
$$\frac{\partial \mathcal{E}}{\partial x_l} = \underbrace{\frac{\partial \mathcal{E}}{\partial x_L}}_{\text{Direct Gradient Highway}} + \underbrace{\frac{\partial \mathcal{E}}{\partial x_L} \left( \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i) \right)}_{\text{Modulated Residual Gradients}}$$

The identity term $\mathbf{I}$ guarantees that the gradient $\frac{\partial \mathcal{E}}{\partial x_L}$ is propagated **directly back to any earlier layer $x_l$ without vanishing**, even if the weight-dependent term $\frac{\partial \mathcal{F}}{\partial x_l}$ becomes small. This direct gradient pathway prevents gradient attenuation across arbitrarily deep networks.

---

## 4. Why `ResidualBlock` Was Selected for Chess

The selection of the residual block architecture for ChessZero is not arbitrary. It addresses several domain-specific structural and algorithmic requirements of chess modeling under self-play reinforcement learning.

### 4.1 Receptive Field Growth and Global Board Awareness

A standard chess board is an $8 \times 8$ grid. Although tactics often occur locally, master-level chess evaluation demands global context:
- A bishop on `a1` can exert decisive control across the long diagonal over square `h8` ($7$ squares away).
- A rook on `a1` pins a queen on `a8` through an entire open file.
- King safety evaluations depend simultaneously on pawn shield integrity, opponent piece concentrations, and cross-board rook files.

A single $3 \times 3$ convolution has a local receptive field of only $3 \times 3$ squares (1 square in each direction). However, each `ResidualBlock` contains **two** sequential $3 \times 3$ convolutions. With stride 1 and padding 1:
- 1st Conv: expands receptive field by $2$ squares (radius $+1$).
- 2nd Conv: expands receptive field by another $2$ squares (radius $+1$).
- **Net Receptive Field expansion per `ResidualBlock`**: $+4$ squares in each dimension.

```
Layer                  Receptive Field (Side)    Receptive Field Area
──────────────────────────────────────────────────────────────────────
Input Stem (1 Conv)            3 × 3                    9 squares
Block 1 (2 Convs)              7 × 7                   49 squares
Block 2 (2 Convs)             11 × 11                  Full 8×8 board covered!
Block 3–6 (8 Convs)           15×15 to 27×27           Multi-path global context
```

By stacking 6 residual blocks (12 convolutions + 1 stem convolution = 13 convolutional layers), the effective receptive field exceeds $27 \times 27$. This ensures:
1. **Full Board Coverage**: Every square's representation integrates information from every other square on the board multiple times over.
2. **Multi-Hop Reasoning**: The network can synthesize complex multi-piece tactical interactions (e.g., Piece A attacks Piece B, which defends Piece C, which pins Piece D to King E).

### 4.2 Preservation of Spatial Invariance & Board Topology

Chess pieces operate within an explicit 2D spatial coordinate system. Translation invariance and locality are fundamental inductive biases:
- **Spatial Weight Sharing**: A knight fork pattern (`N` attacking two pieces separated by a 2:1 geometric jump) obeys identical spatial rules whether it happens on the kingside, queenside, or center. Convolutional filters learn this geometric kernel once and reuse it across all 64 squares.
- **Preserved Resolution**: Unlike image classification models (e.g., ImageNet ResNets) that use strided convolutions or pooling layers to downsample feature maps ($224 \to 112 \to 56 \dots$), `ChessNet` maintains a constant $8 \times 8$ spatial resolution throughout the entire residual tower (`padding=1`). Every intermediate feature channel represents a direct spatial embedding of the 64 board squares.

### 4.3 Incremental Board State Refinement

In game play, consecutive chess states are highly correlated. Moving a single piece modifies only $1$ or $2$ squares (or $4$ in castling), leaving the remaining $60+$ squares structurally unchanged.

Residual learning naturally models this property:
- Early layers in the stem extract fundamental piece positions and occupancy bitboards.
- Intermediate residual blocks apply incremental adjustments ($\Delta x$) to encode subtle features: pin relationships, king shelter weakness, pawn structure mobility, and passed pawn threats.
- If a block does not need to alter a specific feature channel for a given position, it simply propagates the representation forward via the identity connection.

### 4.4 Non-Stationary RL Dynamics and Training Stability

In AlphaZero-style self-play, the reinforcement learning target is non-stationary:
- The policy target $\boldsymbol{\pi}$ (MCTS visit distribution) and the value target $z$ (game outcomes) evolve as the agent improves.
- Early iterations produce noisy, high-variance targets; later iterations produce highly refined positional targets.

Standard deep feed-forward networks without skip connections suffer from catastrophic gradient instability and severe weight drift when exposed to shifting RL distributions. The combination of **Residual Connections** and **Batch Normalization** stabilizes training:
- BatchNorm maintains zero mean and unit variance per channel, preventing activation explosion across deep iterations.
- Skip connections ensure smooth parameter updates without vanishing gradients, allowing stable end-to-end convergence across thousands of self-play games.

### 4.5 High-Throughput Inference for MCTS Search

In AlphaZero, the neural network is evaluated repeatedly inside the MCTS simulation loop. For instance:
- If `num_simulations = 100`, every individual move requires $100$ network evaluations.
- A single 60-move game entails $\approx 6{,}000$ forward passes.

The 2D Convolutional ResNet architecture offers an optimal balance between expressive capacity and computational latency:
- Convolutions and element-wise additions map efficiently to hardware accelerators (GPUs, Apple Silicon Metal Performance Shaders, and CPU SIMD vector instructions).
- Compared to Vision Transformers (ViTs) or large self-attention networks, ResNets have lower inference overhead and predictable memory access patterns, maximizing the number of MCTS rollouts executable per second.

---

## 5. Architectural Comparison

To understand why `ResidualBlock` was selected over alternative paradigms, consider how other architectures perform on this task:

| Architecture | Spatial Prior | Gradient Flow at Depth | Inference Latency | Suitability for Chess RL |
| :--- | :--- | :--- | :--- | :--- |
| **Fully Connected (MLP)** | ❌ None (flattens 8×8 grid) | ⚠️ Degrades rapidly | Fast | ❌ Poor. Loses spatial topology; requires millions of weights per layer; cannot share weights across board regions. |
| **Plain CNN (No Skips / VGG)** | ✅ Preserved | ❌ Vanishing gradients beyond 5–7 layers | Fast | ❌ Poor for deep networks. Cannot scale to sufficient depth to capture long-range board interactions. |
| **Transformer / Attention** | ⚠️ Learned via positional encoding | ✅ Stable via residual layer norms | ⚠️ High per-leaf latency for small batch sizes | ⚠️ Viable for massive cluster scale, but computationally heavy for local training and low-latency MCTS leaf evaluations. |
| **Residual CNN (`ChessNet`)** | ✅ Preserved (strict 8×8 grid) | ✅ Direct gradient highway ($\mathbf{I} + \frac{\partial \mathcal{F}}{\partial x}$) | ⚡ Highly optimized (SIMD/Tensor Cores) | 🌟 **Optimal**. Combines strong spatial priors, deep multi-hop reasoning, stable RL training, and fast MCTS throughput. |

---

## 6. Summary

The `ResidualBlock` in ChessZero represents a principled architectural choice that brings together:
1. **Mathematical Robustness**: Unhindered gradient backpropagation through skip connections, enabling stable deep learning.
2. **Domain-Specific Inductive Bias**: Convolutional kernels that preserve 2D chessboard geometry and translate tactical patterns across squares.
3. **Global Spatial Receptive Field**: Rapid expansion of context to connect distant squares (e.g., long diagonals, open files, king attacks).
4. **RL Training Stability**: Resistance to non-stationary target shifts during iterative self-play.
5. **Computational Efficiency**: Fast evaluation latency essential for powering high-volume MCTS search.
