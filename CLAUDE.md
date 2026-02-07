# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project: **Discrete Normalizing Flows as Variational Ansatze for Classical Statistical Mechanics**. Proposes combining discrete normalizing flows with autoregressive variational models to study the Ising model while preserving exact, tractable log-probabilities and rigorous variational free-energy bounds.

Currently in the theoretical/proposal stage — the `doc/` directory contains the LaTeX paper; no implementation code exists yet.

## Core Theoretical Framework

The approach has three components that will become the main code modules when implemented:

1. **Base distribution** (`p_θ(z)`): Autoregressive model (MADE/PixelCNN) over latent binary spins `z ∈ {±1}^N`
2. **Discrete flow** (`f_φ`): Bijective map `{±1}^N → {±1}^N` via coupling layers with learned checkerboard-partition mask networks (small ConvNets, 2-3 layers, 3×3 kernels)
3. **Physical distribution**: `q(σ) = p_θ(f_φ⁻¹(σ))` — exact log-prob via change of variables (no Jacobian correction needed for discrete bijectionsw)

Key design constraint: all transformations must be **bijective on the discrete configuration space** to maintain the exact free-energy bound. The coupling layers use XOR operations in F₂.

## Planned Implementation Details (from paper Sec. 8)

- System sizes: 8×8, 16×16, 32×32 square Ising lattices
- Flow depth: 4-8 coupling layers with alternating checkerboard partitions
- Training: joint optimization of θ (REINFORCE with baseline) and φ (straight-through estimator / Gumbel-softmax)
- Validation targets: exact transfer-matrix results (small systems), Monte Carlo (larger systems)

