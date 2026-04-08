# ANN Contest Workthrough Explanation

## Project Overview

This notebook is a full from-scratch implementation of a multi-layer neural network using only NumPy. The work was completed challenge-by-challenge to match the contest rules and rubric: architecture analysis, forward pass, backward pass, optimization, and bonus residual learning.

## What We Implemented

### Challenge 0: One-Hot Encoding and Tensor Discipline

Implemented a fully vectorized `one_hot(y, num_classes)`:

- Input labels shape: `(N,)`
- Output matrix shape: `(N, C)`
- No Python loop over samples

Key point: one-hot targets are required for softmax + cross-entropy and strict shape consistency prevents forward/backward dimension errors.

### Challenge 1: Architecture Forensics

Implemented:

- `dense_param_count(n_in, n_out, use_bias=True)`
- `architecture_report(layer_dims, use_bias=True)`
- `rank_feasible_architectures(candidates, budget)`

What this solved:

- Layer-by-layer parameter counting
- Cumulative and total parameter reporting
- Budget-constrained architecture ranking (descending by params, alphabetical tie-break)

### Challenge 2: Stable Forward Pass

Implemented:

- `init_mlp(layer_dims, seed=7, scheme="he")`
- `linear_forward(A_prev, W, b)`
- `relu(Z)`
- `tanh_act(Z)`
- `stable_softmax(Z)`
- `cross_entropy_from_probs(P, y_onehot)`
- `forward_mlp(X, params, hidden_activation="relu")`

Important details followed:

- Correct tensor shapes through all layers
- Numerically stable softmax via row-wise max subtraction
- Proper cache construction for backpropagation

### Challenge 3: Manual Backpropagation + Gradient Checking

Implemented:

- `relu_backward(dA, Z)`
- `tanh_backward(dA, Z)`
- `linear_backward(dZ, A_prev, W)`
- `backward_mlp(params, cache, y_onehot, hidden_activation="relu", l2_lambda=0.0)`

What was validated:

- Correct gradient keys and tensor shapes
- Correct inclusion of L2 regularization in weight gradients
- Numerical gradient check agreement (very low relative errors)

### Challenge 4: Optimization and Training Loop

Implemented:

- `iterate_minibatches(X, y, batch_size, shuffle=True, seed=7)`
- `sgd_momentum_step(params, grads, velocity, lr, momentum=0.9)`
- `train_classifier(...)`

Training pipeline includes:

- Mini-batch training
- Momentum updates
- Per-epoch tracking of `train_loss`, `val_loss`, `train_acc`, `val_acc`
- XOR target performance requirements from public tests

### Challenge 5 (Bonus): Residual Block

Implemented bonus functions:

- `init_residual_block(width, seed=7, scheme="he")`
- `residual_block_forward(A, block_params, activation="relu")`
- `residual_block_backward(dA_out, cache, block_params, activation="relu")`

Also added a bonus experiment:

- Plain deep MLP vs residual classifier comparison on spiral dataset
- Shared plotting of validation accuracy curves for both models

## Written Explanations Added

Separate markdown responses were added for:

- Challenge 0
- Challenge 1
- Challenge 2
- Challenge 3
- Challenge 4
- Bonus interpretation
- Final reflection

These responses explain both conceptual understanding and implementation choices (numerical stability, vectorization, gradient behavior, optimization interpretation).

## Validation and Checks Performed

Public checks were run end-to-end in dependency order:

1. Environment/imports and helper functions
2. Challenge 0 test cell
3. Challenge 1 test cell
4. Challenge 2 test cell
5. Challenge 3 gradient-check test cell
6. Challenge 4 XOR training test cell
7. Curve plotting cells
8. Bonus experiment execution

Observed outcomes:

- All required public tests passed
- Gradient checking produced very low relative errors
- XOR training reached very high accuracy (meeting thresholds)
- Bonus spiral comparison executed successfully and was documented

## What We Went Through (Process Summary)

1. Filled all TODO blocks first (to establish complete functionality)
2. Added all required written explanation cells (to satisfy rubric requirements)
3. Executed tests sequentially to isolate and catch any failure early
4. Verified numerical correctness with gradient checking
5. Verified model behavior with training metrics and plots
6. Ran bonus architecture comparison and corrected interpretation text to match actual measured results

## Final Note

This notebook now represents a complete from-scratch ANN implementation and explanation set aligned with the contest instructions and grading rubric, with both code correctness checks and conceptual write-ups included.
