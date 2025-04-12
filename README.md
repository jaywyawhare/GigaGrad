# GigaGrad

**GigaGrad** is a minimal automatic differentiation (autograd) engine and neural network framework written in pure **C**. 
It is being developed to explore the **viability of implementing autograd** in C. It serves as a testbed for low-level automatic differentiation, and may evolve into or influence future iterations of [C-ML](https://github.com/jaywyawhare/c-ml). 
> ⚡ _Giga performance, micro simplicity — in C._

## Roadmap

- [x] Scalar Value type with gradient tracking
- [x] Support for basic ops: +, *, -, /, **, ReLU
- [ ] Reverse-mode autodiff (backward pass)
- [ ] Computation graph (DAG traversal)
- [ ] Minimal test example (e.g., single neuron or MLP)

## Installation

To install and build GigaGrad locally:

### 1. Clone the Repository

```bash
git clone git@github.com:jaywyawhare/GigaGrad.git
```

### 2. Build the Project

```bash
cd GigaGrad
make
```

### 3. Run the Example

```bash
./bin/gigagrad
```

This will execute the minimal example binary once compiled.

## Inspiration

- [micrograd](https://github.com/karpathy/micrograd) — scalar autograd engine in Python
- [tinygrad](https://github.com/geohot/tinygrad) — minimalist DL framework with GPU support