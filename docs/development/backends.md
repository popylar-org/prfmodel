# Backends

This page contains information about the Keras backends.

prfmodel is built on Keras 3.0 to perform GPU-accelerated operations and gradient tracking. Keras provides a unified
API for several backends that perform the actual computations. prfmodel currently supports three backends: TensorFlow,
PyTorch, and JAX. At least one backend must be installed alongside prfmodel to use the package. Some tasks cannot
be solved by solely relying on Keras and backend-specific implementations are needed. These live in the private
{py:mod}`prfmodel._backend` module are mainly related to stochastic gradient descent fitting and compilation.
Importantly, only the backend selected by the user is imported in the package.

The three backends differ in how they optimize operations:

- TensorFlow uses graph-based execution by wrapping functions inside `tf.function`.
- PyTorch uses compilation through `torch.compile`.
- JAX uses just-in-time compilation via `jax.jit`.

These optimizations can yield substantial increases in speed. However, they also require optimized functions to be
**tracable**. This means that their inputs and outputs must be tensor objects and that the function logic does not
depend on concrete values of the inputs (i.e., no if-else branching on input values).

In prfmodel, the {py:mod}`prfmodel._backend._compile` module exports a backend-specific `compile_fun` that enables
optimization via the above-mentioned functions. Both {py:class}`prfmodel.fitters.GridFitter` and
{py:class}`prfmodel.fitters.SGDFitter` internally use this `compile_fun` to speed up computations if their
`compile_step` flag is enabled. For the optimization to work, model classes must follow a specific implementation
logic that is explained in [](architecture.md).

Additional backend-specific functions are imported via {py:mod}`prfmodel._backend._external`.
