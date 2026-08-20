# Architecture

This page contains information about the package architecture.

prfmodel contains different submodules that contain different types of (sub-) models.

The main model classes that users interact with are in {py:mod}`prfmodel.models`, {py:mod}`prfmodel.impulse`,
and {py:mod}`prfmodel.scaling`. We distinguish between three user profiles:

- **Entry-level** users will interact with high-level canonical model classes such as
    {py:class}`prfmodel.models.prf.Gaussian2DPRFModel`. These classes have good defaults and require very little
    adjustment.
- **Advanced** users will interact with submodel classes in {py:mod}`prfmodel.impulse` and {py:mod}`prfmodel.scaling`.
    For example, they might customize the parameters of impulse submodel in a canonical
    {py:class}`prfmodel.models.prf.Gaussian2DPRFModel`.
- **Expert** uses will interact with all available public classes, potentially defining their own custom
    submodels or canonical models.

## Inhertitance diagrams

To facilitate the development of the package and the creation of custom models, we provide an overview of the
inhertiance structure of the three modules containing model classes.

The inheritance diagram for {py:mod}`prfmodel.models`:

```{eval-rst}
.. inheritance-diagram:: prfmodel.models.base prfmodel.models.prf prfmodel.models.cf prfmodel.models.compression
   :parts: 1
   :include-subclasses:
   :top-classes: prfmodel.utils.ModelProtocol
```

The inheritance diagram for {py:mod}`prfmodel.impulse`:

```{eval-rst}
.. inheritance-diagram:: prfmodel.impulse.base
   :parts: 1
   :include-subclasses:
   :top-classes: prfmodel.utils.ModelProtocol
```

The inheritance diagram for {py:mod}`prfmodel.scaling`:

```{eval-rst}
.. inheritance-diagram:: prfmodel.scaling.base
   :parts: 1
   :include-subclasses:
   :top-classes: prfmodel.utils.ModelProtocol
```

## Protocols and base classes

All (sub-) model classes inherit from {py:class}`prfmodel.utils.ModelProtocol`. This protocol requires subclasses to
implement methods for accessing and checking parameter names.

The modulse {py:mod}`prfmodel.models`, {py:mod}`prfmodel.impulse`, {py:mod}`prfmodel.scaling`, and
{py:mod}`prfmodel.regressors` define abstract base classes (ABCs) that all subsequent models inherit from. These ABCs
define abstract methods and attributes that subclasses must implement. For making model predictions, they use a
"public facade" pattern: They implement a concrete user-facing `__call__` method that accepts NumPy arrays and pandas
dataframe objects as arguments and performs input checks on these objects (e.g., by calling methods inherited from
{py:class}`prfmodel.utils.ModelProtocol`). To perform actual computations, `__call__` converts all arguments to backend
specific tensor objects (i.e., {py:data}`prfmodel.typing.Tensor`) and forwards them to an abstract `call` method that
each subclass must implement. Importantly, `call` must only use tensors as inputs and outputs and implement
tensor operations via {py:mod}`keras.ops` or backend-specific operations (e.g., {py:mod}`tf.math` or
{py:mod}`jax.numpy`).

The main reason behind this split is that Keras can only track gradients for tensors. Therefore, fitter classes that
require gradients, such as {py:class}`prfmodel.fitters.SGDFitter`, interally only call each models `call` method.
However, because working with tensors is not very user-friendly (e.g., for data wrangling and plotting), users can
instead call the less restrictive `__call__` method (e.g., to make a model prediction).

Another requirement is that `call` methods must be traceable to enable backend-specific compilation (see [](backends.md)).
This also requires inputs and outputs to be tensor objects, but also that the control flow inside `call` does not
depend on the **values** of input arguments (but depending on input shapes is allowed).

For an example, see [](../tutorials/tutorials/custom_models.md).

## API

The API also uses a "public facade" design pattern where most complex model classes are defined in private submodules
and exported as public in the next ancestor module in the hierarchy. This prevents the API docs from overflowing with
submodules while keeping class definitions in separate files.
