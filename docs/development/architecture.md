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

## API

The API uses a "public facade" design pattern where most complex model classes are defined in private submodules and
exported as public in the next ancestor module in the hierarchy. This prevents the API docs from overflowing with
submodules while keeping class definitions in separate files.
