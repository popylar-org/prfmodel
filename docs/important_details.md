# Important details

When developing prfmodel, we made certain decisions for the design and implementation of different features.
While the decisions concerning the development of prfmodel are covered in [Development](development/index.rst) (e.g., architecture),
we describe the choices that affect **users** directly in this section.

## Gaussian models use proper probability densities

Some software packages that implement Gaussian population receptive field (pRF) models use unnormalized Gaussian
densities to predict tuning profiles. That is, they use the density:
\begin{equation}
f(x) = e^{-\frac{\lVert x - \mu \rVert^2}{2 \sigma^2}},
\end{equation}
that has a peak amplitude of $\max(f(x)) = 1$. Although this parameterization has advantages in some situations,
we decided that all Gaussian models in prfmodel use proper densities that are normalized by their
volume (see {py:func}`~prfmodel.density.normal_density`):
\begin{equation}
f(x) = \frac{1}{V} e^{-\frac{\lVert x - \mu \rVert^2}{2 \sigma^2}},
\end{equation}
where $V = (2 \pi \sigma^2)^{k / 2}$ is the volume and $k$ is the number of dimensions of the tuning profile.
The proper density has a peak amplitude of $\max(f(x)) = \frac{1}{V}$.

The proper Gaussian density has the advantage that it decouples amplitude parameters from pRF size/tuning width
parameters $\sigma$, making it easier to interpret (although amplitudes are often treated as nuisance parameters). It
also leads to smaller amplitude estimates and stimulus-encoded model responses. However, it does **not** change the
identifiability of the model parameters.

Using the proper Gaussian density implies that parameter estimates for amplitudes
cannot be directly compared to the estimates from software that uses the unnormalized density. This does not only hold
for the Gaussian 1D and 2D pRF models but also for difference of Gaussian and Gaussian divisive normalization pRF
models as well as Gaussian connective field models.

It is possible to convert the amplitudes estimated with the proper density into those estimated with unnormalized
density by dividing by the volume:
\begin{equation}
\beta_\text{unnorm} = \beta_\text{norm} / V.
\end{equation}
Importantly, this conversion assumes that the models for which the amplitudes have been estimated are otherwise equal.

Note that you can implement your own Gaussian models in prfmodel that use unnormalized densities
(see the tutorial on [custom models](tutorials/tutorials/custom_models.md)).

## Spatial receptive fields are not normalized

Some software packages normalize stimulus-encoded model responses of spatial models by the
size of the cells in the spatial grid. For example, for the Gaussian 2D pRF model in visual space, the stimulus-encoded
response can be normalized as follows:
\begin{equation}
r(t) = \sum_{xy} g(x, y) \cdot S(t, x, y) dA,
\end{equation}
where $g(x, y)$ is the Gaussian pRF tuning profile, $S(t, x, y)$ is the stimulus design, and  $dA = dx dy$ is the
size of each cell in the grid. This normalization changes
the scale and the interpretation of amplitude parameters, making them comparable across spatial grid resolutions.

An alternative convention adopted by some software packages is to normalize spatial RFs (called tuning
profiles in prfmodel) by their sum [^1]. This makes the conflict between volume-normalized vs -unnormalized densities
irrelevant and amplitudes comparable across grid cell sizes. Instead it ties amplitudes to the sizes of the spatial
grid dimensions (e.g., width and height)[^2].

However, normalization also leads to numerical issues when RFs are not covered by the
stimulus grid because then their sum is zero. Moreover, because some spatial models have irregular-spaced
one-hot-encoded grids (e.g., the Gaussian 1D pRF model in log numerosity space) and non-spatial models
(e.g., the Gaussian connective field model) do not have an equivalent
grid cell size or meaningful normalizations, we decided **against** normalizing spatial models in prfmodel.
We also do not want to tie our model implementations to the spatial domain.

This decision means that amplitude parameters are not comparable between different spatial grid resolutions (e.g.,
upsampling a $128^2$ to a $256^2$ grid while keeping the overall width and height would amplify amplitude estimates
by 4). However, for regular-spaced grids, it is possible to divide amplitudes by the grid cell size to make them
comparable cross grid resolutions. The normalization does **not** affect the identifiability of model parameters.

## Impulse responses are normalized when they describe measurements

The predicted responses of some impulse models (see {py:mod}`prfmodel.impulse`) are normalized in prfmodel
(sum-normalized by default, but other functions are possible [^3]). This is
because they are used to describe the typical shape of the measurement of a neural response (e.g., the BOLD
response in fMRI). These impulse responses are convolved with the response of a model that describes the behavior of a
neuron population (e.g., a stimulus-encoded pRF response). Here, the sum-normalization decouples amplitude parameters
from impulse response parameters (e.g., the shape of the gamma distribution), but it does **not** affect the
identifiability of model parameters.

It is possible to convert impulse-sum-normalized into impulse-unnormalized amplitudes:
\begin{equation}
\beta_\text{unnorm} = \beta_\text{norm} / \sum_t h_\text{unnorm}(t),
\end{equation}
where $h(t)$ is the unnormalized impulse response.

Some impulse models do not use any normalization by default because they are also used to describe neuron
population behavior. For example, the compressive spatio-temporal pRF models uses transient and
sustained impulse models to describe temporal neuron activation patterns.

## What if I want to deviate from these decisions?

If you have good reasons to deviate from our decisions, you can implement your own models in prfmodel that use
different conventions (see the tutorial on [custom models](tutorials/tutorials/custom_models.md)).

[^1]: Alternative spatial normalization functions (e.g., L2 norm) are possible but bring similar and sometimes even
more problems.

[^2]: It is also possible to normalize the stimulus-encoded response which couples amplitudes to the stimulus
design instead of the grid. This comes with similar problems as normalizing the RF.

[^3]: Normalizing by the L2 norm is actually numerically more stable (because it is never zero for signed impulse
responses) but it also couples amplitudes to impulse model parameters.
