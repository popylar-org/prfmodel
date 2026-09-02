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

The propery Gaussian density has the advantage that it decouples amplitude parameters from pRF size/tuning width
parameters $\sigma$, making it easier to interpret (although amplitudes are often treated as nuisance parameters). It
also leads to smaller amplitude estimates and stimulus-encoded model responses. However, it does **not** change the
identifiability of the model parameters.

Using the proper Gaussian density implies that parameter estimates for amplitudes
cannot be directly compared to the estimates from software that uses the unnormalized density. This does not only hold
for the Gaussian 1D and 2D pRF models but also for Difference of Gaussian and Gaussian Divisive Normalization pRF
models as well as Gaussian Connective Field models.

It is possible to convert the amplitudes estimated with the proper density into those estimated with unnormalized
density by dividing by the volume:
\begin{equation}
\beta_\text{unnorm} = \beta_\text{norm} / V.
\end{equation}
Importantly, this conversion assumes that the models for which the amplitudes have been estimated are otherwise equal.

Note that you can implement your own Gaussian models in prfmodel that use unnormalized densities
(see the tutorial on [custom models](tutorials/tutorials/custom_models.md)).
