# Developer documentation

This document contains practical information about how develop prfmodel. For more in-depth information about the package architecture and implementation choices, see the [Development](https://popylar-org.github.io/prfmodel/development/index.html) section in the online documentation.

## Cloning the repository

Before starting to develop prfmodel, you should clone the GitHub repository to create a local copy of the package that
you can modify. To clone the repository [git](https://git-scm.com/) must be installed in your system (install it first if you don't have it yet).

```shell
cd <where you keep your GitHub repositories>
git clone https://github.com/popylar-org/prfmodel.git
cd prfmodel
```

We recommend that you work on a different branch than `main`, so you should first create and switch to that branch. For example, if you want to create a new `bug-fix` branch, you should do:

```shell
git checkout -b bug-fix
```

Note that this will return an error if the `bug-fix` branch already exists. In case of an existing branch you can switch
with:

```shell
git checkout bug-fix
```

## Installing the package

For development, we recommend installing prfmodel with [uv](https://docs.astral.sh/uv/getting-started/installation/) (install it first if you don't have it yet). Then, you can install prfmodel and its dependencies in a virtual environment from `uv.lock` with:

```shell
uv sync --all-extras
```

This also installs all optional dependencies (i.e., Keras backends, development, and publishing utilities) and might
take a bit of time.

### Adding new dependencies

We also recommend adding new dependencies with [uv](https://docs.astral.sh/uv/concepts/projects/dependencies/). For example to add the `foo` package as a dependency:

```shell
uv add foo
```

Depending on whether your dependency is required for users at runtime or only during development, it should be added
to the main or optional dependencies.

## Testing

This package uses [pytest](https://docs.pytest.org/en/stable/) for testing. All tests should live in the [`tests/`](tests/) folder. Depending on whether a test
is a unit or integration tests, it should live in a corresponding subfolder.

We recommend running the existing tests with [uv](https://docs.astral.sh/uv/concepts/projects/run/):

```shell
uv run pytest
```

The package uses [GitHub action workflows](https://docs.github.com/en/actions) to automatically run tests on GitHub infrastructure against multiple Python versions. Existing workflows can be found in [`.github/workflows`](.github/workflows/)

## Validation

prfmodel is validated against external packages ([braincoder](https://github.com/Gilles86/braincoder/tree/main), [prfpy](https://github.com/VU-Cog-Sci/prfpy)). These validation checks live in the [`validation/`](validation/) folder and are also run through GitHub action workflows.

## Documentation

The general package documentation (e.g., installation guide, examples, tutorials) live in the [`docs/`](docs/) folder.
In this folder, documentation can be written in Markdown or [Restructured Text](https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html). The documentation is rendered with the Sphinx framework.

API documentation is created automatically using [AutoAPI](https://sphinx-autoapi.readthedocs.io/) through docstrings written directly in the package code. The package follows the [numpydoc](https://numpydoc.readthedocs.io/en/latest/) style for writing docstrings. Examples in docstrings can be automatically tested via:

```shell
uv run pytest --doctest-modules --ignore=src/prfmodel/_backend/ src/
```

This is also run as a GitHub actions workflow.

## Code quality

To ensure code quality and consistent formatting, prfmodel uses [ruff](https://docs.astral.sh/ruff/). Ruff checks are run on the whole repository, including tests.

To check whether your changes pass ruff checks, run:

```shell
uv run ruff check
uv run ruff format
```

## Pre-commit checks

prfmodel uses [prek](https://prek.j178.dev/) to run checks (including ruff) before each commit. The checks are
configured in [`prek.toml`](prek.toml). To enable them, install the Git hook once after cloning the repository:

```shell
uv run prek install
```

Whenever you make a commit to a branch of the repository, the checks are run and the commit is rejected if they fail.
Make sure that all checks pass before you commit. You can also run the checks on all files manually:

```shell
uv run prek run --all-files
```

To update the hooks to their latest versions, run:

```shell
uv run prek update
```

## Package version number

- We recommend using [semantic versioning](https://semver.org/).
- While the package is in beta, the major version stays at `0` (e.g., `0.1.0`). We don't use pre-release suffixes
  such as `b1`.
- The package version is stored in `pyproject.toml` (`project.version` and `tool.bumpversion.current_version`) and in `src/prfmodel/__init__.py`.
- To increase the version number in all of these places at once, use [bump-my-version](https://callowayproject.github.io/bump-my-version/) (e.g., `uv run bump-my-version bump patch`).
- Don't forget to update the version number before making a release!

## Making a release

Releases are published to [PyPI](https://pypi.org/project/prfmodel/) by the
[`release.yml`](.github/workflows/release.yml) workflow when a GitHub release is published. The workflow builds the
sdist and wheel, checks their metadata, and installs the wheel into a fresh environment to check that it imports and
contains its package data before uploading it. It uses
[trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API token is needed.

### One-time setup

1. On [PyPI](https://pypi.org/manage/account/publishing/), add a trusted publisher with owner `popylar-org`,
   repository `prfmodel`, workflow `release.yml`, and environment `pypi`. Until the first release, the project
   does not exist on PyPI yet, so add it as a *pending* publisher, which also reserves the name.
2. Do the same on [TestPyPI](https://test.pypi.org/manage/account/publishing/) with environment `testpypi`.
3. In the GitHub repository settings, create the environments `pypi` and `testpypi`. Adding required reviewers to
   `pypi` means that every upload to PyPI has to be approved, even after a release was published by mistake.

### Steps

1. Update [`CHANGELOG.md`](CHANGELOG.md): move the entries under `[Unreleased]` to a new section for the version.
2. Update the version number with bump-my-version (see [Package version number](#package-version-number)), and merge
   the change into `main`.
3. Optionally, do a dry run: in the Actions tab, run the *Publish to PyPI* workflow manually on `main`. This
   uploads the package to TestPyPI instead. Install it from there into a fresh environment to check it:

   ```shell
   python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "prfmodel[tensorflow]==<version>"
   ```

   TestPyPI never accepts the same version twice, so repeating a dry run for the same version checks the build but
   does not replace the uploaded files.
4. Create a [GitHub release](https://github.com/popylar-org/prfmodel/releases/new) from `main` with a new tag that
   is the version with a leading `v` (e.g., `v0.1.0`). The workflow fails without uploading anything if the tag
   does not match the version in `pyproject.toml`.
5. Publishing the release starts the workflow. Check that it succeeded and that the version appears on PyPI.
