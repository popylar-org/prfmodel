# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#

# -- Path setup --------------------------------------------------------------

import pathlib
import sys

sys.path.insert(0, pathlib.Path("../src").resolve().as_posix())
from prfmodel._docstring import _PARAMS

# -- Project information -----------------------------------------------------

project = "prfmodel"
copyright = "2025, Netherlands eScience Center, Vrije Universiteit Amsterdam, Netherlands Institute for Neuroscience"
author = "Malte Lüken, Flavio Hafner, Gilles de Hollander, Tomas Knapen"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "0.0.1"
# The full version, including alpha/beta/rc tags.
release = version

# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "autoapi.extension",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
]

# Add any paths that contain templates here, relative to this directory.
autoapi_template_dir = "_templates"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_templates"]

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Use autoapi.extension to run sphinx-apidoc -------

autoapi_dirs = ["../src/prfmodel"]
autoapi_member_order = "groupwise"
autoapi_own_page_level = "function"

autoapi_options = [
    "members",
    "inherited-members",
    "special-members",
    "imported-members",
]

myst_enable_extensions = [
    "dollarmath",
]

nb_merge_streams = True
nb_execution_mode = "cache"
nb_execution_timeout = 500
nb_execution_raise_on_error = True
nb_scroll_outputs = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# -- Options for Intersphinx

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    # Commonly used libraries, uncomment when used in package
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    # 'scikit-learn': ('https://scikit-learn.org/stable/', None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("http://pandas.pydata.org/docs/", None),
}

# -- Options for myst-nb

nb_kernel_rgx_aliases = {"python312": "python3"}


# -- Docstring placeholder substitution


def _process_docstring(  # noqa: PLR0913
    app: object,  # noqa: ARG001
    what: str,  # noqa: ARG001
    name: str,  # noqa: ARG001
    obj: object,  # noqa: ARG001
    options: dict | None,  # noqa: ARG001
    lines: list[str],
) -> None:
    """Replace %(key)s placeholders in docstrings with shared snippets.

    sphinx-autoapi does not automatically replace keys in the docstring when rendering the HTML documentation because
    it does not import the package modules. This function adds performs the replacement when sphinx-autoapi is run.

    """
    joined = "\n".join(lines)
    for key, value in _PARAMS.items():
        joined = joined.replace(f"%({key})s", value)
    lines[:] = joined.split("\n")


# -- Skip external inherited members


def _skip_external_inherited_members(  # noqa: PLR0913
    app: object,  # noqa: ARG001
    what: str,  # noqa: ARG001
    name: str,  # noqa: ARG001
    obj: object,
    skip: bool,  # noqa: ARG001
    options: dict | None,  # noqa: ARG001
) -> None:
    if not obj.inherited:
        return None

    inherited_from = obj.obj.get("inherited_from", {})
    if not inherited_from.get("full_name", "").startswith("prfmodel."):
        return True

    return None


def setup(app: object) -> None:
    """Connect `_process_docstring` to autodoc event."""
    app.connect("autoapi-skip-member", _skip_external_inherited_members)
    app.connect("autodoc-process-docstring", _process_docstring, priority=100)
