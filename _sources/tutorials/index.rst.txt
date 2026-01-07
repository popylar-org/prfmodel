Tutorials
=========

Here you can find tutorials that explain how to use prfmodel. Stay tuned because more tutorials are in the making.

You want to contribute a tutorial? Make a `pull request <https://github.com/popylar-org/prfmodel/pulls>`_!

List of tutorials:

.. toctree::
   :maxdepth: 0
   :titlesonly:
   :glob:

   simple_prf_simulated.md

Tutorial development
--------------------

Tutorials are stored in the `MyST Markdown <https://myst-parser.readthedocs.io/en/latest/>`_ format. They are
automatically rendered by Sphinx in the local and online documentation. However, for development purposes, it can be
useful to locally convert the MyST Markdown files into Jupyter notebooks (and vice versa). The conversion can be done
with `jupytext <https://jupytext.readthedocs.io/en/latest/index.html>`_.

To locally convert MyST Markdown files into Jupyter notebooks:

.. code-block:: bash

   jupytext docs/tutorials/tutorial.ipynb --to myst

To locally convert Jupyter notebooks into MyST Markdown files:

.. code-block:: bash

   jupytext docs/tutorials/tutorial.md --to ipynb
