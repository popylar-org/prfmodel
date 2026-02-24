Tutorials
=========

Here you can find tutorials and examples that explain how to use prfmodel. Tutorials are shorter and explain specific
features of prfmodel while examples are longer and contain end-to-end applications of the package. Stay tuned because
more tutorials and examples are in the making.

You want to contribute a tutorial or example? Make a `pull request <https://github.com/popylar-org/prfmodel/pulls>`_!

.. _tutorials:

Tutorials
---------

There are no tutorials available yet. Please stay tuned!

.. _examples:

Examples
--------

.. toctree::
   :maxdepth: 0
   :titlesonly:
   :glob:

   prf_2d_fmri_visual.md

Development
-----------

Tutorials and examples are stored in the `MyST Markdown <https://myst-parser.readthedocs.io/en/latest/>`_ format.
They are automatically rendered by Sphinx in the local and online documentation. However, for development purposes,
it can be useful to locally convert the MyST Markdown files into Jupyter notebooks (and vice versa). The conversion
can be done with `jupytext <https://jupytext.readthedocs.io/en/latest/index.html>`_.

To locally convert MyST Markdown files into Jupyter notebooks:

.. code-block:: bash

   jupytext docs/tutorials/tutorial.md --to ipynb

To locally convert Jupyter notebooks into MyST Markdown files:

.. code-block:: bash

   jupytext docs/tutorials/tutorial.ipynb --to myst
