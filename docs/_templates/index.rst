API Reference
=============

This page contains auto-generated API reference documentation [#f1]_. While the API reference includes small exmaples
for functions and classes, we recommend taking a look at the :ref:`tutorials` section for a detailed introduction to
the package.

The prfmodel package includes several submodules listed below. The most important functions and classes can be found
in the :py:mod:`~prmfodel.models`, :py:mod:`~prmfodel.fitters`, and :py:mod:`~prmfodel.stimuli` submodules.

.. toctree::
   :hidden:
   :maxdepth: 1

   {% for page in pages|selectattr("is_top_level_object") %}
   {% set visible_subpackages = page.subpackages|selectattr("display")|list %}
   {% set visible_submodules = page.submodules|selectattr("display")|list %}
   {% for submodule in (visible_subpackages + visible_submodules)|sort %}
   {{ submodule.include_path }}
   {% endfor %}
   {% endfor %}

{% for page in pages|selectattr("is_top_level_object") %}
{% set visible_subpackages = page.subpackages|selectattr("display")|list %}
{% set visible_submodules = page.submodules|selectattr("display")|list %}
{% if visible_subpackages or visible_submodules %}
.. autoapisummary::

   {% for submodule in (visible_subpackages + visible_submodules)|sort %}
   {{ submodule.id }}
   {% endfor %}

{% endif %}
{% endfor %}

.. [#f1] Created with `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_
