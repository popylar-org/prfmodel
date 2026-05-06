API Reference
=============

This page contains auto-generated API reference documentation [#f1]_.

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
