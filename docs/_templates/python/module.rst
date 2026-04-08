{% if obj.display %}
   {% if is_own_page %}
{{ obj.id }}
{{ "=" * obj.id|length }}

.. py:module:: {{ obj.name }}

      {% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|indent(3) }}

      {% endif %}

      {% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
      {% set visible_submodules = obj.submodules|selectattr("display")|list %}
      {% set visible_submodules = (visible_subpackages + visible_submodules)|sort %}
      {% block submodules %}
         {% if visible_submodules %}
Submodules
----------

.. toctree::
   :hidden:

            {% for submodule in visible_submodules %}
   {{ submodule.include_path }}
            {% endfor %}

.. autoapisummary::

                  {% for submodule in visible_submodules %}
   {{ submodule.id }}
                  {% endfor %}


         {% endif %}
      {% endblock %}
      {% block content %}
         {% set visible_children = obj.children|selectattr("display")|list %}
         {% set ns = namespace(class_ids=[], top_ids=[]) %}
         {% for submod in visible_submodules %}
            {% for klass in submod.children|selectattr("display")|selectattr("type", "equalto", "class")|list %}
               {% if klass.id not in ns.class_ids %}
                  {% set ns.class_ids = ns.class_ids + [klass.id] %}
               {% endif %}
            {% endfor %}
         {% endfor %}
         {% for submod in visible_submodules %}
            {% for klass in submod.children|selectattr("display")|selectattr("type", "equalto", "class")|list %}
               {% set has_internal_base = namespace(value=false) %}
               {% for base in klass.bases %}
                  {% for cid in ns.class_ids %}
                     {% if cid.endswith("." + base) or cid == base %}
                        {% set has_internal_base.value = true %}
                     {% endif %}
                  {% endfor %}
               {% endfor %}
               {% if not has_internal_base.value and klass.id not in ns.top_ids %}
                  {% set ns.top_ids = ns.top_ids + [klass.id] %}
               {% endif %}
            {% endfor %}
         {% endfor %}
         {% if ns.class_ids %}

Inheritance diagram
-------------------

.. inheritance-diagram:: {{ ns.class_ids | join(" ") }}
   {% if ns.top_ids %}
   :top-classes: {{ ns.top_ids | join(", ") }}
   {% endif %}
   :parts: -1

         {% endif %}
         {% if visible_children %}
            {% set visible_attributes = visible_children|selectattr("type", "equalto", "data")|list %}
            {% if visible_attributes %}
               {% if "attribute" in own_page_types or "show-module-summary" in autoapi_options %}

Attributes
----------

                  {% if "attribute" in own_page_types %}
.. toctree::
   :hidden:

                     {% for attribute in visible_attributes %}
   {{ attribute.short_name }} <{{ attribute.include_path }}>
                     {% endfor %}

                  {% endif %}
.. autoapisummary::

                  {% for attribute in visible_attributes %}
   {{ attribute.id }}
                  {% endfor %}
               {% endif %}


            {% endif %}
            {% set visible_exceptions = visible_children|selectattr("type", "equalto", "exception")|list %}
            {% if visible_exceptions %}
               {% if "exception" in own_page_types or "show-module-summary" in autoapi_options %}
Exceptions
----------

                  {% if "exception" in own_page_types %}
.. toctree::
   :hidden:

                     {% for exception in visible_exceptions %}
   {{ exception.short_name }} <{{ exception.include_path }}>
                     {% endfor %}

                  {% endif %}
.. autoapisummary::

                  {% for exception in visible_exceptions %}
   {{ exception.id }}
                  {% endfor %}
               {% endif %}


            {% endif %}
            {% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
            {% if visible_classes %}
               {% if "class" in own_page_types or "show-module-summary" in autoapi_options %}
Classes
-------

                  {% if "class" in own_page_types %}
.. toctree::
   :hidden:

                     {% for klass in visible_classes %}
   {{ klass.short_name }} <{{ klass.include_path }}>
                     {% endfor %}

                  {% endif %}
.. autoapisummary::

                  {% for klass in visible_classes %}
   {{ klass.id }}
                  {% endfor %}
               {% endif %}


            {% endif %}
            {% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}
            {% if visible_functions %}
               {% if "function" in own_page_types or "show-module-summary" in autoapi_options %}
Functions
---------

                  {% if "function" in own_page_types %}
.. toctree::
   :hidden:

                     {% for function in visible_functions %}
   {{ function.short_name }} <{{ function.include_path }}>
                     {% endfor %}

                  {% endif %}
.. autoapisummary::

                  {% for function in visible_functions %}
   {{ function.id }}
                  {% endfor %}
               {% endif %}


            {% endif %}
            {% set this_page_children = visible_children|rejectattr("type", "in", own_page_types)|list %}
            {% if this_page_children %}
{{ obj.type|title }} Contents
{{ "-" * obj.type|length }}---------

               {% for obj_item in this_page_children %}
{{ obj_item.render()|indent(0) }}
               {% endfor %}
            {% endif %}
         {% endif %}
      {% endblock %}
   {% else %}
.. py:module:: {{ obj.name }}

      {% if obj.docstring %}
   .. autoapi-nested-parse::

      {{ obj.docstring|indent(6) }}

      {% endif %}
      {% for obj_item in visible_children %}
   {{ obj_item.render()|indent(3) }}
      {% endfor %}
   {% endif %}
{% endif %}
