#################
Input File Format
#################

Koopmans input files define the structure, workflow settings, k-points, and calculator
parameters for a calculation. They can be written in either ``json`` or ``yaml`` format as per the following example:

.. literalinclude:: tutorials/si.json
   :language: json
   :caption: Silicon tutorial input file


The formats of each of the sections are defined as follows

.. autopydantic_model:: koopmans.input_file.KoopmansInput
   :members:
   :member-order: bysource
   :exclude-members: from_file

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Specifications for all elements of the input file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: koopmans.input_file
   :members:
   :imported-members:
   :member-order: alphabetical
   :exclude-members: KoopmansInput, load, Path, Field, model_validator, field_validator, safe_load, ErrorDetails, BaseModel, convert_errors, prettify_errors
