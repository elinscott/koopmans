#################
Input File Format
#################

Koopmans input files define the structure, workflow settings, k-points, and calculator
parameters for a calculation. They can be written in either ``json`` or ``yaml`` format as per the following example:

.. literalinclude:: tutorials/band_structures/silicon_finite_differences/si.json
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

.. Members come from ``koopmans.input_file.__all__``; adding an import cannot
   put a name on this page. ``KoopmansInput`` is the one exclusion because it is
   documented in full above.

.. automodule:: koopmans.input_file
   :members:
   :member-order: alphabetical
   :exclude-members: KoopmansInput
