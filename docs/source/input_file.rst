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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
External pseudoatomic projectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting ``pw2wannier90.atom_proj_ext`` Wannierizes from external projectors instead of the
pseudopotentials' own pseudoatomic orbitals. ``pw2wannier90.atom_proj_dir`` must point at a
directory holding one ``<element>.dat`` file per element in pw2wannier90's radial-projector format:
optional leading ``#`` comment lines, a ``<ngrid> <nproj>`` header, then the projectors' angular
momenta as Fortran list-directed input (values separated by blanks or commas, possibly spanning
lines, with ``r*v`` repeat counts), followed by the radial tables. Each entry of angular momentum
:math:`l` contributes :math:`2l+1` projectors, and together they fix the number of Wannier
functions. The directory is read on the computer the pw2wannier90 code runs on (with the default
localhost setup, the machine ``koopmans`` runs on, where it is also validated). All of the external
projectors are Löwdin-orthonormalized before pw2wannier90 projects onto them; freezing a subset of
them instead is deliberately unsupported (to be revisited with an ``element: orbital`` interface if
a use case appears).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Specifications for all elements of the input file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: koopmans.input_file
   :members:
   :imported-members:
   :member-order: alphabetical
   :exclude-members: KoopmansInput, load, Path, Field, model_validator, field_validator, AfterValidator, safe_load, ErrorDetails, BaseModel, convert_errors, prettify_errors
