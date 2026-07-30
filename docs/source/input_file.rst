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
directory (on the machine ``koopmans`` runs on) holding one ``<element>.dat`` file per element in
pw2wannier90's radial-projector format; each entry of angular momentum :math:`l` contributes
:math:`2l+1` projectors, and together they fix the number of Wannier functions. By default
pw2wannier90 Löwdin-orthonormalizes all of the projectors before projecting onto them;
``pw2wannier90.atom_proj_frozen`` (a list of 1-based projector indices, counting site by site and
then file entry by file entry) instead holds the listed projectors *frozen* — kept exactly as given
— while only the remaining ones are orthonormalized against them. Freezing acts per orbital: an
entry's full :math:`2l+1` multiplet, on every site of its element, must be listed together.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Specifications for all elements of the input file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: koopmans.input_file
   :members:
   :imported-members:
   :member-order: alphabetical
   :exclude-members: KoopmansInput, load, Path, Field, model_validator, field_validator, safe_load, ErrorDetails, BaseModel, convert_errors, prettify_errors
