#############
 From python
#############

Everything ``koopmans run`` does is available from python, for scripting
parameter sweeps or driving calculations from a notebook. The same input
that lives in a JSON or YAML file can be read or constructed directly, and
three verbs cover the life of a calculation:

- :func:`koopmans.build` prepares the calculation without running anything,
  so an input can be checked cheaply;
- :func:`koopmans.run` runs it to completion, blocking until it finishes;
- :func:`koopmans.submit` hands it to the background daemon and returns
  immediately (or with ``wait=True``, blocks like ``run``).

``run`` and ``submit`` return a :class:`koopmans.Results`, which exposes
the quantities the tutorials read — total energy, orbital energies,
screening parameters, ionization potential and electron affinity, all in
eV — and can write the same per-step directory layout as ``koopmans run``:

.. code:: python

   from koopmans import read_input_file, run

   results = run(read_input_file("ozone.json"))

   print(f"IP = {results.ionization_potential:.2f} eV")
   print(f"EA = {results.electron_affinity:.2f} eV")
   results.dump("ozone")

An input can equally be built without a file — it is the same model the
file parser produces:

.. code:: python

   from koopmans import KoopmansInput, submit

   inp = KoopmansInput.model_validate(
       {
           "workflow": {"task": "singlepoint", "correction": "ki"},
           # ... the same blocks an input file holds ...
       }
   )
   results = submit(inp)  # returns immediately; the daemon runs it

A submitted calculation is identified by its integer id: ``results.pk``
survives the python session, and ``Results.from_pk`` reconnects to it
later. Provenance is stored by AiiDA, the workflow engine underneath; none
of its machinery is needed to read results back.

***********
 Reference
***********

.. automodule:: koopmans.api
   :members:
