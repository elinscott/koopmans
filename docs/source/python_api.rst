#############
 From python
#############

Everything ``koopmans run`` does is available from python, for scripting
parameter sweeps or driving calculations from a notebook. The same input
that lives in a JSON or YAML file can be read or constructed directly, and
three verbs cover the life of a calculation:

- :func:`koopmans.build` prepares the calculation without running anything,
  so an input can be checked cheaply;
- :func:`koopmans.run` runs it to completion, blocking until it finishes,
  and returns its outputs;
- :func:`koopmans.submit` hands it to the background daemon and returns the
  calculation's integer id immediately (or with ``wait=True``, blocks like
  ``run``).

Outputs come back as a plain dict, keyed by output name with every value a
plain python or numpy one — energies in eV:

.. code:: python

   from koopmans import read_input_file, run

   results = run(read_input_file("ozone.yaml"))

   print(f"IP = {-results['parameters']['homo_energy']:.2f} eV")
   print(f"EA = {-results['parameters']['lumo_energy']:.2f} eV")

An input can equally be built without a file — it is the same object the
file parser produces:

.. code:: python

   from koopmans import KoopmansInput, outputs, submit

   inp = KoopmansInput(
       workflow={"task": "singlepoint", "correction": "ki"},
       # ... the same blocks an input file holds ...
   )
   pk = submit(inp)  # returns immediately; the daemon runs it
   # ... later, in this session or another:
   results = outputs(pk)

The integer id survives the python session, and :func:`koopmans.outputs`
reads the finished calculation back by it — raising, rather than returning
half a result, while the calculation still runs or if it failed.
Provenance is stored by AiiDA, the workflow engine underneath; none of its
machinery is needed to read outputs back, and the per-step directory
layout ``koopmans run`` writes is available from
``koopmans.aiida.dumping.dump_workgraph``.

***********
 Reference
***********

.. automodule:: koopmans.api
   :members:
