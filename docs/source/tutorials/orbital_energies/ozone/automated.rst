###########################
 Letting koopmans do it
###########################

The :doc:`previous part <by_hand>` computed one screening parameter, for one orbital, in
three calculations run by hand. This part computes one for every orbital of the same
molecule, and both the ionization potential and the electron affinity that follow from
them, in a single command — the same physics, with the bookkeeping handed over.

****************
 The input file
****************

Download :download:`ozone.yaml <ozone.yaml>` and place it in an empty directory. Here it
is in full:

.. literalinclude:: ozone.yaml
    :language: yaml

The comments say what each keyword is; the ``workflow`` block is worth a little more
than that:

- ``screening_method: dscf`` means that, for each orbital, the code performs constrained
  calculations with :math:`N`, :math:`N-1`, or :math:`N+1` electrons and compares the
  resulting energy differences against the orbital energies (see :doc:`the theory page
  <../../../theory>`).
- ``init_orbitals: kohn-sham`` uses the Kohn-Sham orbitals of the base functional as the
  variational orbitals — and for KI they also *remain* the variational orbitals, since
  KI leaves the orbitals of its base functional unchanged. This is common practice for
  molecules; periodic systems use Wannier functions instead.
- ``alpha_numsteps`` above is 1, so the screening parameters are computed once, from a
  guess, and used. More steps refine them self-consistently.
- ``pseudo_library`` also fixes PBE as the base functional that the KI correction is
  applied on top of.

The ``atoms`` block describes the cell and the atoms in it, much like a Quantum ESPRESSO
input file. The positions are Cartesian, in the units the block declares.

.. question:: Why is the simulation cell so much larger than the molecule itself?

    The cell is a box of vacuum that keeps the molecule's periodic images apart — a
    plane-wave code works in a supercell even when the system is treated as
    non-periodic. Vacuum matters more here than in a ground-state calculation, because
    the screening calculations give the molecule a net charge and charged images
    interact through the long-range Coulomb tail; on top of the vacuum, the code applies
    a counter-charge correction that compensates the residual interaction between
    images.

The ``calculator_parameters`` block holds the plane-wave settings:

- Ozone has 18 valence electrons and therefore nine filled orbitals; ``nbnd: 10`` adds
  one empty orbital, which we need because the electron affinity is the energy of the
  LUMO.
- ``ecutrho`` sits under ``kcp.system`` rather than at the top level because it is
  passed straight through to ``kcp.x``, the Quantum ESPRESSO code that evaluates the
  corrected functional. Keywords that ``koopmans`` does not define itself can be handed
  to the underlying codes this way.

*************************
 Running the calculation
*************************

Make sure you have :doc:`installed the engine <../../../installation>` (``koopmans
install``; this workflow needs ``pw.x`` and ``kcp.x``), then run

.. code-block:: console

    $ koopmans run ozone.yaml

The terminal shows a live progress table that grows as the workflow proceeds. At the end
it reads

.. code-block:: text

     Step                                                                      Status
     Koopmans DSCF Workflow                                                  finished
       DFT Init (nspin=1)                                                    finished
       DFT Init (nspin=2; dummy)                                             finished
       DFT Init (nspin=2)                                                    finished
       Compute Screening Parameters                                          finished
         Iteration 1                                                         finished
           KI Trial                                                          finished
           Compute Orbital Screening Parameters                              finished
             Compute Alpha Orb 1                                             finished
               DFT N-1                                                       finished
             Compute Alpha Orb 2                                             finished
               DFT N-1                                                       finished
             ...
             Compute Alpha Orb 9                                             finished
               DFT N-1                                                       finished
             Compute Alpha Orb 10                                            finished
               DFT N+1 Dummy                                                 finished
               PZ Print                                                      finished
               DFT N+1                                                       finished
       Run Final KI                                                          finished
         KI Final                                                            finished

    Workflow completed successfully!

Reading it top to bottom:

**Initialization.** The first three steps initialize the electron density and the
variational orbitals with PBE. The screening calculations later on add and remove single
electrons, which requires a spin-resolved description, so the workflow ends up at a
spin-resolved calculation — but it gets there via a detour: a spin-unpolarized
calculation, a dummy spin-resolved calculation that lays out the restart files, and a
final spin-resolved calculation that restarts from the spin-unpolarized density
duplicated into both spin channels.

.. question:: Why the detour, instead of one spin-resolved calculation from scratch?

    For ozone — a closed-shell molecule — a direct spin-resolved calculation would in
    fact converge to the correct spin-symmetric solution. The detour has two virtues
    all the same. In harder systems, a spin-resolved calculation started from scratch
    can collapse into a spurious broken-symmetry solution with
    :math:`n^\uparrow(\mathbf{r}) \neq n^\downarrow(\mathbf{r})`; handing it an
    already-converged symmetric density avoids that. And it is cheaper: most of the
    self-consistency cycles happen in the spin-unpolarized problem, which has half
    the wavefunctions.

From this point on the density never changes: KI, by construction, returns the same
density as its base functional. (This is not true of KIPZ.)

**Compute screening parameters.** This is the bulk of the calculation, and the part that
makes a Koopmans calculation more than a DFT calculation. Each orbital :math:`i` has a
screening parameter :math:`\alpha_i` that accounts for how the rest of the system
relaxes when that orbital's occupancy changes. With the ΔSCF method they come from
total-energy differences:

- the *KI trial* step evaluates the KI functional with a starting guess for every
  screening parameter (:math:`\alpha_i = 0.6`), yielding :math:`E(N)` and each orbital's
  energy both at the guessed screening and at zero screening;
- then, for each of the nine filled orbitals, an :math:`N-1`-electron constrained PBE
  calculation empties that orbital while the rest of the density relaxes, yielding
  :math:`E_i(N-1)`;
- for the one empty orbital the procedure runs in reverse — two preparatory calculations
  followed by an :math:`N+1`-electron constrained calculation in which the orbital is
  filled, yielding :math:`E_i(N+1)`.

Comparing these total-energy differences with the corresponding orbital energies gives
an updated screening parameter for every orbital. With ``alpha_numsteps: 1``, the loop
stops there.

**KI final.** The workflow evaluates the KI functional once more, now with the computed
screening parameters in place. The orbital energies of this final calculation are the
spectral properties we are after.

*************
 The outputs
*************

The results land in a directory named after the input file — here ``ozone/`` — laid out
to mirror the workflow outline above:

.. code-block:: text

    ozone
    ├── 01-count_electrons_task
    │   └── inputs
    ├── 02-dft_init_nspin1
    │   ├── inputs
    │   └── outputs
    ├── 03-dft_init_nspin2_dummy
    │   ├── inputs
    │   └── outputs
    ├── 04-dft_init_nspin2
    │   ├── inputs
    │   └── outputs
    ├── 05-ComputeScreeningParameters
    │   └── 01-ScreeningIteration
    │       ├── 01-ki_trial
    │       │   ├── inputs
    │       │   └── outputs
    │       └── 02-compute_orbital_screening_parameters
    │           ├── 01-compute_alpha_orb_1
    │           │   ├── 01-dft_n_minus_1
    │           │   └── 02-compute_alpha_from_dscf
    │           ├── ...
    │           └── 10-compute_alpha_orb_10
    │               ├── 01-dft_n_plus_1_dummy
    │               ├── 02-pz_print
    │               ├── 03-dft_n_plus_1
    │               └── 04-compute_alpha_from_dscf
    ├── 06-RunFinalKI
    │   ├── inputs
    │   └── outputs
    └── README

One directory per step, numbered in the order the steps ran. A step that is a Quantum
ESPRESSO calculation holds the exact input file the engine generated (``aiida.cpi``)
plus its pseudopotentials in ``inputs/``, and everything the calculation wrote
(``aiida.cpo`` and more) in ``outputs/``. A step that is a piece of python — counting
the electrons, turning the trial energies into a screening parameter — has an
``inputs/`` and no ``outputs/``; steps that write no files at all do not appear.

Files shared between steps are stored once and referenced from everywhere else by a
relative symlink: each calculation's copy of the pseudopotential is a link back to the
first step that staged it. The ``README`` in the directory root says which copying tools
preserve those links — most do, but ``rsync -r`` without ``-a`` silently drops them.

.. note::

    The engine also keeps its own complete record of every calculation in its database —
    that is how an interrupted workflow resumes where it left off, and how a repeated
    calculation is served from cache instead of running twice. The ``ozone/`` directory
    is a plain-file export of that record for you to read.

**************************
 Interpreting the results
**************************

The ionization potential (IP) is the negative of the HOMO energy, and the electron
affinity (EA) is the negative of the LUMO energy. Open
``ozone/06-RunFinalKI/outputs/aiida.cpo`` and search near the bottom for the ``HOMO
Eigenvalue`` and ``LUMO Eigenvalue`` lines — and, for the PBE comparison, find the same
lines in the initialization output, ``ozone/04-dft_init_nspin2/outputs/aiida.cpo``.

.. question:: What do you find?

    Near the bottom of the final KI output:

    .. code-block:: text

        HOMO Eigenvalue (eV)

        -12.5234

        LUMO Eigenvalue (eV)

        -1.8221

        Electronic Gap (eV) =    10.7013


        Eigenvalues (eV), kp =   1 , spin =  1

        -40.1865  -32.9126  -24.2279  -19.6844  -19.4901  -19.2698  -13.6039  -12.7621  -12.5234

        Empty States Eigenvalues (eV), kp =   1 , spin =  1

        -1.8221

    The initialization output puts the PBE HOMO at −7.9550 eV and the PBE LUMO at
    −6.1684 eV.

The KI ionization potential is therefore 12.52 eV. This compares extremely well with the
`experimental value
<https://webbook.nist.gov/cgi/cbook.cgi?ID=C10028156&Mask=20#Ion-Energetics>`_ of ~12.5
eV, and is a dramatic improvement on PBE, whose HOMO would put it at 7.96 eV — more than
4.5 eV too small. Likewise the KI electron affinity is 1.82 eV (experiment: ~2.1 eV),
where PBE would have given 6.17 eV.

The comparison need not stop at the frontier orbitals: KI predicts a binding energy —
minus the orbital energy — for *every* occupied orbital, and gas-phase photoemission
measures them. The fair comparison is against the three outermost occupied orbitals,
whose experimental binding energies are cleanly resolved — 12.73, 13.00 and 13.54 eV
(`Mocellin et al., Chem. Phys. Lett. 375, 76 (2003)
<https://doi.org/10.1016/S0009-2614(03)00818-2>`_); the assignments of the deeper
orbitals are less certain, so they are left out. The three KI values sit at the end of
the final KI output's ``Eigenvalues`` line, and the PBE ones in the corresponding line
of the initialization output. Plotting one against the other:

.. figure:: ozone_spectrum.svg
    :width: 420
    :align: center

    Calculated against experimental binding energies for the three outermost occupied
    orbitals of ozone, from this run; a point on the dashed line agrees perfectly with
    experiment. Generated from the ``ozone/`` output directory with
    :download:`plot_ozone_spectrum.py <plot_ozone_spectrum.py>`.

KI lands within a quarter of an electronvolt of experiment for all three orbitals; PBE
misses by more than four.

The screening parameters behind this result are recorded in the final calculation's
input: ``ozone/06-RunFinalKI/inputs/file_alpharef.txt`` lists one :math:`\alpha_i` per
orbital, here ranging from 0.66 to 0.78 — each one computed, not fitted.

.. question:: Why one screening parameter per orbital, rather than a single α?

    A single :math:`\alpha` fitted to the HOMO would enforce the Koopmans condition
    on the HOMO alone — the ionization potential would come out the same by
    construction, but every other level would suffer. In particular the electron
    affinity worsens noticeably, because an :math:`\alpha` tuned for the HOMO cannot
    simultaneously describe the LUMO. Screening is an orbital-by-orbital affair.

.. question:: What happens if you increase ``alpha_numsteps`` to 2 and rerun?

    With ``alpha_numsteps: 1`` the screening parameters are computed once from the
    starting guess and never checked for self-consistency. With a second step, the
    workflow repeats the screening loop starting from the just-computed parameters
    instead of 0.6, and only updates a parameter further if its residual is above the
    convergence threshold (``alpha_conv_thr``). You should find that the parameters
    barely move — the first pass already brought them close to self-consistency.

.. question:: How does the cost of all this scale with system size?

    Each screening iteration runs roughly one constrained calculation per orbital, so
    a ΔSCF Koopmans calculation costs about :math:`N_\text{orb}` times a single DFT
    calculation — multiplied by ``alpha_numsteps``. Crystals are worse still: every
    constrained :math:`N \pm 1` calculation needs a supercell large enough that the
    added electron or hole does not overlap its own periodic images, so each
    calculation in the loop is itself far more expensive. That is what makes ΔSCF
    impractical for solids, and why the :doc:`silicon tutorial
    <../../band_structures/silicon_linear_response/index>` computes the screening
    parameters from linear response (DFPT) instead.

*************
 From python
*************

The same calculation runs from python, which is the easier route for sweeping a
parameter or working in a notebook. Reading the input file and running it are a line
each, and the results come back as a dict instead of as output files to search:

.. code-block:: python

    from koopmans import read_input_file, run

    results = run(read_input_file("ozone.yaml"))

    homo = results["parameters"]["homo_energy"]
    lumo = results["parameters"]["lumo_energy"]

    print(f"IP = {-homo:.2f} eV")  # IP = 12.52 eV
    print(f"EA = {-lumo:.2f} eV")  # EA = 1.82 eV

Energies are in eV, and the ionization potential and electron affinity are the negated
HOMO and LUMO energies as before — the same two numbers read out of ``aiida.cpo`` above.
``results`` carries the rest of the final calculation's outputs alongside them,
``eigenvalues`` among them: the orbital energies of the ``Eigenvalues`` line, as an
array.

``run`` blocks until the workflow finishes. For a calculation long enough that this is
inconvenient, ``submit`` returns an integer id immediately and leaves the workflow
running in the background, and ``outputs(pk)`` reads the finished result back by that id
— in this python session or a later one. See :doc:`../../../python_api` for the three
verbs in full.

.. admonition:: Coming from the ASE-based koopmans 1.x?

    - ``functional`` and ``method`` are now ``correction`` and ``screening_method``, and
      the ``engine`` block is gone — restarting and caching are handled by the engine
      automatically.
    - There is no ``ozone.md`` outline and no ``ozone.pkl``; progress is shown live in
      the terminal, and the calculations' files are exported to ``ozone/`` when the
      workflow ends rather than written in place as it runs.
    - Quantum ESPRESSO files are named ``aiida.cpi`` / ``aiida.cpo`` inside each
      calculation's ``inputs/`` and ``outputs/`` directories, instead of
      ``<calculation>.cpi`` / ``<calculation>.cpo``.

*****************
 Further reading
*****************

- :doc:`The theory page <../../../theory>` explains the functionals and the role of the
  screening parameters.
- The `2023 koopmans paper <https://doi.org/10.1021/acs.jctc.3c00652>`_ derives the ΔSCF
  screening procedure in full and benchmarks it against experiment.
- The :doc:`next tutorial <../magnetic/index>` treats molecules whose ground state is
  spin-polarized.
- The :doc:`silicon tutorial <../../band_structures/silicon_linear_response/index>`
  computes the screening parameters from linear response — the route that makes crystals
  affordable.
