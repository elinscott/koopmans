###########################################
 Screening parameters from a trained model
###########################################


Computing the screening parameters is the expensive part of a Koopmans calculation. When
you have many similar systems — snapshots along a molecular-dynamics
trajectory, say — it is a waste of time to repeatedly compute screening parameters
for what are ultimately very similar orbitals and screening environments. In this case,
you can train a model on a few systems and use it to predict the screening parameters for
the rest. This tutorial trains a model on a handful of water molecules, checks it against
configurations it has never seen, and then runs a calculation with the model's predictions
in place of explicit screening parameter calculations.

.. note::

    This machine-learning approach was first published in :cite:`Schubert2024` --- read that
    paper for all of the details!

The twenty configurations are one water molecule with its atoms displaced at random.
:download:`generate_snapshots.py <generate_snapshots.py>` writes them as two multi-frame
xyz files: five to train on, and fifteen to test and predict with.

.. figure:: snapshots.gif
    :width: 400
    :align: center

    The twenty configurations of water used in this tutorial.

.. note::

    A model like this earns its keep for liquids and solids, while here we apply
    it to a molecule for demonstration purposes. For this reason, we will treat
    the system like a liquid or a solid:

    - the molecule sits in a box with periodic boundary conditions
    - we use maximally localized Wannier functions for the variational orbitals rather than Kohn-Sham orbitals.

The key settings are found in the ``ml`` block. We can choose one of three modes:

``train``
    computes the screening parameters of every configuration and fits a model to them;

``test``
    computes them *and* predicts them, and reports how far apart the two are;

``predict``
    predicts them, and never computes them.

We will go through each of these in turn.

******************
 Training a model
******************

Download :download:`train.yaml <train.yaml>` and :download:`training_snapshots.xyz
<training_snapshots.xyz>` into an empty directory. Here is the input file in full:

.. literalinclude:: train.yaml
    :language: yaml

.. warning::

    The cell, the cutoff and the k-point sampling in these files are deliberately rough,
    so that the tutorial finishes in a reasonable time. They are not converged! Use
    these files to learn the input format, not as something to copy-paste for production
    work.

Most of this file describes an ordinary Koopmans calculation you have seen before; two things are new:

.. literalinclude:: train.yaml
    :language: yaml
    :start-at: snapshots:
    :end-at: snapshots:

means we run one calculation on each of the configurations found in this multi-frame xyz
file in place of the ``atomic_positions`` block we have been using for single-structure
calculations. Every frame shares the same cell, composition, and so on.

Meanwhile, the ``ml`` block:

.. literalinclude:: train.yaml
    :language: yaml
    :start-at: ml:
    :end-at: occ_and_emp_together

defines the model. ``descriptor`` decides what the model sees of an orbital: ``power_spectrum``
expands each orbital's density on a radial basis out to ``r_max``, with radial channels
up to ``n_max`` and angular momenta up to ``l_max``, and feeds the model the rotationally
invariant power spectrum of that expansion. (The cheaper
alternative, ``self_hartree``, sees only the electrostatic self-interaction energy of
that density; we will discuss it later.) ``estimator`` decides how the model fits screening parameters to
the descriptor, and ``occ_and_emp_together: false`` fits the filled and the empty
orbitals separately — one screening parameter says what happens when an electron leaves
an orbital, the other what happens when one arrives, and the two need not be related.

Run the calculation with

.. code-block:: console

    $ koopmans run train.yaml

The progress table has one branch per configuration, each of them a complete
Koopmans calculation, plus a second branch per configuration that builds the
``power_spectrum`` dataset the model is fitted to:

.. code-block:: text

     Step                                                                      Status
     Trajectory                                                              finished
       Snapshot 1                                                            finished
         Wannier initialization                                              finished
           Wannierization                                                    finished
           Supercell folding                                                 finished
           DFT staging                                                       finished
           DFT initialization                                                finished
         Screening parameters                                                finished
           Iteration 1                                                       finished
             Trial KI                                                        finished
             Orbital screening                                               finished
               Orbital 1                                                     finished
               ...
               Orbital 6                                                     finished
         Final KI                                                            finished
       Descriptors (snapshot 1)                                              finished
       Snapshot 2                                                            finished
         ...
       Descriptors (snapshot 2)                                              finished
       ...
       Snapshot 5                                                            finished
       Descriptors (snapshot 5)                                              finished

    Workflow completed successfully!
    Trained model stored as node <id number> (…) — reference it via `ml: {model: <id number>}`.

Each ``Descriptors (snapshot N)`` branch decomposes the Wannier functions into
the power spectrum descriptors, and this is paired with the screening parameters the
``Snapshot N`` branch above it already computed.

The workflow generates a model from that dataset, as can be seen from the last line of the
output.

*******************
 Testing the model
*******************

A model that has only ever been checked against its own training data tells you nothing.
Download :download:`test.yaml <test.yaml>` and :download:`testing_snapshots.xyz
<testing_snapshots.xyz>` into the same directory as ``train.yaml`` and run

.. code-block:: console

    $ koopmans run test.yaml

The input file differs from ``train.yaml`` in two places — it reads the other fifteen
configurations (i.e. configurations not seen during training), and its ``ml`` block
only differs in its choice of mode and in naming the model to use:

.. literalinclude:: test.yaml
    :language: yaml
    :start-at: ml:
    :end-at: model_file

A test run does everything the training run did, computing every screening parameter
from first principles, and then does two things more: it predicts every screening
parameter too, and it runs a *second* final KI calculation with the predicted values in place of the computed ones.
Both show up inside each configuration's branch:

.. code-block:: text

    Snapshot 1                                                               finished
      ...
      Descriptors                                                            finished
      Final KI                                                               finished
      Final KI (predicted alphas)                                            finished
    Descriptors (snapshot 1)                                                 finished
    ...

The two final KI calculations both start from the same trial calculation and differ
only in the screening parameters, so whatever separates their orbital energies is purely
due to the differences between the predicted and true screening parameters. These differences
can easily be visualised:

.. code-block:: console

    $ koopmans plot parity test/ --residuals

.. figure:: parity.svg
    :width: 620
    :align: center

    Left: the predicted screening parameters minus the computed ones, against the
    computed value, for the 90 orbitals of the fifteen test configurations, with a
    marginal histogram of the residual. Right: the same, for the orbital energies of
    the final KI calculation.

Over the 90 orbitals of the fifteen test configurations, the screening parameters come
out with a mean absolute error of 0.0075 and a root-mean-square error of 0.0145, the
worst orbital out by 0.074 (for reference, the screening parameters for this system range from 0.33 to 0.59).

Ultimately, we don't care how the screening parameters change: we care about the orbital
energies. As the parity plot shows, the fifteen configurations' orbital energies move by a
root-mean-square of 26.7 meV on average, 27.0 meV at the median, and as much as 55 meV
for the worst configuration.

.. question:: Does the cheaper ``self_hartree`` descriptor do just as well?
    (To test this, set ``descriptor: self_hartree`` in the ``ml`` block of every input above,
    dropping ``n_max``/``l_max``/``r_min``/``r_max`` — ``self_hartree`` carries no radial
    basis — and repeat the training and testing runs.)

    No. On the same five-configuration
    training set and fifteen-configuration test set, the screening parameters come out
    with a mean absolute error of 0.0272, a root-mean-square error of 0.0327, and a
    worst orbital out by 0.097 — more than twice ``power_spectrum``'s error on every
    measure.

    The reason is visible in the model itself: ``self_hartree`` sees one number per
    orbital, so its filled-orbital submodel has one coefficient to work with. It would
    work if the filled orbitals self-Hartree energies were perfectly correlated with
    their screening parameters, but they are not. The power spectrum descriptor is much
    richer: its 441 features give its filled-orbital submodel much more flexibility.

    Training ``self_hartree`` on ten configurations instead of five does not fix this.
    A one-number descriptor's limit is what it can distinguish, not how much data it is fitted to.

*****************
 Using the model
*****************

Download :download:`predict.yaml <predict.yaml>` into the same directory as
``train.yaml``, and run

.. code-block:: console

    $ koopmans run predict.yaml

Again, its ``ml`` block only differs in its choice of mode and in naming the model:

.. literalinclude:: predict.yaml
    :language: yaml
    :start-at: ml:
    :end-at: model_file

In this mode, the screening parameters are never computed. Each configuration runs a
decompose pass and a trial KI calculation side by side, the model turns the descriptor
into screening parameters, and the final KI calculation applies them:

.. code-block:: text

    Snapshot 1                                                               finished
      Wannier initialization                                                 finished
      Predicted screening parameters                                         finished
        Descriptors                                                          finished
        Trial KI                                                             finished
      Final KI                                                               finished

Compare that with the training run's branch: the whole ``Orbital screening`` fan-out,
one constrained calculation per orbital, is gone. What remains — the Wannierization,
the initialization, the descriptor pass, the trial and the final calculation — is what
sets the floor on how cheap a predicted Koopmans calculation can be.
