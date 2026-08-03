###########################################
 Screening parameters from a trained model
###########################################

Computing the screening parameters is the expensive part of a Koopmans calculation. When
you have many similar systems to get through — snapshots along a molecular-dynamics
trajectory, say — you can train a model on a few of them and predict the rest. This
tutorial trains on a handful of water configurations, predicts the screening parameters
for the others, and checks how far the prediction can be trusted.

.. note::

    This tutorial has not been written yet. It is tracked by `issue #67
    <https://github.com/elinscott/koopmans/issues/67>`_.
