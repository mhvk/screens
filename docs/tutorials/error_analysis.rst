**************
Error analysis
**************

This tutorial describes how to propagate (a) the statistical uncertainties on
the parameters of a model that was fit to a time series of scintillation
velocities to (b) uncertainties on the physical parameters that describe the
pulsar system and the scattering screen. The model assumes a pulsar on a
circular orbit whose radiation is scattered by a single one-dimensional screen.
The tutorial builds upon a :doc:`preceding tutorial <infer_phys_pars>` that
described in more detail how the physical parameters can be inferred from the
model parameters. Like in the second part of that tutorial, this
error-propagation tutorial assumes there are some constraints on the distance
to the pulsar. The tutorial uses the fit results generated in the tutorial on
:doc:`fitting scintillation velocities <fit_velocities>`. These fit results are
available for download:
:download:`fit-results-J0437.npz <../data/fit-results-J0437.npz>`

For a derivation of the equations seen here, refer to the
:doc:`scintillation velocities background <../background/velocities>`.
Further explanations can be found in `Marten's scintillometry page
<http://www.astro.utoronto.ca/~mhvk/scintillometry.html#org5ea6450>`_
and Daniel Baker's "`Orbital Parameters and Distances
<https://eor.cita.utoronto.ca/images/4/44/DB_Orbital_Parameters.pdf>`_"
document. As in that document, the practical example here uses the parameter
values for the pulsar PSR J0437--4715 as derived by `Reardon et al. (2020)
<https://ui.adsabs.harvard.edu/abs/2020ApJ...904..104R/abstract>`_.

The combined codeblocks in this tutorial can be downloaded as a Python script
and as a Jupyter notebook:

:Python script:
    :jupyter-download:script:`error_analysis.py <error_analysis>`
:Jupyter notebook:
    :jupyter-download:notebook:`error_analysis.ipynb <error_analysis>`
