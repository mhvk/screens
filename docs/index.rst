.. include:: ../README.rst

.. toctree::
    :hidden:

    background/simple_examples
    background/multiple_screens
    background/velocities
    background/glossary

    tutorials/single_screen
    tutorials/screen1d
    tutorials/two_screens

    tutorials/gen_velocities
    tutorials/fit_velocities
    tutorials/infer_phys_pars


Background
==========

* :doc:`background/simple_examples`:
  the dynamic and secondary spectra, and wavefield for elementary situations
* :doc:`background/multiple_screens`:
  derivation of the scattering angles and velocities
* :doc:`background/velocities`:
  derivation for a pulsar on a circular orbit and a single linear screen
* :doc:`background/glossary`:
  list of terms used in modelling scintillation velocities


Tutorials
=========


Generating and processing scintillometry data
---------------------------------------------

* :doc:`tutorials/single_screen`:
  how to generate the dynamic and secondary spectrum
* :doc:`tutorials/screen1d`:
  generate synthetic data and visualize the system (for a single screen)
* :doc:`tutorials/two_screens`:
  how to use Screen1D for radiation scattered by multiple screens


Modelling scintillation velocities
----------------------------------

This is a sequence of interconnected tutorials dealing with time series of
scintillation velocities (or curvature). See also the background document on
:doc:`scintillation velocities <background/velocities>` for a derivation of the
equation appearing in these tutorials.

* :doc:`tutorials/gen_velocities`:
  make synthetic time series of scintillation velocities or curvature
* :doc:`tutorials/fit_velocities`:
  fit a phenomenological model to a time series of scintillation velocities
* :doc:`tutorials/infer_phys_pars`:
  retrieve the physical parameters of the system from the fit


Reference/API
=============

.. automodapi:: screens.fields
.. automodapi:: screens.dynspec
.. automodapi:: screens.secspec
.. automodapi:: screens.screen
.. automodapi:: screens.visualization
