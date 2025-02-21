.. include:: ../README.rst

.. Sphinx requires that all source document files it finds are part of some
.. table of contents using the `toctree` directive. Without this hidden ToC,
.. sphinx gives a warning for all `.rst` files in the documentation:
.. `WARNING: document isn't included in any toctree`

.. toctree::
    :hidden:

    background/simple_examples
    background/multiple_screens
    background/velocities
    background/glossary

    tutorials/single_screen
    tutorials/different_transforms
    tutorials/screen1d
    tutorials/two_screens
    tutorials/vlbi_simulation

    tutorials/gen_velocities
    tutorials/fit_velocities
    tutorials/infer_phys_pars
    tutorials/error_analysis


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
* :doc:`tutorials/different_transforms`:
  how the nu-t transform preserves features best in wide-band data
* :doc:`tutorials/screen1d`:
  generate synthetic data and visualize the system (for a single screen)
* :doc:`tutorials/two_screens`:
  how to use Screen1D for radiation scattered by multiple screens
* :doc:`tutorials/vlbi_simulation`:
  how to use the Screen1D class to generate VLBI data


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
* :doc:`tutorials/error_analysis`:
  propagate uncertainties from fitting parameters to physical parameters


Reference/API
=============

.. automodapi:: screens
.. automodapi:: screens.screen
.. automodapi:: screens.fields
.. automodapi:: screens.dynspec
.. automodapi:: screens.conjspec
.. automodapi:: screens.remap
.. automodapi:: screens.modeling
.. automodapi:: screens.visualization
