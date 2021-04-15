********************************
Fitting scintillation velocities
********************************

This tutorial describes how to fit a phenomenological model to a time series of
scintillation velocities for a pulsar in a circular orbit and a single
one-dimensional screen. It requires a file containing synthetic data:
:download:`fake-data-J0437.npz <../fake-data-J0437.npz>`

Further explanations and derivations of the equations seen here can be found in
`Marten's scintillometry page
<http://www.astro.utoronto.ca/~mhvk/scintillometry.html#org5ea6450>`_
and Daniel Baker's "`Orbital Parameters and Distances
<https://eor.cita.utoronto.ca/images/4/44/DB_Orbital_Parameters.pdf>`_"
document. As in that document, the practical example here uses the parameter
values for the pulsar PSR J0437-4715 as studied by `Reardon et al. (2020)
<https://ui.adsabs.harvard.edu/abs/2020ApJ...904..104R/abstract>`_.

The combined codeblocks in this tutorial can be downloaded as a Python script
and as a Jupyter notebook:

:Python script:
    :jupyter-download:script:`fit_velocities.py <fit_velocities>`
:Jupyter notebook:
    :jupyter-download:notebook:`fit_velocities.ipynb <fit_velocities>`

Preliminaries
=============

Imports.

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    # Note: matplotlib.colors.CenteredNorm requires matplotlib version >= 3.4.0

    from astropy import units as u

    from astropy.time import Time
    from astropy.coordinates import SkyCoord

    from astropy.visualization import quantity_support, time_support

    from scipy.optimize import minimize

Set up support for plotting Astropy's
:py:class:`~astropy.units.quantity.Quantity` and :py:class:`~astropy.time.Time`
objects, and make sure that the output of plotting commands is displayed inline
(i.e., directly below the code cell that produced it).

.. jupyter-execute::

    quantity_support()
    time_support(format='iso')

    %matplotlib inline

Define some line and marker properties to easily give the model and the data a
consistent appearance throughout the notebook. Also assign a rather long label
string to a variable, so it doesn't need to be rewritten for each plot, and
define a little dictionary of arguments to move plot titles inside their axes.

.. jupyter-execute::
    
    obs_style = {
        'linestyle': 'none',
        'color': 'grey',
        'marker': 'o',
        'markerfacecolor': 'none'
    }

    mdl_style = {
        'linestyle': '-',
        'linewidth': 2,
        'color': 'C0'
    }

    dveff_lbl = (r'scaled effective velocity '
                 r'$\frac{ | v_\mathrm{eff} | }{ \sqrt{ d_\mathrm{eff} } }$ '
                 r'$\left( \frac{\mathrm{km/s}}{\sqrt{\mathrm{pc}}} \right)$')
        
    title_kwargs = {
        'loc': 'left', 
        'x': 0.01,
        'y': 1.0,
        'pad': -14
    }

Set known parameters
====================

Set the pulsar's orbital period :math:`P_\mathrm{b}` and time of ascending node
:math:`T_\mathrm{asc,p}`, which are known from pulsar timing.

.. jupyter-execute::
    
    p_b = 5.7410459 * u.day
    t_asc_p = Time(54501.4671, format='mjd')

Set the Earth's orbital period :math:`P_\mathrm{E}` and derive its time of
ascending node :math:`T_\mathrm{asc,E}` from the pulsar's coordinates.

.. jupyter-execute::

    p_e = 1. * u.yr
    t_equinox = Time('2005-03-21 12:33', format='iso', scale='utc')

    psr_coord = SkyCoord('04h37m15.99744s -47d15m09.7170s')

    psr_coord_eclip = psr_coord.barycentricmeanecliptic
    ascnod_eclip_lon = psr_coord_eclip.lon + 90.*u.deg
    
    t_asc_e = t_equinox + ascnod_eclip_lon.cycle * p_e

Load and inspect the data
=========================

Load the data (available for download here:
:download:`fake-data-J0437.npz <../fake-data-J0437.npz>`)
and convert the NumPy arrays that are stored in the file to Astropy
:py:class:`~astropy.time.Time` and :py:class:`~astropy.units.quantity.Quantity`
objects.

.. jupyter-execute::

    data = np.load('fake-data-J0437.npz')

    t_obs = Time(data['t_mjd'], format='mjd', scale='utc')
    dveff_obs = data['dveff_obs'] * u.km/u.s/u.pc**0.5
    dveff_err = data['dveff_err'] * u.km/u.s/u.pc**0.5

We can now precompute the orbital phases (measured from the ascending node) of
the pulsar, :math:`\phi_\mathrm{p}(t)`, and the Earth,
:math:`\phi_\mathrm{E}(t)`, for the observation times.

.. math::

    \phi_\mathrm{p}(t) = \frac{ t - T_\mathrm{asc,p} }{ P_\mathrm{b} }
    \qquad \mathrm{and} \qquad
    \phi_\mathrm{E}(t) = \frac{ t - T_\mathrm{asc,E} }{ P_\mathrm{E} }

.. jupyter-execute::

    ph_p_obs = ((t_obs - t_asc_p) / p_b).to(u.dimensionless_unscaled) * u.cycle
    ph_e_obs = ((t_obs - t_asc_e) / p_e).to(u.dimensionless_unscaled) * u.cycle

Let's have a look at all the data.

.. jupyter-execute::
    
    plt.figure(figsize=(12., 5.))

    plt.errorbar(t_obs.jyear, dveff_obs, yerr=dveff_err, **obs_style,
                 alpha=0.3)
    
    plt.xlim(t_obs[0].jyear, t_obs[-1].jyear)

    plt.xlabel('time')
    plt.ylabel(dveff_lbl)

    plt.show()

Because the pulsar's orbital period is much shorter than the baseline of the
observation, it cannot be discerned in the raw time series. To visualize the
modulations in scintillation velocity caused by the pulsar's orbital motion and
that of the Earth in one plot, one should make a 2D phase fold of the dataset.

.. jupyter-execute::

    plt.figure(figsize=(10., 6.))

    plt.hexbin(ph_e_obs.value % 1., ph_p_obs.value % 1., C=dveff_obs.value,
               reduce_C_function=np.median, gridsize=19)

    plt.xlim(0., 1.)
    plt.ylim(0., 1.)

    plt.xlabel('Earth orbit phase')
    plt.ylabel('Pulsar orbit phase')

    cbar = plt.colorbar()
    cbar.set_label(dveff_lbl)

The phenomenological model
==========================

There are many possible ways of writing the formula for scaled effective
velocity, all with their advantages and disadvantages. Here, we model the
velocities as the sum of two sinusoids with known periods (one for the pulsar's
orbital modulation and one for the Earth's) and a constant offset (due to the
pulsar's systemic velocity and the motion of the lens). We then need to take
the absolute value of this sum, because measuring the curvature of a parabola
in a secondary spectrum only constrains the square of the effective velocity.
Thus, the model is given by

.. math::

    \frac{ \left| v_\mathrm{eff} \right| }{ \sqrt{d_\mathrm{eff}} }
      = \left| A_\mathrm{p} \sin( \phi_\mathrm{p} - \xi_\mathrm{p} )
             + A_\mathrm{E} \sin( \phi_\mathrm{E} - \xi_\mathrm{E} ) + C
        \right|.

There are five free parameters: the amplitudes of the pulsar's and the Earth's
orbital scaled-effective-velocity modulation, :math:`A_\mathrm{p}` and
:math:`A_\mathrm{E}`, their phase offsets, :math:`\xi_\mathrm{p}` and
:math:`\xi_\mathrm{E}`, and a constant scaled-effective-velocity offset,
:math:`C`. In principle, the amplitudes should be non-negative
(:math:`A_\mathrm{p} \geq 0`, :math:`A_\mathrm{E} \geq 0`). In practice,
however, when fitting yields a negative amplitude, this can be resolved by
flipping the sign of the amplitude :math:`A` and rotating the corresponding
phase offset :math:`\xi` by :math:`180^\circ`.

This formulation of the scaled-effective-velocity equation has the advantage
that it is clear how its free parameters affect the model in data space (hence,
when fitting the model to data, it is clear how the fit can be improved by
changing the the values of the free parameters). However, it obscures how the
model depends on the physical parameters of interest. A
:doc:`follow-up tutorial <extract_phys_pars>` describes how the free parameters
in this equation are related to the physical parameters of the system.

When putting the model equation into a Python function, it is useful to keep
the modulus operation separate from the rest of the model. This will allow us
to model the individual components of the scaled effective velocity separately.

.. jupyter-execute::

    def model_dveff_signed(pars, t):
    
        ph_p = ((t - t_asc_p) / p_b).to(u.dimensionless_unscaled) * u.cycle
        ph_e = ((t - t_asc_e) / p_e).to(u.dimensionless_unscaled) * u.cycle
        
        dveff_p = pars['amp_p'] * np.sin(ph_p - pars['xi_p'])
        dveff_e = pars['amp_e'] * np.sin(ph_e - pars['xi_e'])
        
        dveff = dveff_p + dveff_e + pars['dveff_c']
    
        return (dveff).to(u.km/u.s/u.pc**0.5)
    
    def model_dveff_abs(pars, t):
        dveff_signed = model_dveff_signed(pars, t)
        return np.abs(dveff_signed)

Note that the first argument of these functions, `pars`, should be a dictionary
containing the free parameters as :py:class:`~astropy.units.quantity.Quantity`
objects; their second argument, `t`, should be a :py:class:`~astropy.time.Time`
object containing the times at which the model should be evaluated.

Estimating the free-parameter values by eye
===========================================

When fitting a model to data, it is helpful to understand the effect of varying
the different free parameters. One can, for example, start by evaluating the
model at some random point in free-parameter space and then explore the space
by varying the parameters one by one. In this case, however, the relation
between the free parameters and the model is fairly clear from the model
equation. Moreover, the (synthetic) data are of sufficient quality that we can
make rough estimates of the free-parameters values simply by looking at the
data.

The amplitudes :math:`A_\mathrm{p}` and :math:`A_\mathrm{E}` and the offset
:math:`C` can be estimated by eye from the time-series plot above:

- :math:`C` corresponds to the mean of the time series
  (around 15 km/s/pc\ :sup:`1/2`);
- :math:`A_\mathrm{E}` is the amplitude of the visible sinusoid
  (around 2 km/s/pc\ :sup:`1/2`);
- :math:`A_\mathrm{p}` is roughly the half-width of the band of data points
  that constitutes the visible sinusoid (around 1.5 km/s/pc\ :sup:`1/2`).

The phase offsets :math:`\xi_\mathrm{p}` and :math:`\xi_\mathrm{E}` are a bit
harder to estimate by eye, but the 2D phase fold of the dataset can be used for
this. For phase offsets
:math:`(\xi_\mathrm{E}, \xi_\mathrm{p}) = (0^\circ, 0^\circ)`, the 2D sinusoid
should peak at phases :math:`(0.25, 0.25)`. Since the peak in the plot seems to
be around :math:`(0.45, 0.45)`, we can estimate the phase offsets to be roughly
:math:`(\xi_\mathrm{E}, \xi_\mathrm{p}) \approx (60^\circ, 60^\circ)`.

To prepare the set of parameter values for use with our model functions, put
them in a dictionary with the appropriate keys.

.. jupyter-execute::

    pars_try = {
        'amp_p':     1.5 * u.km/u.s/u.pc**0.5,
        'xi_p':     60.  * u.deg,
        'amp_e':     2.  * u.km/u.s/u.pc**0.5,
        'xi_e':     60.  * u.deg,
        'dveff_c':  15.  * u.km/u.s/u.pc**0.5
    }

Visual model-data comparison
============================

To test if a set of parameter values yields a good fit to the data, we should
produce a few key model-data comparison figures. Since we will likely want to
repeat these tests for different instances of the model, we will put them in
Python functions that evaluate the model for a given set of parameter values
and generate the desired plots. The resulting functions are somewhat lengthy;
to avoid them interrupting the flow of the tutorial, they they are by default
hidden from view. The codeblocks with these functions can be expanded using the
**"Show function definition"** buttons.

The most straightforward way of model-data comparison is to overplot the model
on the data and show the residuals. Since the two orbital periods in the system
under investigation have very different timescales, we show two different
zooms of the time series: one in which the Earth's orbital modulation is
visible and one in which the pulsar's can be resolved. The observations are
relatively sparse compared to the pulsar's orbital period, so to make the
pulsar's orbit visible in the time series, we have to also evaluate the model
at a higher time resolution.

.. jupyter-execute::
    :hide-code:

    def visualize_model_full(pars):

        dveff_mdl = model_dveff_abs(pars, t_obs)
        dveff_res = dveff_obs - dveff_mdl

        tlim_long = [t_obs[0].mjd, t_obs[0].mjd + 3. * p_e.to_value(u.day)]
        tlim_zoom = [t_obs[0].mjd, t_obs[0].mjd + 5. * p_b.to_value(u.day)]

        t_mjd_many = np.arange(tlim_long[0], tlim_long[-1], 0.2)
        t_many = Time(t_mjd_many, format='mjd')

        dveff_mdl_many = model_dveff_abs(pars, t_many)

        plt.figure(figsize=(12., 9.))
        
        plt.subplots_adjust(wspace=0.1)

        ax1 = plt.subplot(221)
        plt.plot(t_many, dveff_mdl_many, **mdl_style, alpha=0.3)
        plt.errorbar(t_obs.mjd, dveff_obs, yerr=dveff_err, **obs_style,
                     alpha=0.3)
        plt.xlim(tlim_long)
        plt.title('full model', **title_kwargs)
        plt.xlabel('')
        plt.ylabel(dveff_lbl)

        ax2 = plt.subplot(223, sharex=ax1)
        plt.errorbar(t_obs.mjd, dveff_res, yerr=dveff_err, **obs_style,
                     alpha=0.3)
        plt.axhline(**mdl_style)
        plt.xlim(tlim_long)
        plt.title('residuals', **title_kwargs)
        plt.ylabel(dveff_lbl)

        ax3 = plt.subplot(222, sharey=ax1)
        plt.plot(t_many, dveff_mdl_many, **mdl_style)
        plt.errorbar(t_obs.mjd, dveff_obs, yerr=dveff_err, **obs_style)
        plt.xlim(tlim_zoom)
        plt.title('full model, zoom', **title_kwargs)
        plt.xlabel('')
        plt.ylabel(dveff_lbl)
        ax3.yaxis.set_label_position('right')
        ax3.yaxis.tick_right()

        ax4 = plt.subplot(224, sharex=ax3, sharey=ax2)
        plt.errorbar(t_obs.mjd, dveff_res, yerr=dveff_err, **obs_style)
        plt.axhline(**mdl_style)
        plt.xlim(tlim_zoom)
        plt.title('residuals, zoom', **title_kwargs)
        plt.ylabel(dveff_lbl)
        ax4.yaxis.set_label_position('right')
        ax4.yaxis.tick_right()

        plt.show()

.. jupyter-execute::

    visualize_model_full(pars_try)

Next, let's make plots in which the data is folded over the Earth's and the
pulsar's orbital period. To do this, it is necessary to generate the scaled-
effective-velocity terms due to Earth's orbit and the pulsar's orbit
separately. This can be achieved using the `model_dveff_signed()` function
(which does not include the modulus operation) and with specific parameters set
to zero. (When copying a dictionary of parameters, pay attention not to modify
the original dictionary.) A model of only the Earth's component can then be
compared with the data minus the remaining model components, and likewise for
the pulsar. For these plots to show a good agreement between data and model,
all model components need to be accurate.

.. jupyter-execute::
    :hide-code:

    def visualize_model_folded(pars):
        
        pars_earth = pars.copy()
        pars_earth['amp_p'] = 0. * u.km/u.s/u.pc**0.5
        pars_earth['dveff_c'] = 0. * u.km/u.s/u.pc**0.5
        dveff_mdl_earth = model_dveff_signed(pars_earth, t_obs)
        
        pars_psr = pars.copy()
        pars_psr['amp_e'] = 0. * u.km/u.s/u.pc**0.5
        pars_psr['dveff_c'] = 0. * u.km/u.s/u.pc**0.5
        dveff_mdl_psr = model_dveff_signed(pars_psr, t_obs)
        
        pars_const = pars.copy()
        pars_const['amp_e'] = 0. * u.km/u.s/u.pc**0.5
        pars_const['amp_p'] = 0. * u.km/u.s/u.pc**0.5
        dveff_mdl_const = model_dveff_signed(pars_const, t_obs)

        dveff_res_earth = dveff_obs - dveff_mdl_psr - dveff_mdl_const
        dveff_res_psr = dveff_obs - dveff_mdl_earth - dveff_mdl_const

        plt.figure(figsize=(12., 5.))

        plt.subplots_adjust(wspace=0.1)
        
        ax1 = plt.subplot(121)
        idx_e = np.argsort(ph_e_obs.value % 1.)
        plt.plot(ph_e_obs[idx_e].value % 1., dveff_mdl_earth[idx_e],
                 **mdl_style)
        plt.errorbar(ph_e_obs.value % 1., dveff_res_earth, yerr=dveff_err,
                     **obs_style, alpha=0.2, zorder=-3)
        plt.xlim(0., 1.)
        plt.title('Earth motion', **title_kwargs)
        plt.xlabel('Earth orbital phase')
        plt.ylabel(dveff_lbl)
        
        ax2 = plt.subplot(122, sharey=ax1)
        idx_p = np.argsort(ph_p_obs.value % 1.)
        plt.plot(ph_p_obs[idx_p].value % 1., dveff_mdl_psr[idx_p], **mdl_style)
        plt.errorbar(ph_p_obs.value % 1., dveff_res_psr, yerr=dveff_err,
                     **obs_style, alpha=0.2, zorder=-3)
        plt.xlim(0., 1.)
        plt.title('Pulsar motion', **title_kwargs)
        plt.xlabel('Pulsar orbital phase')
        plt.ylabel(dveff_lbl)
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()

        plt.show()

.. jupyter-execute::

    visualize_model_folded(pars_try)


Finally, the 2D phase fold of the data can be compared with the same 2D phase
fold of the full model.

.. jupyter-execute::
    :hide-code:

    def visualize_model_fold2d(pars):

        dveff_mdl = model_dveff_abs(pars, t_obs)
        dveff_res = dveff_obs - dveff_mdl

        plt.figure(figsize=(12., 4.))

        gridsize = 19
        labelpad = 16
            
        plt.subplot(131)
        plt.hexbin(ph_e_obs.value % 1., ph_p_obs.value % 1., C=dveff_obs.value,
                   reduce_C_function=np.median, gridsize=gridsize)
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.xlabel('Earth orbit phase')
        plt.ylabel('Pulsar orbit phase')
        plt.title('data', **title_kwargs,
                  fontdict={'color': 'w', 'fontweight': 'bold'})
        cbar = plt.colorbar(location='top')
        cbar.ax.invert_xaxis()
        cbar.set_label(dveff_lbl, labelpad=labelpad)
        
        plt.subplot(132)
        plt.hexbin(ph_e_obs.value % 1., ph_p_obs.value % 1., C=dveff_mdl.value,
                   reduce_C_function=np.median, gridsize=gridsize)
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.xlabel('Earth orbit phase')
        plt.title('model', **title_kwargs,
                fontdict={'color': 'w', 'fontweight': 'bold'})
        cbar = plt.colorbar(location='top')
        cbar.ax.invert_xaxis()
        cbar.set_label(dveff_lbl, labelpad=labelpad)
        
        plt.subplot(133)
        plt.hexbin(ph_e_obs.value % 1., ph_p_obs.value % 1., C=dveff_res.value,
                   reduce_C_function=np.median, gridsize=gridsize,
                   norm=mcolors.CenteredNorm(), cmap='coolwarm')
        # Note: CenteredNorm requires matplotlib version >= 3.4.0
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.xlabel('Earth orbit phase')
        plt.title('residuals', **title_kwargs,
                  fontdict={'color': 'k', 'fontweight': 'bold'})
        cbar = plt.colorbar(location='top')
        cbar.ax.invert_xaxis()
        cbar.set_label(dveff_lbl, labelpad=labelpad)

        plt.show()

.. jupyter-execute::

    visualize_model_fold2d(pars_try)


Quantifying the goodness of fit
===============================

To quantify the goodness of fit of a given instance of the model to the data,
we will compute its :math:`\chi^2` statistic.

.. jupyter-execute::

    def get_chi2(mdl, obs, err):
        chi2 = np.sum(((obs - mdl) / err)**2)
        return chi2

One can now evaluate the model for a given set of parameter values and compute
the corresponding goodness of fit. It may also be useful to calculate the
reduced :math:`\chi^2` statistic.

.. jupyter-execute::

    dveff_mdl = model_dveff_abs(pars_try, t_obs)
    chi2 = get_chi2(dveff_mdl, dveff_obs, dveff_err)
    print(f'chi2     {chi2:8.2f}')

    ndof = len(t_obs) - len(pars_try)
    chi2_red = chi2 / ndof
    print(f'chi2_red {chi2_red:8.2f}')

Algorithmic maximum likelihood estimation
=========================================

To find the parameter values that give the maximum likelihood, we can use
:py:func:`scipy.optimize.minimize` to find a local minimum in :math:`\chi^2`
given an initial guess. This function needs the free parameters as an array of
(unitless) floats, so let's first make two functions for converting our set of
free parameters from a dictionary of Astropy
:py:class:`~astropy.units.quantity.Quantity` objects to a NumPy
:py:class:`~numpy.ndarray` and the other way round.

.. jupyter-execute::

    def pars_dict2array(pars_dict):
        pars_array = np.array([
            pars_dict['amp_p'].to_value(u.km/u.s/u.pc**0.5),
            pars_dict['xi_p'].to_value(u.rad),
            pars_dict['amp_e'].to_value(u.km/u.s/u.pc**0.5),
            pars_dict['xi_e'].to_value(u.rad),
            pars_dict['dveff_c'].to_value(u.km/u.s/u.pc**0.5),
        ])
        return pars_array

    def pars_array2dict(pars_array):
        pars_dict = {
            'amp_p': pars_array[0] * u.km/u.s/u.pc**0.5,
            'xi_p': (pars_array[1] * u.rad).to(u.deg),
            'amp_e': pars_array[2] * u.km/u.s/u.pc**0.5,
            'xi_e': (pars_array[3] * u.rad).to(u.deg),
            'dveff_c': pars_array[4] * u.km/u.s/u.pc**0.5,
        }
        return pars_dict


Next, define a wrapper function that :py:func:`~scipy.optimize.minimize` can
work with. This function needs to convert the free parameters into the
dictionary expected by the model function, call the model function, and compute
the :math:`\chi^2` statistic. To comply with the call signature of
:py:func:`~scipy.optimize.minimize`, its first argument should be the array of
free parameters (see below).

.. jupyter-execute::

    def fit_wrapper(pars_float_array, t_obs, dveff_obs, dveff_err):
        pars_quantity_dict = pars_array2dict(pars_float_array)
        dveff_mdl = model_dveff_abs(pars_quantity_dict, t_obs)
        chi2 = get_chi2(dveff_mdl, dveff_obs, dveff_err)
        return chi2

As an initial guess we use the set of free-parameter values tried earlier,
converted to the array format expected by :py:func:`~scipy.optimize.minimize`.

.. jupyter-execute::

    init_guess = pars_dict2array(pars_try)

Everything is now ready to run :py:func:`~scipy.optimize.minimize`. It may be
useful to review its call signature:

- The first argument is the function to be minimized, whose first argument in
  turn needs to be the array of free parameters.
- The second argument is an array of free-parameter values that serve as
  an initial guess. The length of this array sets the number of independent
  variables.
- The `args` argument is a tuple of extra arguments that are passed to the
  function to be minimized (i.e., in addition to the array of free parameters).
- The `method` argument specifies which solver/algorithm is used to do the
  minimization. For this problem, the `Nelder--Mead method
  <https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method>`_ seems to work.
- The `options` argument is a dictionary of options for the solver, and its
  `maxiter` key specifies the maximum number of iterations the algorithm is
  allowed to do, after which the minimization is terminated.

.. jupyter-execute::
    
    fit = minimize(fit_wrapper, init_guess, args=(t_obs, dveff_obs, dveff_err),
                   method='Nelder-Mead', options={'maxiter': 10000})

The :py:func:`~scipy.optimize.minimize` function returns a
:py:class:`scipy.optimize.OptimizeResult` object, which contains a bunch of
additional information about the fitting process. The actual solution of the
minimization is contained as an array in its `x` attribute. To make the result
more readable and ready as input for our model functions, we convert this array
into a dictionary of Astropy :py:class:`~astropy.units.quantity.Quantity`
objects using the `pars_array2dict()` function defined earlier.

.. jupyter-execute::

    pars_fit = pars_array2dict(fit.x)
        
    for par_name in pars_fit:
        print(f'{par_name:8s} {pars_fit[par_name]:8.2f}')

How these fitted free parameters can be converted to the physical parameters of
interest is covered in a :doc:`follow-up tutorial <extract_phys_pars>`.

Let's find out if the :math:`\chi^2` minimization worked.

.. jupyter-execute::

    dveff_mdl = model_dveff_abs(pars_fit, t_obs)
    chi2 = get_chi2(dveff_mdl, dveff_obs, dveff_err)
    chi2_red = chi2 / ndof

    print(f'\nchi2     {chi2:8.2f}'
          f'\nchi2_red {chi2_red:8.2f}')

Finally, to check if the fitting worked well, it is also important to visually
inspect the solution. This can be done using the visualization functions we
made earlier:

.. jupyter-execute::

    visualize_model_full(pars_fit)
    visualize_model_folded(pars_fit)
    visualize_model_fold2d(pars_fit)
