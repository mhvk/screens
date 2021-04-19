*********************************************
Generating synthetic scintillation velocities
*********************************************

This tutorial describes how to generate synthetic scintillation velocities for
a pulsar in a circular orbit and a single one-dimensional screen, both with
known parameters.

Further explanations and derivations of the equations seen here can be found in
`Marten's scintillometry page
<http://www.astro.utoronto.ca/~mhvk/scintillometry.html#org5ea6450>`_
and Daniel Baker's "`Orbital Parameters and Distances
<https://eor.cita.utoronto.ca/images/4/44/DB_Orbital_Parameters.pdf>`_"
document. As in that document, the practical example here uses the parameter
values for the pulsar PSR J0437--4715 as studied by `Reardon et al. (2020)
<https://ui.adsabs.harvard.edu/abs/2020ApJ...904..104R/abstract>`_.

The combined codeblocks in this tutorial can be downloaded as a Python script
and as a Jupyter notebook:

:Python script:
    :jupyter-download:script:`gen_velocities.py <gen_velocities>`
:Jupyter notebook:
    :jupyter-download:notebook:`gen_velocities.ipynb <gen_velocities>`

Preliminaries
=============

Imports.

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt

    from astropy import units as u
    from astropy import constants as const

    from astropy.time import Time
    from astropy.coordinates import SkyCoord, SkyOffsetFrame, EarthLocation

    from astropy.visualization import quantity_support, time_support

Set up support for plotting astropy's
:py:class:`~astropy.units.quantity.Quantity` and :py:class:`~astropy.time.Time`
objects, and make sure that the output of plotting commands is displayed inline
(i.e., directly below the code cell that produced it).

.. jupyter-execute::

    quantity_support()
    time_support(format='iso')

    %matplotlib inline

Initializing parameters
=======================

Set the parameters of the pulsar system:

.. list-table::
    :widths: 2 1 2
    :header-rows: 1

    * - Parameter
      - Symbol
      - Remarks

    * - **Coordinates of the pulsar system**
      -  
      -

    * - declination
      - :math:`\delta`
      -

    * - right ascension
      - :math:`\alpha`
      -

    * - proper motion in declination
      - :math:`\mu_\delta`
      -

    * - proper motion in right ascension
      - :math:`\mu_{\alpha\ast}`
      - including the :math:`\cos(\delta)` term

    * - **Orbital elements of the pulsar binary**
      -  
      -
    
    * - binary period
      - :math:`P_\mathrm{b}`
      - 

    * - projected semi-major axis
      - :math:`a_\mathrm{p} \sin( i_\mathrm{p} )`
      -

    * - orbital inclination
      - :math:`i_\mathrm{p}`
      - :math:`0^\circ \leq i_\mathrm{p} < 180^\circ`,
        with :math:`i_\mathrm{p} = 90^\circ` for an edge-on orbit;
        for :math:`i_\mathrm{p} < 90^\circ`, the binary rotates
        counterclockwise on the sky (from north through east),
        for :math:`i_\mathrm{p} \geq 90^\circ` it rotates clockwise

    * - longitude of ascending node
      - :math:`\Omega_\mathrm{p}`
      - measured from the celestial north through east;
        :math:`0^\circ \leq \Omega_\mathrm{p} < 360^\circ`

    * - time of ascending node
      - :math:`T_\mathrm{asc,p}`
      -

    * - **Further parameters**
      -  
      -

    * - distance to the pulsar system
      - :math:`d_\mathrm{p}`
      -

    * - distance to the screen
      - :math:`d_\mathrm{s}`
      - from Earth

    * - position angle of the lensed images
      - :math:`\Omega_\mathrm{s}`
      - measured from the celestial north, through east, to the line of images
        formed by the lens;
        :math:`0^\circ \leq \Omega_\mathrm{s} < 360^\circ`;
        note that this angle has a :math:`180^\circ` ambiguity

    * - velocity of the lens
      - :math:`v_\mathrm{lens}`
      - component along the screen direction

.. jupyter-execute::

    p_b = 5.7410459 * u.day
    asini_p = 3.3667144 * const.c * u.s
    i_p = 137.56 * u.deg
    omega_p = 207. * u.deg
    t0_p = Time(54501.4671, format='mjd', scale='tdb')

    d_p = 156.79 * u.pc
    d_s = 90.6 * u.pc
    omega_s = 134.6 * u.deg
    v_lens = -31.9 * u.km / u.s

The coordinates should be placed directly in a
:py:class:`~astropy.coordinates.SkyCoord` object, that includes the pulsar
system's position on the sky, its distance, and its proper motion.

.. jupyter-execute::

    psr_coord = SkyCoord('04h37m15.99744s -47d15m09.7170s',
                         distance=d_p,
                         pm_ra_cosdec=121.4385 * u.mas / u.yr,
                         pm_dec=-71.4754 * u.mas / u.yr)

Calculate some derived quantities:

.. list-table::
    :widths: 2 1
    :header-rows: 1

    * - Parameter
      - Equation

    * - pulsar's radial-velocity amplitude
      - 
        .. math::
            
            K_\mathrm{p} = \frac{ 2 \pi a_\mathrm{p} \sin( i_\mathrm{p} ) }
                                { P_\mathrm{b} }

    * - fractional distance to the screen (from the pulsar)
      - 
        .. math::
            
            s = 1 - \frac{ d_\mathrm{s} }{ d_\mathrm{p} }

    * - effective distance
      - 
        .. math::
        
            d_\mathrm{eff} = \frac{ 1 - s }{ s } d_\mathrm{p}

    * - angle from the lens to the pulsar orbit's line of nodes
      - 
        .. math::
        
            \Delta\Omega_\mathrm{p} = \Omega_\mathrm{s} - \Omega_\mathrm{p}

.. jupyter-execute::

    k_p = 2.*np.pi * asini_p / p_b

    s = 1 - d_s / d_p
    d_eff = d_p * d_s / (d_p - d_s)

    delta_omega_p = omega_s - omega_p

Define a grid of observing times :math:`t` for which you want to calculate
velocities using a :py:class:`~astropy.time.Time` object.

.. jupyter-execute::

    t_mjd = np.arange(55000., 55700., 0.25)
    t = Time(t_mjd, format='mjd', scale='utc')

The lens frame
==============

Make a :py:class:`~astropy.coordinates.SkyOffsetFrame` centered on the pulsar
system, rotated to the one-dimensional lens.

.. jupyter-execute::

    lens_frame = SkyOffsetFrame(origin=psr_coord, rotation=omega_s)

On its own, ``SkyOffsetFrame(origin=psr_coord)`` creates a spherical frame with
its primary direction pointing along the line of sight, latitude in the
direction of Dec, and longitude in the direction of RA. By passing the argument
``rotation=omega_s``, the longitude and latitude dimensions rotate so longitude
is perpedicular to the lens and latitude parallel to the lens. When converting
positions or velocities in this frame to cartesian representation, the x-axis
will point along the line of sight, the y-axis perpendicular to the screen, and
the z-axis parallel to the screen (in the direction of its motion). Hence, we
need to compute the cartesian z-component of velocities in ``lens_frame``.

Calculating effective velocities
================================

There are several components of the effective velocity that can be computed
separately:

.. list-table::
    :widths: 2 1
    :header-rows: 1

    * - Velocity component
      - Symbol
    * - Earth's velocity as a function of time
      - :math:`v_\mathrm{E}( t )`
    * - pulsar's orbital velocity as a function of time
      - :math:`v_\mathrm{p,orb}( t )`
    * - pulsar systemic velocity (corresponding to the proper motion)
      - :math:`v_\mathrm{p,sys}`
    * - velocity of the lens (known in this example)
      - :math:`v_\mathrm{lens}`

All these refer to the component of the velocity along the line of images
formed by the lens.

Earth's velocity
----------------

To obtain Earth's velocity in the lens frame, first generate a location on
Earth's surface using the :py:class:`~astropy.coordinates.EarthLocation` class
(in this case the location of the Parkes radio telescope). This class has the
:py:meth:`~astropy.coordinates.EarthLocation.get_gcrs` method, which returns
positions (with respect to the centre of the Earth) as a function of time.
These are transformed into the lens frame using the
:py:meth:`~astropy.coordinates.BaseCoordinateFrame.transform_to` method.
Velocities can then be extracted using the
:py:attr:`~astropy.coordinates.BaseCoordinateFrame.velocity` attribute, and
finally :py:attr:`~astropy.coordinates.CartesianDifferential.d_z` isolates the
z-component of the velocity (in the direction of the screen).

.. jupyter-execute::

    earth_loc = EarthLocation('148°15′47″E', '32°59′52″S')
    
    v_earth = earth_loc.get_gcrs(t).transform_to(lens_frame).velocity.d_z

Pulsar's orbital velocity
-------------------------

Compute the pulsar's orbital velocity projected onto the screen
    
.. math::

    v_\mathrm{p,orb} = - \frac{ K_\mathrm{p} }{ \sin( i_\mathrm{p} ) }
                         \left[ \cos( i_\mathrm{p} )
                                \sin( \Delta\Omega_\mathrm{p} )
                                \cos( \phi_\mathrm{p} )
                              - \cos( \Delta\Omega_\mathrm{p} )
                                \sin( \phi_\mathrm{p} )
                         \right].

Here, :math:`\phi_\mathrm{p}( t )` is the phase of pulsar orbit as measured
from its ascending node.

.. jupyter-execute::

    ph_p = ((t - t0_p) / p_b).to(u.dimensionless_unscaled) * u.cycle

    v_p_orb = (-k_p / np.sin(i_p)
               * (np.cos(i_p) * np.sin(delta_omega_p) * np.cos(ph_p)
                              - np.cos(delta_omega_p) * np.sin(ph_p)))

Pulsar systemic velocity
------------------------

The pulsar systemic velocity projected onto the screen is given by

.. math::

    v_\mathrm{p,sys} \simeq d_\mathrm{p}
                              \left[ \mu_{\alpha\ast} \sin( \Omega_\mathrm{s} )
                                         + \mu_\delta \cos( \Omega_\mathrm{s} )
                              \right].

This can be computed manually, but it can also be retrieved from the
:py:class:`~astropy.coordinates.SkyCoord` of the pulsar system (which contains
the system's proper motion) by transforming it to ``lens_frame``.

.. jupyter-execute::
    
    v_p_sys = psr_coord.transform_to(lens_frame).velocity.d_z

Effective velocity
------------------

Combine the velocities of the pulsar, Earth, and the lens into the effective
velocity

.. math::

    v_\mathrm{eff} = \frac{1}{s} v_\mathrm{lens}
                     - \frac{1 - s}{s} ( v_\mathrm{p,orb} + v_\mathrm{p,sys} )
                     - v_\mathrm{E}

.. jupyter-execute::
    
    v_eff = 1. / s * v_lens - ((1 - s) / s) * (v_p_orb + v_p_sys) - v_earth

Have a look at the contribution of each of the terms to the effective velocity.

.. jupyter-execute::

    plt.figure(figsize=(12., 6.))
    
    plt.plot(t, - v_earth)
    plt.plot(t, - ((1 - s) / s) * v_p_orb)
    plt.plot(t[::len(t)-1], 1. / s * v_lens * [1, 1])
    plt.plot(t[::len(t)-1], - ((1 - s) / s) * v_p_sys * [1, 1])
    plt.plot(t, v_eff)
    plt.legend([r'$-v_\mathrm{E}$',
                r'$-((1 - s) / s) v_\mathrm{p,orb}$',
                r'$(1. / s) v_\mathrm{lens}$',
                r'$-((1 - s) / s) v_\mathrm{p,sys}$',
                r'$v_\mathrm{eff}$'],
               bbox_to_anchor=(1.04,1), loc='upper left')
    plt.xlim(t[0], t[-1])
    plt.ylabel(r'velocity (km/s)')
    
    plt.show()

Curvature and scaled effective velocity
=======================================

The curvature :math:`\eta` can be computed from the effective velocity
according to

.. math::
    
    \eta = \frac{ \lambda^2 d_\mathrm{eff} }{ 2 c v_\mathrm{eff}^2 },

where :math:`\lambda` is the observing wavelength and :math:`c` is the speed of
light.

.. jupyter-execute::

    lambda_obs = (1400. * u.MHz).to(u.m, equivalencies=u.spectral())

    eta = lambda_obs**2 * d_eff / (2. * const.c * v_eff**2)

Have a look at the curvature at a function of time.

.. jupyter-execute::

    plt.figure(figsize=(12., 6.))
    
    plt.plot(t, eta.to(u.s**3))
    plt.xlim(t[0], t[-1])
    plt.ylabel(r'curvature $\eta$ (s$^3$)')
    
    plt.show()

Since :math:`v_\mathrm{eff}` can be arbitrarily close to zero (letting
:math:`\eta` blow up), curvature has a strongly non-uniform prior probability
distribution (as can be seen from the modulation in amplitude in the figure
above). For this reason, it is sometimes better to fit for the curvature of the
secondary spectrum parabola in a space of "scaled effective velocity"

.. math::
    
    \frac{ \lambda }{ \sqrt{ 2 \eta c } }
      = \frac{  \left| v_\mathrm{eff} \right| }{ \sqrt{ d_\mathrm{eff} } }

.. jupyter-execute::
    
    dveff = np.abs(v_eff) / np.sqrt(d_eff)
    
Plot this quantity as function of time.

.. jupyter-execute::

    plt.figure(figsize=(12., 6.))
    
    plt.plot(t, dveff)
    plt.xlim(t[0], t[-1])
    dveff_lbl = (r'scaled effective velocity '
                 r'$\frac{ | v_\mathrm{eff} | }{ \sqrt{ d_\mathrm{eff} } }$ '
                 r'$\left( \frac{\mathrm{km/s}}{\sqrt{\mathrm{pc}}} \right)$')
    plt.ylabel(dveff_lbl)
    
    plt.show()
    
To visualize the modulation in scintillation velocity caused by both the
pulsar's orbital motion and that of the Earth, we can make a 2D phase fold of
the data.

.. jupyter-execute::

    plt.figure(figsize=(10., 6.))

    plt.hexbin(t.jyear % 1., ph_p.value % 1., C=dveff.value,
               reduce_C_function=np.median, gridsize=19)
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.xlabel('Earth orbit phase (from Jan 1st)')
    plt.ylabel('Pulsar orbit phase (from ascending node)')
    cbar = plt.colorbar()
    cbar.set_label(dveff_lbl)

    plt.show()

