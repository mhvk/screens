*****************************
Inferring physical parameters
*****************************

This tutorial describes how to infer physical parameters from the free
parameters of a phenomenological model for scintillation velocities. The
tutorial builds upon a :doc:`preceding tutorial <fit_velocities>` in which such
a phenomenological model is fit to a time series of scintillation velocities.

Further explanations and derivations of the equations seen here can be found in
`Marten's scintillometry page
<http://www.astro.utoronto.ca/~mhvk/scintillometry.html#org5ea6450>`_
and Daniel Baker's "`Orbital Parameters and Distances
<https://eor.cita.utoronto.ca/images/4/44/DB_Orbital_Parameters.pdf>`_"
document. As in that document, the practical example here uses the parameter
values for the pulsar PSR J0437-4715 as derived by `Reardon et al. (2020)
<https://ui.adsabs.harvard.edu/abs/2020ApJ...904..104R/abstract>`_.

The combined codeblocks in this tutorial can be downloaded as a Python script
and as a Jupyter notebook:

:Python script:
    :jupyter-download:script:`infer_phys_pars.py <infer_phys_pars>`
:Jupyter notebook:
    :jupyter-download:notebook:`infer_phys_pars.ipynb <infer_phys_pars>`

Preliminaries
=============

Imports.

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt

    from astropy import units as u
    from astropy import constants as const

    from astropy.coordinates import Angle, SkyCoord, BarycentricMeanEcliptic

    from astropy.visualization import quantity_support

Set up support for plotting astropy's
:py:class:`~astropy.units.quantity.Quantity` objects, and make sure that the
output of plotting commands is displayed inline (i.e., directly below the code
cell that produced it).

.. jupyter-execute::

    quantity_support()

    %matplotlib inline

The model parameters
====================

The phenomenological model used to fit the scaled effective velocities consists
of two sinusoids (with known periods) and an offset:

.. math::

    \frac{ \left| v_\mathrm{eff} \right| }{ \sqrt{d_\mathrm{eff}} }
      = \left| A_\mathrm{p} \sin( \phi_\mathrm{p} - \xi_\mathrm{p} )
             + A_\mathrm{E} \sin( \phi_\mathrm{E} - \xi_\mathrm{E} ) + C
        \right|.

The free parameters in this equation are the amplitudes of the pulsar's and the
Earth's orbital scaled-effective-velocity modulation :math:`A_\mathrm{p}` and
:math:`A_\mathrm{E}` (assumed to be non-negative, :math:`A_\mathrm{p} \geq 0`,
:math:`A_\mathrm{E} \geq 0`), their phase offsets :math:`\xi_\mathrm{p}` and
:math:`\xi_\mathrm{E}`, and a constant scaled-effective-velocity offset
:math:`C`. The model parameters are related to the physical parameters of the
system according to

.. math::

    A_\mathrm{p} &= \frac{ \sqrt{ d_\mathrm{eff} } }{ d_\mathrm{p} }
                    \frac{ K_\mathrm{p} }{ \sin( i_\mathrm{p} ) }
                    b_\mathrm{p},

    A_\mathrm{E} &= \frac{ v_\mathrm{orb,E} }{ \sqrt{ d_\mathrm{eff} } }
                    b_\mathrm{E},

    \tan( \xi_\mathrm{p} ) &= \tan( \Delta\Omega_\mathrm{p} )
                              \cos( i_\mathrm{p} ),

    \tan( \xi_\mathrm{E} ) &= \tan( \Delta\Omega_\mathrm{E} )
                              \cos( i_\mathrm{E} ),

    C &= \pm \frac{ v_\mathrm{lens} }{ s \sqrt{ d_\mathrm{eff} } }
         \mp \frac{ v_\mathrm{p,sys,eff} }{ \sqrt{ d_\mathrm{eff} } }.

Here, :math:`\Delta\Omega_\mathrm{p}` and :math:`\Delta\Omega_\mathrm{E}`
denote the angles from the position angle of the screen to the longitude of
ascending node of the orbit of the pulsar and the Earth, respectively, i.e.,

.. math::

    \Delta\Omega_\mathrm{p} = \Omega_\mathrm{s} - \Omega_\mathrm{p},
    \qquad
    \Delta\Omega_\mathrm{E} = \Omega_\mathrm{s} - \Omega_\mathrm{E}.

The factors :math:`b_\mathrm{p}` and :math:`b_\mathrm{E}` modifying the
amplitudes (with :math:`0 < b < 1`) are given by

.. math::

    b^2 &= \cos^2( \Delta\Omega ) + \sin^2( \Delta\Omega ) \cos^2( i ) \\
        &= \frac{ 1 - \sin^2( i ) } { 1 - \sin^2( i ) \cos^2( \xi ) }.

Finally, :math:`v_\mathrm{p,sys,eff}` is the pulsar's systemic effective
velocity, given by

.. math::

    v_\mathrm{p,sys,eff} \simeq d_\mathrm{eff}
                              \left[ \mu_{\alpha\ast} \sin( \Omega_\mathrm{s} )
                                         + \mu_\delta \cos( \Omega_\mathrm{s} )
                              \right].

Set known parameters
====================

The coordinates and parameters of the pulsar system, known from timing studies.

.. jupyter-execute::

    psr_coord = SkyCoord('04h37m15.99744s -47d15m09.7170s')
    mu_alpha_star = 121.4385 * u.mas / u.yr
    mu_delta = -71.4754 * u.mas / u.yr
    
    p_b = 5.7410459 * u.day
    asini_p = 3.3667144 * const.c * u.s
    
    k_p = 2.*np.pi * asini_p / p_b

Set the known properties of Earth's orbit, and derive its orientation with
respect to the line of sight.

.. jupyter-execute::

    p_e = 1. * u.yr
    a_e = 1. * u.au

    v_orb_e = 2.*np.pi * a_e / p_e
    
    psr_coord_eclip = psr_coord.barycentricmeanecliptic
    ascnod_eclip_lon = psr_coord_eclip.lon + 90.*u.deg
    ascnod_eclip = BarycentricMeanEcliptic(lon=ascnod_eclip_lon, lat=0.*u.deg)
    ascnod_equat = SkyCoord(ascnod_eclip).icrs
    
    i_e = psr_coord_eclip.lat + 90.*u.deg
    omega_e = psr_coord.position_angle(ascnod_equat)

For the example in this tutorial, we use the values for the model parameters
found in the :doc:`preceding tutorial <fit_velocities>`. When copying these
numbers in your own case, make sure to use non-negative amplitudes

.. jupyter-execute::

    amp_p =     1.38 * u.km/u.s/u.pc**0.5
    amp_e =     1.91 * u.km/u.s/u.pc**0.5
    xi_p =     67.63 * u.deg
    xi_e =     65.13 * u.deg
    dveff_c =  14.68 * u.km/u.s/u.pc**0.5

Constraints on physical parameters
==================================

These are the physical parameters of interest:
the position angle of the screen :math:`\Omega_\mathrm{s}`,
the pulsar's longitude of ascending node :math:`\Omega_\mathrm{p}`,
the pulsar's orbital inclination :math:`i_\mathrm{p}`,
the distance to the pulsar :math:`d_\mathrm{p}`,
the distance to the screen :math:`d_\mathrm{s}`,
and the velocity of the lens :math:`v_\mathrm{lens}`.
Let's first consider the general case in which all six of these are unknown.
Since the fit only provides five constraints, not all six physical parameters
will have a unique solution. The absolute-value operation in the model equation
causes further non-uniqueness of the solution. Nevertheless, it is possible to
constrain some of the parameters, and derive relation between the remaining
ones.

The position angle of the screen
--------------------------------

The first physical parameter to infer from the free parameters of our model is
the position angle of the screen :math:`\Omega_\mathrm{s}`. This parameter can
be computed from the fitted phase offset of Earth's orbital velocity signature
:math:`\xi_\mathrm{E}` and the known orientation of Earth's orbit
(:math:`i_\mathrm{E}` and :math:`\Omega_\mathrm{E}`), using the equation

.. math::

    \Omega_\mathrm{s} = \Omega_\mathrm{E} + \Delta\Omega_\mathrm{E},
    \qquad \mathrm{with} \qquad
    \tan( \Delta\Omega_\mathrm{E} ) = \frac{ \tan( \xi_\mathrm{E} ) }
                                           { \cos( i_\mathrm{E} ) }.

Note that for a given :math:`\xi_\mathrm{E}`, there are two possible solutions
for :math:`\Delta\Omega_\mathrm{E}` to this equation. These correspond to
rotating the screen by :math:`180^\circ` on the sky, and either one is
acceptable, since the sign (direction) of the lens velocity cannot be retrieved
from the data in hand (the norms of the scaled effective velocities).

.. jupyter-execute::

    delta_omega_e1 = np.arctan(np.tan(xi_e) / np.cos(i_e))
    delta_omega_e2 = delta_omega_e1 + 180.*u.deg
    omega_s1 = delta_omega_e1 + omega_e
    omega_s2 = delta_omega_e2 + omega_e

    print(f'omega_s1: {omega_s1.to(u.deg):8.2f}')
    print(f'omega_s2: {omega_s2.to(u.deg):8.2f}')


The orientation of the pulsar's orbit
-------------------------------------

Knowing :math:`\Omega_\mathrm{s}`, it is possible to retrieve a relation
between :math:`\Omega_\mathrm{p}` and :math:`i_\mathrm{p}` from the equation

.. math::

    \Omega_\mathrm{p} = \Omega_\mathrm{s} - \Delta\Omega_\mathrm{p},
    \qquad \mathrm{with} \qquad
    \tan( \Delta\Omega_\mathrm{p} ) = \frac{ \tan( \xi_\mathrm{p} ) }
                                           { \cos( i_\mathrm{p} ) }.

The :math:`180^\circ` ambiguity in :math:`\Omega_\mathrm{s}` does not matter
for this calculation of :math:`\Delta\Omega_\mathrm{p}`, since both solutions
of :math:`\Omega_\mathrm{s}` give the same value for
:math:`\tan( \Delta\Omega_\mathrm{p} )`. However, when computing
:math:`\Omega_\mathrm{p}`, it is important to consider the ambiguity again:
there are two possible :math:`i_\mathrm{p}`-:math:`\Omega_\mathrm{p}`
relations, offset by :math:`180^\circ` in :math:`\Omega_\mathrm{p}`.

.. jupyter-execute::

    i_p = np.linspace(0., 180., 181) << u.deg

    delta_omega_p = np.arctan(np.tan(xi_p) / np.cos(i_p))

    omega_p1 = omega_s1 - delta_omega_p
    omega_p2 = omega_p1 + 180.*u.deg


For plotting, we use astropy's :py:class:`~astropy.coordinates.Angle` class and
its :py:meth:`~astropy.coordinates.Angle.wrap_at` method to restrict the values
of :math:`\Omega_\mathrm{p}` to the range
:math:`0^\circ \leq \Omega_\mathrm{p} < 360^\circ`.
Also, the two branches are disjointed at :math:`i_\mathrm{p} = 90^\circ`
(where :math:`\cos( i_\mathrm{p} )` changes sign). We stitch the two halves of
the two branches together appropriately to create two continuous curves in
:math:`i_\mathrm{p}`-:math:`\Omega_\mathrm{p}` space.

.. jupyter-execute::

    omega_p1_wrap = Angle(omega_p1).wrap_at(360.*u.deg).deg * u.deg
    omega_p2_wrap = Angle(omega_p2).wrap_at(360.*u.deg).deg * u.deg

    ii_ccw = (i_p <= 90.*u.deg)
    ii_cw =  (i_p >  90.*u.deg)

    omega_p_stitch1 = np.concatenate((omega_p1_wrap[ii_ccw],
                                      omega_p2_wrap[ii_cw]))
    omega_p_stitch2 = np.concatenate((omega_p2_wrap[ii_ccw],
                                      omega_p1_wrap[ii_cw]))

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot(i_p, omega_p_stitch1, c='C0')
    plt.plot(i_p, omega_p_stitch2, c='C0')

    plt.xlim(0., 180.)
    plt.ylim(0., 360.)

    plt.xlabel(r'$i_\mathrm{p}$')
    plt.ylabel(r'$\Omega_\mathrm{p}$')

    plt.show()


Plotting the relations shows how :math:`\Omega_\mathrm{p}` is restricted to two
ranges of values (while :math:`i_\mathrm{p}` is still unrestricted).

.. jupyter-execute::

    print(f'{omega_p_stitch1[-1].to(u.deg):.2f} < omega_p < '
          f'{omega_p_stitch1[0].to(u.deg):.2f}    or    '
          f'{omega_p_stitch2[-1].to(u.deg):.2f} < omega_p < '
          f'{omega_p_stitch2[0].to(u.deg):.2f}')


The effective distance
----------------------

Next, the effective distance :math:`d_\mathrm{eff}` can be calculated using

.. math::

    d_\mathrm{eff} = \frac{ v_\mathrm{orb,E}^2 }{ A_\mathrm{E}^2 }
                     b_\mathrm{E}^2.


.. jupyter-execute::

    b2_e = (1 - np.sin(i_e)**2) / (1 - np.sin(i_e)**2 * np.cos(xi_e)**2)
    d_eff = v_orb_e**2 / amp_e**2 * b2_e

    print(f'd_eff:   {d_eff.to(u.pc):8.2f}')


Given the effective distance, it is possible to derive a relation between
the distance to the pulsar :math:`d_\mathrm{p}` and the distance to the screen
:math:`d_\mathrm{s}`. In terms of the fractional pulsar-screen distance
:math:`s`, the two actual distances are given by

.. math::

    d_\mathrm{p} &= \frac{ s }{ 1 - s } d_\mathrm{eff}, \\
    d_\mathrm{s} &= s d_\mathrm{eff},

.. jupyter-execute::

    ns = 250
    s = np.arange(0.5/ns, 1., 1./ns)

    d_p = s / (1. - s) * d_eff
    d_s = s * d_eff

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot(s, d_p.to(u.pc), label=r'pulsar distance $d_\mathrm{p}$')
    plt.plot(s, d_s.to(u.pc), label=r'screen distance $d_\mathrm{s}$')

    plt.yscale('log')

    plt.xlim(0., 1.)
    plt.ylim(1., 1.e5)

    plt.legend(loc='upper left')

    plt.xlabel(r'fractional pulsar-screen distance $s$')
    plt.ylabel(r'distance from Earth (pc)')

    plt.show()

This also shows that the effective distance sets a maximum on the distance to
the screen :math:`d_\mathrm{s} < d_\mathrm{eff}`.


Pulsar distance--orbital inclination relation
---------------------------------------------

The aplitude of the pulsar's orbital velocity signature :math:`A_\mathrm{p}`
can be used to derive a relation between the distance to the pulsar system
:math:`d_\mathrm{p}` and the sine of its orbital inclination
:math:`\sin( i_\mathrm{p} )`, following

.. math::

    d_\mathrm{p} &= \frac{ \sqrt{ d_\mathrm{eff} } }{ A_\mathrm{p} }
                    \frac{ K_\mathrm{p} }{ \sin( i_\mathrm{p} ) }
                    b_\mathrm{p} \\
                 &= \frac{ v_\mathrm{orb,E} K_\mathrm{p} }
                         { A_\mathrm{E} A_\mathrm{p} }
                    \frac{ b_\mathrm{E} b_\mathrm{p} }{ \sin( i_\mathrm{p} ) }.

.. jupyter-execute::

    nsini_p = 400
    sini_p = np.arange(0.5/nsini_p, 1., 1./nsini_p)

    b2_p = (1 - sini_p**2) / (1 - sini_p**2 * np.cos(xi_p)**2)
    d_p = v_orb_e * k_p / (amp_e * amp_p) * np.sqrt(b2_e * b2_p) / sini_p

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot(sini_p, d_p.to(u.pc))

    plt.yscale('log')

    plt.xlim(0., 1.)
    plt.ylim(10., 1.e5)

    plt.xlabel(r"sine of pulsar's orbital inclination $\sin( i_\mathrm{p} )$")
    plt.ylabel(r"pulsar's distance from Earth $d_\mathrm{p}$ (pc)")

    plt.show()


The lens velocity
-----------------

Finally, it is possible to find a constraint on the lens velocity
:math:`v_\mathrm{lens}`. This is best expressed in terms of some intermediate
quantities derived above (:math:`\Omega_\mathrm{s}` and :math:`d_\mathrm{eff}`)
and as a function the fractional pulsar-screen distance :math:`s`:

.. math::

    v_\mathrm{lens} = s \left( v_\mathrm{p,sys,eff}
                               \pm \sqrt{ d_\mathrm{eff} } C \right),
    \qquad \mathrm{with} \qquad
    v_\mathrm{p,sys,eff} \simeq d_\mathrm{eff}
                              \left[ \mu_{\alpha\ast} \sin( \Omega_\mathrm{s} )
                                         + \mu_\delta \cos( \Omega_\mathrm{s} )
                              \right].

Note that the two possible values of the screen position angle
:math:`\Omega_\mathrm{s}` yield different solutions for the lens velocity
:math:`v_\mathrm{lens}` (because they make different angles with the
proper-motion vector).

To compute a velocity from a proper motion and a distance, we use the
:py:func:`~astropy.units.equivalencies.dimensionless_angles` equivalency, which
takes care of handling :py:mod:`astropy.units` correctly when using the
small-angle approximation (see the `Astropy documentation about equivalencies
<https://docs.astropy.org/en/stable/units/equivalencies.html>`_
for further explanation).

.. jupyter-execute::

    s = [0., 1.]

    v_p_sys_eff1 = ((d_eff * (mu_alpha_star * np.sin(omega_s1)
                                 + mu_delta * np.cos(omega_s1)))
                    .to(u.km/u.s, equivalencies=u.dimensionless_angles()))
    v_p_sys_eff2 = ((d_eff * (mu_alpha_star * np.sin(omega_s2)
                                 + mu_delta * np.cos(omega_s2)))
                    .to(u.km/u.s, equivalencies=u.dimensionless_angles()))

    v_lens1p = s * (np.sqrt(d_eff) *  dveff_c + v_p_sys_eff1)
    v_lens1m = s * (np.sqrt(d_eff) * -dveff_c + v_p_sys_eff1)
    v_lens2p = s * (np.sqrt(d_eff) *  dveff_c + v_p_sys_eff2)
    v_lens2m = s * (np.sqrt(d_eff) * -dveff_c + v_p_sys_eff2)

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot(s, v_lens1p.to(u.km/u.s), c='C0',
        label=rf'$\Omega_\mathrm{{s}} = {omega_s1.to(u.deg).value:.0f}^\circ$')
    plt.plot(s, v_lens1m.to(u.km/u.s), c='C0')
    plt.plot(s, v_lens2p.to(u.km/u.s), c='C1',
        label=rf'$\Omega_\mathrm{{s}} = {omega_s2.to(u.deg).value:.0f}^\circ$')
    plt.plot(s, v_lens2m.to(u.km/u.s), c='C1')

    plt.xlim(0., 1.)

    plt.legend(loc='upper left')

    plt.xlabel(r'fractional pulsar-screen distance $s$')
    plt.ylabel(r'lens velocity $v_\mathrm{lens}$ (km/s)')

    plt.show()

TODO: check this
               
