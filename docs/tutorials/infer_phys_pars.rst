*****************************
Inferring physical parameters
*****************************

This tutorial describes how to infer physical parameters from the free
parameters of a phenomenological model for the scintillation velocities of a
pulsar on a circular orbit whose radiation is scattered by a single
one-dimensional screen. The tutorial builds upon a :doc:`preceding tutorial
<fit_velocities>` in which such a phenomenological model is fit to a time
series of scintillation velocities.

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

Set up support for plotting Astropy's
:py:class:`~astropy.units.quantity.Quantity` objects, and make sure that the
output of plotting commands is displayed inline (i.e., directly below the code
cell that produced it).

.. jupyter-execute::

    quantity_support()

    %matplotlib inline

Set known parameters
====================

Set the pulsar system's coordinates :math:`(\alpha, \delta)`
and proper motion components :math:`(\mu_{\alpha\ast}, \mu_\delta)`,
as well as some of the system's parameters that are known from timing studies:
its orbital period :math:`P_\mathrm{b}`, projected semi-major axis
:math:`a_\mathrm{p} \sin( i_\mathrm{p} )`, and radial-velocity amplitude
:math:`K_\mathrm{p} = 2 \pi a_\mathrm{p} \sin( i_\mathrm{p} ) / P_\mathrm{b}`.

.. jupyter-execute::

    psr_coord = SkyCoord('04h37m15.99744s -47d15m09.7170s')
    mu_alpha_star = 121.4385 * u.mas / u.yr
    mu_delta = -71.4754 * u.mas / u.yr
    
    p_b = 5.7410459 * u.day
    asini_p = 3.3667144 * const.c * u.s
    
    k_p = 2.*np.pi * asini_p / p_b

Set the known properties of Earth's orbit (the orbital period
:math:`P_\mathrm{E}`, its semi-major axis :math:`a_\mathrm{E}`, and the mean
orbital speed :math:`v_\mathrm{orb,E} = 2 \pi a_\mathrm{E} / P_\mathrm{E}`),
and derive its orientation with respect to the line of sight
(i.e., the orbit's inclination :math:`i_\mathrm{E}`
and longitude of ascending node :math:`\Omega_\mathrm{E}`).

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

.. warning::

    This calculation assumes that Earth's orbit is circular, which is of course
    not completely accurate. As noted above, the pulsar's orbit is also assumed
    to be circular. These simplifications result in a model in which it is
    clear how the scintillation velocities depend on the physical parameters
    of the system, but this model can clearly be improved by implementing more
    realistic orbits for the pulsar and Earth.

The model parameters
====================

The phenomenological model used to fit the scaled effective velocities
:math:`\left| v_\mathrm{eff} \right| / \sqrt{d_\mathrm{eff}}`
consists of two sinusoids (with known periods) and an offset:

.. math::

    \frac{ \left| v_\mathrm{eff} \right| }{ \sqrt{d_\mathrm{eff}} }
      = \left| A_\mathrm{p} \sin( \phi_\mathrm{p} - \chi_\mathrm{p} )
             + A_\mathrm{E} \sin( \phi_\mathrm{E} - \chi_\mathrm{E} ) + C
        \right|.

Here, :math:`\phi_\mathrm{p}` and :math:`\phi_\mathrm{E}` are the orbital
phases of the pulsar and the Earth , measured from their ascending node.
The free parameters in this equation are the amplitudes of the pulsar's and the
Earth's orbital scaled-effective-velocity modulation :math:`A_\mathrm{p}` and
:math:`A_\mathrm{E}` (assumed to be non-negative: :math:`A_\mathrm{p} \geq 0`,
:math:`A_\mathrm{E} \geq 0`), their phase offsets :math:`\chi_\mathrm{p}` and
:math:`\chi_\mathrm{E}`, and a constant scaled-effective-velocity offset
:math:`C`.

We want to figure out how these model parameters are related to the system's
physical parameters of interest, which are:
the pulsar's longitude of ascending node :math:`\Omega_\mathrm{p}`,
the pulsar's orbital inclination :math:`i_\mathrm{p}`,
the distance to the pulsar :math:`d_\mathrm{p}`,
the distance to the screen :math:`d_\mathrm{s}`,
the position angle of the lens :math:`\xi`,
and the velocity of the lens :math:`v_\mathrm{lens}`
(in this tutorial, velocities generally refer to the component of the full
three-dimensional velocity that is along the line of images formed by the lens).
In terms of these physical parameters, the model parameters can be expressed as

.. math::

    A_\mathrm{p} &= \frac{ \sqrt{ d_\mathrm{eff} } }{ d_\mathrm{p} }
                    \frac{ K_\mathrm{p} }{ \sin( i_\mathrm{p} ) }
                    b_\mathrm{p},

    A_\mathrm{E} &= \frac{ v_\mathrm{orb,E} }{ \sqrt{ d_\mathrm{eff} } }
                    b_\mathrm{E},

    \tan( \chi_\mathrm{p} ) &= \tan( \Delta\Omega_\mathrm{p} )
                               \cos( i_\mathrm{p} ),

    \tan( \chi_\mathrm{E} ) &= \tan( \Delta\Omega_\mathrm{E} )
                               \cos( i_\mathrm{E} ),

    C &= \pm \frac{ v_\mathrm{lens} }{ s \sqrt{ d_\mathrm{eff} } }
         \mp \frac{ v_\mathrm{p,sys,eff} }{ \sqrt{ d_\mathrm{eff} } }.

These equations contain several auxiliary parameters that need to be defined.
As usual, :math:`d_\mathrm{eff}` refers to the effective distance and :math:`s`
is the fractional screen--pulsar distance (with :math:`0 < s < 1`).
They are related to the distances of the pulsar and the screen according to

.. math::

    d_\mathrm{eff} = \frac{ d_\mathrm{p} d_\mathrm{s} }
                          { d_\mathrm{p} - d_\mathrm{s} },
    \qquad
    s = 1 - \frac{ d_\mathrm{s} }{ d_\mathrm{p} }.

The factors :math:`b_\mathrm{p}` and :math:`b_\mathrm{E}` modifying the
sinusoid amplitudes (with :math:`0 \leq b \leq 1`) are given by (omitting the
subscripts)

.. math::

    b^2 &= \cos^2( \Delta\Omega ) + \sin^2( \Delta\Omega ) \cos^2( i ) \\
        &= \frac{ 1 - \sin^2( i ) } { 1 - \sin^2( i ) \cos^2( \xi ) }.

The symbols :math:`\Delta\Omega_\mathrm{p}` and :math:`\Delta\Omega_\mathrm{E}`
denote the angles from the position angle of the screen to the longitude of
ascending node of the orbit of the pulsar and the Earth, respectively, i.e.,

.. math::

    \Delta\Omega_\mathrm{p} = \xi - \Omega_\mathrm{p},
    \qquad
    \Delta\Omega_\mathrm{E} = \xi - \Omega_\mathrm{E}.

Finally, :math:`v_\mathrm{p,sys,eff}` is the pulsar's systemic effective
velocity, given by

.. math::

    v_\mathrm{p,sys,eff} \simeq d_\mathrm{eff}
                                \left[ \mu_{\alpha\ast} \sin( \xi )
                                           + \mu_\delta \cos( \xi )
                                \right].

For the example in this tutorial, we use the values for the model parameters
found in the :doc:`preceding tutorial <fit_velocities>`.

.. jupyter-execute::

    amp_p =     1.38 * u.km/u.s/u.pc**0.5
    amp_e =     1.91 * u.km/u.s/u.pc**0.5
    chi_p =    67.63 * u.deg
    chi_e =    65.13 * u.deg
    dveff_c =  14.68 * u.km/u.s/u.pc**0.5

Constraints on physical parameters
==================================

Let's first consider the general case in which none of the six physical
parameters of interest are known. Since the fit only provides five
constraints, not all six physical parameters will have a unique solution.
The absolute-value operation in the model equation causes further
non-uniqueness of the solution. Nevertheless, it is possible to constrain
some of the parameters, and derive relations between the remaining ones.

The position angle of the screen
--------------------------------

The first physical parameter to infer from the free parameters of our model is
the position angle of the screen :math:`\xi`. This parameter can be computed
from the fitted phase offset of Earth's orbital velocity signature
:math:`\chi_\mathrm{E}` and the known orientation of Earth's orbit
(:math:`i_\mathrm{E}` and :math:`\Omega_\mathrm{E}`), using the equation

.. math::

    \xi = \Omega_\mathrm{E} + \Delta\Omega_\mathrm{E},
    \qquad \mathrm{with} \qquad
    \tan( \Delta\Omega_\mathrm{E} ) = \frac{ \tan( \chi_\mathrm{E} ) }
                                           { \cos( i_\mathrm{E} ) }.

Note that for a given value of :math:`\chi_\mathrm{E}`, there are two possible
solutions to the right-hand-side equation for :math:`\Delta\Omega_\mathrm{E}`,
offset by :math:`180^\circ`. These correspond to rotating the screen by
:math:`180^\circ` on the sky and this ambiguity in screen orientation cannot be
resolved using single-telescope data. The angle :math:`\xi`, however, is
restricted to the range :math:`0^\circ \leq \xi < 180^\circ` (because we use
the convention that :math:`\xi` refers to the position angle of the *eastern*
half of the line of lensed images). So, for the purpose of inferring
:math:`\xi`, it is only necessary to consider one of the two
:math:`\Delta\Omega_\mathrm{E}` solutions. We use Astropy's
:py:class:`~astropy.coordinates.Angle` class and its
:py:meth:`~astropy.coordinates.Angle.wrap_at` method to restrict the value of
:math:`\xi` to its allowed range.

.. jupyter-execute::

    delta_omega_e = np.arctan(np.tan(chi_e) / np.cos(i_e))
    xi = delta_omega_e + omega_e
    xi = Angle(xi).wrap_at(180.*u.deg).deg * u.deg

    print(f'xi: {xi.to(u.deg):8.2f}')


The orientation of the pulsar's orbit
-------------------------------------

Knowing :math:`\xi`, it is possible to retrieve a relation between
:math:`\Omega_\mathrm{p}` and :math:`i_\mathrm{p}` from the equation

.. math::

    \Omega_\mathrm{p} = \xi - \Delta\Omega_\mathrm{p},
    \qquad \mathrm{with} \qquad
    \tan( \Delta\Omega_\mathrm{p} ) = \frac{ \tan( \chi_\mathrm{p} ) }
                                           { \cos( i_\mathrm{p} ) }.

Again, for a given value of :math:`\chi_\mathrm{p}`, there are two possible
solutions for :math:`\Delta\Omega_\mathrm{p}`, offset by :math:`180^\circ`.
Hence, there are two possible :math:`i_\mathrm{p}`--:math:`\Omega_\mathrm{p}`
relations, offset by :math:`180^\circ` in :math:`\Omega_\mathrm{p}`. We use
Astropy's :py:class:`~astropy.coordinates.Angle` and
:py:meth:`~astropy.coordinates.Angle.wrap_at` to restrict the values of
:math:`\Omega_\mathrm{p}` to its allowed range of
:math:`0^\circ \leq \Omega_\mathrm{p} < 360^\circ`.

.. jupyter-execute::

    i_p = np.linspace(0., 180., 181) << u.deg

    delta_omega_p1 = np.arctan(np.tan(chi_p) / np.cos(i_p))
    delta_omega_p2 = delta_omega_p1 + 180.*u.deg

    omega_p1 = xi - delta_omega_p1
    omega_p2 = xi - delta_omega_p2

    omega_p1 = Angle(omega_p1).wrap_at(360.*u.deg).deg * u.deg
    omega_p2 = Angle(omega_p2).wrap_at(360.*u.deg).deg * u.deg

The two :math:`i_\mathrm{p}`--:math:`\Omega_\mathrm{p}` relations are
disjointed at :math:`i_\mathrm{p} = 90^\circ`, where
:math:`\cos( i_\mathrm{p} )` changes sign. For plotting, we stitch the four
halves of the two solutions together appropriately to create two continuous
curves in :math:`i_\mathrm{p}`--:math:`\Omega_\mathrm{p}` space.

.. jupyter-execute::

    ii_ccw = (i_p <= 90.*u.deg)
    ii_cw =  (i_p >  90.*u.deg)

    omega_p_stitch1 = np.concatenate((omega_p1[ii_ccw], omega_p2[ii_cw]))
    omega_p_stitch2 = np.concatenate((omega_p2[ii_ccw], omega_p1[ii_cw]))

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot(i_p, omega_p_stitch1, c='C0')
    plt.plot(i_p, omega_p_stitch2, c='C0')

    plt.xlim(0., 180.)
    plt.ylim(0., 360.)

    plt.xlabel(r"pulsar's orbital inclination $i_\mathrm{p}$")
    plt.ylabel(r"pulsar's longitude of ascending node $\Omega_\mathrm{p}$")

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

    b2_e = (1 - np.sin(i_e)**2) / (1 - np.sin(i_e)**2 * np.cos(chi_e)**2)
    d_eff = v_orb_e**2 / amp_e**2 * b2_e

    print(f'd_eff:   {d_eff.to(u.pc):8.2f}')


Given the effective distance, it is possible to derive a relation between
the distance to the pulsar :math:`d_\mathrm{p}` and the distance to the screen
:math:`d_\mathrm{s}`. In terms of the fractional screen--pulsar distance
:math:`s`, the two true distances are given by

.. math::

    d_\mathrm{p} &= \frac{ s }{ 1 - s } d_\mathrm{eff}, \\
    d_\mathrm{s} &= s d_\mathrm{eff}.

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

    plt.xlabel(r'fractional screen-pulsar distance $s$')
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

    b2_p = (1 - sini_p**2) / (1 - sini_p**2 * np.cos(chi_p)**2)
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

Finally, it is possible to find a constraint on the projected lens velocity
:math:`v_\mathrm{lens}`. This is best expressed in terms of some intermediate
quantities derived above (:math:`\xi` and :math:`d_\mathrm{eff}`) and as a
function the fractional screen--pulsar distance :math:`s`:

.. math::

    v_\mathrm{lens} = s \left( v_\mathrm{p,sys,eff}
                               \pm \sqrt{ d_\mathrm{eff} } C \right),
    \qquad \mathrm{with} \qquad
    v_\mathrm{p,sys,eff} \simeq d_\mathrm{eff}
                                \left[ \mu_{\alpha\ast} \sin( \xi )
                                           + \mu_\delta \cos( \xi )
                                \right].

To compute a velocity from a proper motion and a distance, we use the
:py:func:`~astropy.units.equivalencies.dimensionless_angles` equivalency. This
takes care of handling the units of Astropy :py:class:`~astropy.units.Quantity`
objects correctly when using the small-angle approximation
(for further explanation, see the `Astropy documentation about equivalencies
<https://docs.astropy.org/en/stable/units/equivalencies.html>`_).

.. jupyter-execute::

    s = [0., 1.]

    v_p_sys_eff = ((d_eff * (mu_alpha_star * np.sin(xi)
                                + mu_delta * np.cos(xi)))
                   .to(u.km/u.s, equivalencies=u.dimensionless_angles()))

    v_lens1 = s * (np.sqrt(d_eff) *  dveff_c + v_p_sys_eff)
    v_lens2 = s * (np.sqrt(d_eff) * -dveff_c + v_p_sys_eff)

Because only the *norm* of the scintillation velocity can be measured, there
are two possible solutions for :math:`v_\mathrm{lens}`: one in which the lens
motion and the pulsar's systemic motion add up to a large offset in
scintillation velocity, and one in which they counteract one another's
contribution to the scintillation-velocity offset. For known values of the
scaled-scintillation-velocity offset :math:`C` and the pulsar's systemic
effective velocity :math:`v_\mathrm{p,sys,eff}`, this translates to solutions
for :math:`v_\mathrm{lens}` with low and high absolute values, respectively.

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot(s, v_lens1.to(u.km/u.s),
             label=r'$\mathrm{{sgn}}(v_\mathrm{{lens}}) \neq '
                   r'\mathrm{{sgn}}(v_\mathrm{{p,sys}})$')
    plt.plot(s, v_lens2.to(u.km/u.s),
             label=r'$\mathrm{{sgn}}(v_\mathrm{{lens}}) = '
                   r'\mathrm{{sgn}}(v_\mathrm{{p,sys}})$')

    plt.xlim(0., 1.)

    plt.legend(loc='upper left')

    plt.xlabel(r'fractional screen-pulsar distance $s$')
    plt.ylabel(r'lens velocity $v_\mathrm{lens}$ (km/s)')

    plt.show()

