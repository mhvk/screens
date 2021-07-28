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

    from astropy.coordinates import SkyCoord

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

Set the pulsar system's coordinates
:math:`(\alpha_\mathrm{p}, \delta_\mathrm{p})` and proper motion components
:math:`(\mu_\mathrm{p,sys,\alpha\ast}, \mu_\mathrm{p,sys,\delta})`,
as well as some of the system's parameters that are known from timing studies:
its orbital period :math:`P_\mathrm{orb,p}`, projected semi-major axis
:math:`a_\mathrm{p} \sin( i_\mathrm{p} )`, and radial-velocity amplitude
:math:`K_\mathrm{p} = 2 \pi a_\mathrm{p} \sin( i_\mathrm{p} )
/ P_\mathrm{orb,p}` [which relates to the pulsar's mean orbital speed as
:math:`v_\mathrm{0,p} = K_\mathrm{p} / \sin( i_\mathrm{p} )`].

.. jupyter-execute::

    psr_coord = SkyCoord('04h37m15.99744s -47d15m09.7170s',
                         pm_ra_cosdec=121.4385 * u.mas / u.yr,
                         pm_dec=-71.4754 * u.mas / u.yr)
    
    p_orb_p = 5.7410459 * u.day
    asini_p = 3.3667144 * const.c * u.s
    
    k_p = 2.*np.pi * asini_p / p_orb_p

Set the known properties of Earth's orbit (the orbital period :math:`P_\oplus`,
its semi-major axis :math:`a_\oplus`, and the mean orbital speed
:math:`v_{0,\oplus} = 2 \pi a_\oplus / P_\mathrm{orb,\oplus}`), and derive its
orientation with respect to the line of sight (i.e., the orbit's inclination
:math:`i_\oplus` and longitude of ascending node :math:`\Omega_\oplus`).

.. jupyter-execute::

    p_orb_e = 1. * u.yr
    a_e = 1. * u.au

    v_0_e = 2.*np.pi * a_e / p_orb_e
    
    psr_coord_eclip = psr_coord.barycentricmeanecliptic
    ascnod_eclip = SkyCoord(lon=psr_coord_eclip.lon - 90.*u.deg, lat=0.*u.deg,
                            frame='barycentricmeanecliptic')
    ascnod_equat = ascnod_eclip.icrs
    
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
:math:`\left| v_\mathrm{eff,\parallel} \right| / \sqrt{d_\mathrm{eff}}`
consists of two sinusoids (with known periods) and an offset:

.. math::

    \frac{ \left| v_\mathrm{eff,\parallel} \right| }{ \sqrt{ d_\mathrm{eff} } }
      = \left| A_\oplus     \sin( \phi_\oplus     - \chi_\oplus     )
             + A_\mathrm{p} \sin( \phi_\mathrm{p} - \chi_\mathrm{p} ) + C
        \right|.

Here, :math:`\phi_\oplus` and :math:`\phi_\mathrm{p}` and are the orbital
phases of the Earth and the pulsar, respectively, measured from their ascending
nodes. The free parameters in this equation are the amplitudes of Earth's and
the pulsar's orbital scaled-effective-velocity modulation :math:`A_\oplus` and
:math:`A_\mathrm{p}` (assumed to be non-negative: :math:`A_\oplus \geq 0`,
:math:`A_\mathrm{p} \geq 0`), their phase offsets :math:`\chi_\oplus` and
:math:`\chi_\mathrm{p}`, and a constant scaled-effective-velocity offset
:math:`C`.

We want to figure out how these model parameters are related to the system's
physical parameters of interest, which are:
the pulsar's orbital inclination :math:`i_\mathrm{p}`,
the pulsar's longitude of ascending node :math:`\Omega_\mathrm{p}`,
the distance to the pulsar :math:`d_\mathrm{p}`,
the distance to the screen :math:`d_\mathrm{s}`,
the position angle of the lens :math:`\xi`,
and the velocity of the lens :math:`v_\mathrm{lens,\parallel}`
(in this tutorial, velocities generally refer to the component of the full
three-dimensional velocity that is along the line of images formed by the lens).
In terms of these physical parameters, the model parameters can be expressed as

.. math::

    A_\oplus &= \frac{ v_{0,\oplus} }{ \sqrt{ d_\mathrm{eff} } } b_\oplus
              = \frac{ 1 }{ \sqrt{ d_\mathrm{eff} } }
                \frac{ 2 \pi a_\oplus }{ P_\mathrm{orb,\oplus} } b_\oplus,
    \\[1em]
    A_\mathrm{p} &= \frac{ 1 - s }{ s }
                    \frac{ v_\mathrm{0,p} }{ \sqrt{ d_\mathrm{eff} } }
                    b_\mathrm{p}
                  = \frac{ \sqrt{ d_\mathrm{eff} } }{ d_\mathrm{p} }
                    \frac{ K_\mathrm{p} }{ \sin( i_\mathrm{p} ) }
                    b_\mathrm{p},
    \\[1em]
    \tan( \chi_\oplus )     &= \tan( \Delta\Omega_\oplus ) \cos( i_\oplus ),
    \\[1em]
    \tan( \chi_\mathrm{p} ) &= \tan( \Delta\Omega_\mathrm{p} )
                               \cos( i_\mathrm{p} ),
    \\[1em]
    C &= \frac{ 1 }{ s }
         \frac{ v_\mathrm{lens,\parallel} }{ \sqrt{ d_\mathrm{eff} } }
       - \frac{ 1 - s }{ s }
         \frac{ v_\mathrm{p,sys,\parallel} }{ \sqrt{ d_\mathrm{eff} } }.

These equations contain several auxiliary parameters that need to be defined.
As usual, :math:`d_\mathrm{eff}` refers to the effective distance and :math:`s`
is the fractional screen--pulsar distance (with :math:`0 < s < 1`).
They are related to the distances of the pulsar and the screen according to

.. math::

    d_\mathrm{eff} = \frac{ d_\mathrm{p} d_\mathrm{s} }
                          { d_\mathrm{p} - d_\mathrm{s} },
    \qquad
    s = 1 - \frac{ d_\mathrm{s} }{ d_\mathrm{p} }.

The factors :math:`b_\mathrm{p}` and :math:`b_\oplus`, which modify the
sinusoid amplitudes (with :math:`0 \leq b \leq 1`), are given by (omitting the
subscripts)

.. math::

    b^2 &= \cos^2( \Delta\Omega ) + \sin^2( \Delta\Omega ) \cos^2( i ) \\
        &= \frac{ 1 - \sin^2( i ) } { 1 - \sin^2( i ) \cos^2( \chi ) }.

The symbols :math:`\Delta\Omega_\mathrm{p}` and :math:`\Delta\Omega_\oplus`
denote the angles from the position angle of the screen to the longitude of
ascending node of the orbit of the pulsar and the Earth, respectively, i.e.,

.. math::

    \Delta\Omega_\mathrm{p} = \xi - \Omega_\mathrm{p},
    \qquad
    \Delta\Omega_\oplus     = \xi - \Omega_\oplus.

Finally, :math:`v_\mathrm{p,sys,\parallel}` is the pulsar's systemic velocity
projected onto the line of images formed by the lens. It is given by

.. math::

    v_\mathrm{p,sys,\parallel} = d_\mathrm{p} \mu_\mathrm{p,sys,\parallel},
    \qquad \mathrm{with} \qquad
    \mu_\mathrm{p,sys,\parallel} =
        \left[ \mu_\mathrm{p,sys,\alpha\ast} \sin( \xi )
             + \mu_\mathrm{p,sys,\delta}     \cos( \xi )
        \right],

where :math:`\mu_\mathrm{p,sys}` denotes the pulsar system's proper motion
projected onto the line of images.

For the example in this tutorial, we use the values for the model parameters
found in the :doc:`preceding tutorial <fit_velocities>`.

.. jupyter-execute::

    amp_e =     1.91 * u.km/u.s/u.pc**0.5
    amp_p =     1.34 * u.km/u.s/u.pc**0.5
    chi_e =    65.14 * u.deg
    chi_p =   245.83 * u.deg
    dveff_c =  14.67 * u.km/u.s/u.pc**0.5

Constraints without additional information
==========================================

Let's first consider the general case in which none of the six physical
parameters of interest (:math:`i_\mathrm{p}`, :math:`\Omega_\mathrm{p}`,
:math:`d_\mathrm{p}`, :math:`d_\mathrm{s}`, :math:`\xi`,
:math:`v_\mathrm{lens,\parallel}`) are known. Since the fit only provides five
constraints, not all six physical parameters will have a unique solution.
The absolute-value operation in the model equation causes further
non-uniqueness of the solution. Nevertheless, it is possible to constrain
some of the parameters, and derive relations between the remaining ones.

The position angle of the screen
--------------------------------

The first physical parameter to infer from the free parameters of our model is
the position angle of the screen :math:`\xi`. This parameter can be computed
from the fitted phase offset of Earth's orbital velocity signature
:math:`\chi_\oplus` and the known orientation of Earth's orbit
(:math:`i_\oplus` and :math:`\Omega_\oplus`), using the equation

.. math::

    \xi = \Omega_\oplus + \Delta\Omega_\oplus,
    \qquad \mathrm{with} \qquad
    \tan( \Delta\Omega_\oplus ) = \frac{ \tan( \chi_\oplus ) }
                                       { \cos( i_\oplus ) }.

Note that for a given value of :math:`\chi_\oplus`, there are two possible
solutions to the right-hand-side equation for :math:`\Delta\Omega_\oplus`,
offset by :math:`180^\circ`. These correspond to rotating the screen by
:math:`180^\circ` on the sky and this ambiguity in screen orientation cannot be
resolved using single-telescope data. The angle :math:`\xi`, however, is
restricted to the range :math:`0^\circ \leq \xi < 180^\circ` (because we use
the convention that :math:`\xi` refers to the position angle of the *eastern*
half of the line of lensed images). So, for the purpose of inferring
:math:`\xi`, it is only necessary to consider one of the two
:math:`\Delta\Omega_\oplus` solutions.

.. jupyter-execute::

    delta_omega_e = np.arctan(np.tan(chi_e) / np.cos(i_e))
    xi = (delta_omega_e + omega_e) % (180.*u.deg)

    print(f'xi: {xi.to(u.deg):8.2f}')


The orientation of the pulsar's orbit
-------------------------------------

Knowing :math:`\xi`, it is possible to retrieve a relation between
:math:`i_\mathrm{p}` and :math:`\Omega_\mathrm{p}` from the equation

.. math::

    \Omega_\mathrm{p} = \xi - \Delta\Omega_\mathrm{p},
    \qquad \mathrm{with} \qquad
    \tan( \Delta\Omega_\mathrm{p} ) = \frac{ \tan( \chi_\mathrm{p} ) }
                                           { \cos( i_\mathrm{p} ) }.

Again, for a given value of :math:`\chi_\mathrm{p}`, there are two possible
solutions for :math:`\Delta\Omega_\mathrm{p}`, offset by :math:`180^\circ`.
Hence, there are two possible :math:`i_\mathrm{p}`--:math:`\Omega_\mathrm{p}`
relations, offset by :math:`180^\circ` in :math:`\Omega_\mathrm{p}`. We
restrict the values of :math:`\Omega_\mathrm{p}` to its allowed range of
:math:`0^\circ \leq \Omega_\mathrm{p} < 360^\circ`.

.. jupyter-execute::

    i_p = np.linspace(0.*u.deg, 180.*u.deg, 181)

    delta_omega_p = np.arctan(np.tan(chi_p) / np.cos(i_p)) + [[0.], [180.]] * u.deg
    omega_p = (xi - delta_omega_p) % (360.*u.deg)

The two :math:`i_\mathrm{p}`--:math:`\Omega_\mathrm{p}` relations we found are
disjointed at :math:`i_\mathrm{p} = 90^\circ`, where
:math:`\cos( i_\mathrm{p} )` changes sign. For plotting, we stitch the four
halves of the two solutions together appropriately to create two continuous
curves in :math:`i_\mathrm{p}`--:math:`\Omega_\mathrm{p}` space.

.. jupyter-execute::

    ii_ccw = (i_p <= 90.*u.deg)
    ii_cw =  (i_p >  90.*u.deg)

    omega_p = np.concatenate((omega_p[:,ii_ccw], omega_p[::-1,ii_cw]), axis=1)

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot([0., 180.], [1., 1.] * xi.to(u.deg), '--', c='gray')

    plt.plot(i_p, omega_p.to(u.deg).T, c='C0')

    plt.xlim(0., 180.)
    plt.ylim(0., 360.)

    plt.legend([r"angle of line of lensed images $\xi$",
                r"pulsar's longitude of ascending node $\Omega_\mathrm{p}$"])

    plt.xlabel(r"pulsar's orbital inclination $i_\mathrm{p}$")
    plt.ylabel('angle on the sky (east of north)')

    plt.show()

Plotting the relations shows how :math:`\Omega_\mathrm{p}` is restricted to two
ranges of values (while :math:`i_\mathrm{p}` is still unrestricted).

.. jupyter-execute::

    print(f'{omega_p[0,-1].to(u.deg):.2f} < omega_p < '
          f'{omega_p[0, 0].to(u.deg):.2f}    or    '
          f'{omega_p[1,-1].to(u.deg):.2f} < omega_p < '
          f'{omega_p[1, 0].to(u.deg):.2f}')


The effective distance
----------------------

Next, the effective distance :math:`d_\mathrm{eff}` can be calculated using

.. math::

    d_\mathrm{eff} = \frac{ v_{0,\oplus}^2 }{ A_\oplus^2 } b_\oplus^2.


.. jupyter-execute::

    b2_e = (1. - np.sin(i_e)**2) / (1. - np.sin(i_e)**2 * np.cos(chi_e)**2)
    d_eff = v_0_e**2 / amp_e**2 * b2_e

    print(f'd_eff:   {d_eff.to(u.pc):8.2f}')


Given the effective distance, it is possible to derive a relation between
the distance to the pulsar :math:`d_\mathrm{p}` and the distance to the screen
:math:`d_\mathrm{s}` in terms of the fractional screen--pulsar distance
:math:`s` (with :math:`0 < s < 1`):

.. math::

    d_\mathrm{s} &= s d_\mathrm{eff}, \\
    d_\mathrm{p} &= \frac{ d_\mathrm{s} }{ 1 - s }.

.. jupyter-execute::

    ns = 250
    s = np.linspace(0.5/ns, 1. - 0.5/ns, ns)

    d_s = s * d_eff
    d_p = d_s / (1. - s)

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot([0., 1.], [1., 1.] * d_eff.to(u.pc), '--', c='gray',
             label=r'effective distance $d_\mathrm{eff}$')
    plt.plot(s, d_p.to(u.pc), label=r'pulsar distance $d_\mathrm{p}$')
    plt.plot(s, d_s.to(u.pc), label=r'screen distance $d_\mathrm{s}$')

    plt.yscale('log')

    plt.xlim(0., 1.)
    plt.ylim(10., 1.e4)

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
                    \frac{ K_\mathrm{p} }{ \sin( i_\mathrm{p} ) } b_\mathrm{p}
                 \\
                 &= \frac{ v_{0,\oplus} K_\mathrm{p} }
                         { A_\oplus A_\mathrm{p} }
                    \frac{ b_\oplus b_\mathrm{p} }{ \sin( i_\mathrm{p} ) }.

.. jupyter-execute::

    nsini_p = 250
    sini_p = np.linspace(0.5/nsini_p, 1. - 0.5/nsini_p, nsini_p)

    b2_p = (1. - sini_p**2) / (1. - sini_p**2 * np.cos(chi_p)**2)
    d_p = v_0_e * k_p / (amp_e * amp_p) * np.sqrt(b2_e * b2_p) / sini_p

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot(sini_p, d_p.to(u.pc))

    plt.yscale('log')

    plt.xlim(0., 1.)
    plt.ylim(10., 1.e4)

    plt.xlabel(r"sine of pulsar's orbital inclination $\sin( i_\mathrm{p} )$")
    plt.ylabel(r"pulsar's distance from Earth $d_\mathrm{p}$ (pc)")

    plt.show()


The lens velocity
-----------------

Finally, it is possible to find a constraint on the projected lens velocity
:math:`v_\mathrm{lens,\parallel}`. This is best expressed in terms of some
intermediate quantities derived above (:math:`\xi` and :math:`d_\mathrm{eff}`)
and as a function the fractional screen--pulsar distance :math:`s`:

.. math::

    v_\mathrm{lens,\parallel} = s \left( v_\mathrm{eff,\parallel,p,sys}
                                         + \sqrt{ d_\mathrm{eff} } C \right),

where :math:`v_\mathrm{eff,\parallel,p,sys}` denotes the contribution of the
pulsar's systemic motion to the effective velocity
:math:`v_\mathrm{eff,\parallel}`:

.. math::

    v_\mathrm{eff,\parallel,p,sys}
        = \frac{ 1 - s }{ s } v_\mathrm{p,sys,\parallel}
        = d_\mathrm{eff} \mu_\mathrm{p,sys}
        = d_\mathrm{eff} \left[ \mu_\mathrm{p,sys,\alpha\ast} \sin( \xi )
                              + \mu_\mathrm{p,sys,\delta}     \cos( \xi )
                         \right].

To compute a velocity from a proper motion and a distance, we use the
:py:func:`~astropy.units.equivalencies.dimensionless_angles` equivalency. This
takes care of handling the units of Astropy :py:class:`~astropy.units.Quantity`
objects correctly when using the small-angle approximation
(for further explanation, see the `Astropy documentation about equivalencies
<https://docs.astropy.org/en/stable/units/equivalencies.html>`_).

.. jupyter-execute::

    s = [0., 1.]

    mu_p_sys = psr_coord.pm_ra_cosdec * np.sin(xi) + psr_coord.pm_dec * np.cos(xi)

    v_eff_p_sys = (d_eff * mu_p_sys
                  ).to(u.km/u.s, equivalencies=u.dimensionless_angles())
    
    v_lens = s * (v_eff_p_sys + np.sqrt(d_eff) * dveff_c)

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot(s, v_lens.to(u.km/u.s))

    plt.xlim(0., 1.)

    plt.xlabel(r'fractional screen-pulsar distance $s$')
    plt.ylabel(r'lens velocity $v_\mathrm{lens,\!\!\parallel}$ (km/s)')

    plt.show()


Constraints with a known pulsar distance
========================================

We now consider situations in which there is additional information that
provides constraints on one of the six physical parameters of interest.
Together with the five constraints from scintillometry, this will allow better
constraints on the remaining physical parameters of interest, although some
ambiguity will remain.

In many cases, some external constraints exist on the distance to the pulsar.
An example of such an constraint would be a parallax measurement. While in
reality there will always be some uncertainty associated with the constraint,
here we will assume perfect knowledge to examine how this constrains the
remaining parameters.

Set the known pulsar distance :math:`d_\mathrm{p}`.

.. jupyter-execute::

    d_p = 156.79 * u.pc


The screen distance
-------------------

First of all, together with the scintillometric constraint on the effective
distance :math:`d_\mathrm{eff}`, this immediately sets the distance to the
screen :math:`d_\mathrm{s}` and the fractional screen--pulsar distance
:math:`s`.

.. math::

    d_\mathrm{s} &= \frac{ d_\mathrm{p} d_\mathrm{eff} }
                         { d_\mathrm{p} + d_\mathrm{eff} }, \\
    s &= 1 - \frac{ d_\mathrm{s} }{ d_\mathrm{p} }.

.. jupyter-execute::

    d_s = d_p * d_eff / (d_p + d_eff)
    s = 1. - d_s / d_p

    print(f'd_s:  {d_s.to(u.pc):8.2f}')
    print(f's:    {s:8.2f}')

.. jupyter-execute::

    ns = 250
    s_all = np.linspace(0.5/ns, 1. - 0.5/ns, ns)

    d_s_all = s_all * d_eff
    d_p_all = d_s_all / (1. - s_all)

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot([0., 1.], [1., 1.] * d_eff.to(u.pc), '--', c='gray',
             label=r'effective distance $d_\mathrm{eff}$')
    plt.plot(s_all, d_p_all.to(u.pc), label=r'pulsar distance $d_\mathrm{p}$')
    plt.plot(s_all, d_s_all.to(u.pc), label=r'screen distance $d_\mathrm{s}$')

    plt.plot(s, d_p.to(u.pc), 'k.')
    plt.plot([0., 1., 1.] * s, [1., 1., 1.e-30] * d_p.to(u.pc), ':k')
    plt.plot(s, d_s.to(u.pc), 'k.')
    plt.plot([0., 1.] * s, [1., 1.] * d_s.to(u.pc), ':k')

    plt.yscale('log')

    plt.xlim(0., 1.)
    plt.ylim(10., 1.e4)

    plt.legend(loc='upper left')

    plt.xlabel(r'fractional screen-pulsar distance $s$')
    plt.ylabel(r'distance from Earth (pc)')

    plt.show()


Pulsar orbital inclination
--------------------------

Next, the relation between pulsar distance and orbital inclination can be
solved for :math:`\sin( i_\mathrm{p} )`. This relation first needs to be
rewritten as a (somewhat ugly) quadratic equation in
:math:`\sin^2( i_\mathrm{p} )`:

.. math::

    \cos^2( \chi_\mathrm{p} ) \sin^4( i_\mathrm{p} )
        - ( 1 + Z^2 ) \sin^2( i_\mathrm{p} ) + Z^2 = 0,
    \qquad \mathrm{with} \qquad
    Z = \frac{ v_{0,\oplus} K_\mathrm{p} b_\oplus }
             { A_\oplus A_\mathrm{p} d_\mathrm{p} }.

The standard quadratic formula then gives the solutions

.. math::

    \sin^2( i_\mathrm{p} ) = \frac{ 1 + Z^2 \pm \sqrt{ ( 1 + Z^2 )^2
        - 4 \cos^2( \chi_\mathrm{p} ) Z^2 } }{ 2 \cos^2( \chi_\mathrm{p} ) }.

One of the two solutions should be in the range
:math:`0 \le \sin^2( i_\mathrm{p} ) \le 1`, giving a single real solution for
:math:`\sin( i_\mathrm{p} )` that corresponds to two possible values of
:math:`i_\mathrm{p}`.

.. jupyter-execute::

    z2 = b2_e * (v_0_e * k_p / ( amp_e * amp_p * d_p ) )**2
    cos2chi_p = np.cos(chi_p)**2
    discrim = (1. + z2)**2 - 4. * cos2chi_p * z2
    sin2i_p = ((1. + z2 + [+1., -1.] * np.sqrt(discrim) ) / ( 2. * cos2chi_p ))

    index_real = np.logical_and(sin2i_p >= 0., sin2i_p <= 1.)
    sin2i_p = sin2i_p[index_real][0]
    sini_p = np.sqrt(sin2i_p)

    i_p = [1., -1.] * np.arcsin(sini_p) % (180.*u.deg)

    print(f'sin^2(i_p):   {sin2i_p:8.2f}')
    print(f'sin(i_p):     {sini_p:8.2f}')
    print(f'\ni_p: {i_p[0].to(u.deg):8.2f}   or {i_p[1].to(u.deg):8.2f}')

.. jupyter-execute::

    nsini_p = 250
    sini_p_all = np.linspace(0.5/nsini_p, 1. - 0.5/nsini_p, nsini_p)

    b2_p = (1. - sini_p_all**2) / (1. - sini_p_all**2 * np.cos(chi_p)**2)
    d_p_all = v_0_e * k_p / (amp_e * amp_p) * np.sqrt(b2_e * b2_p) / sini_p_all

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot(sini_p_all, d_p_all.to(u.pc))

    plt.plot(sini_p, d_p.to(u.pc), 'k.')
    plt.plot([0., 1., 1.] * sini_p, [1., 1., 1.e-30] * d_p.to(u.pc), ':k')

    plt.yscale('log')

    plt.xlim(0., 1.)
    plt.ylim(10., 1.e4)

    plt.xlabel(r"sine of pulsar's orbital inclination $\sin( i_\mathrm{p} )$")
    plt.ylabel(r"pulsar's distance from Earth $d_\mathrm{p}$ (pc)")

    plt.show()


Pulsar's longitude of ascending node
------------------------------------

Knowing :math:`\sin( i_\mathrm{p} )`, it is possible to constrain the pulsar's
longitude of ascending node to four possible values.

.. math::

    \Omega_\mathrm{p} = \xi - \Delta\Omega_\mathrm{p},
    \qquad \mathrm{with} \qquad
    \tan( \Delta\Omega_\mathrm{p} ) = \frac{ \tan( \chi_\mathrm{p} ) }
                                           { \cos( i_\mathrm{p} ) },
    \qquad \mathrm{and} \qquad
    \cos( i_\mathrm{p} ) = \pm \sqrt{ 1 - \sin^2( i_\mathrm{p} ) }.

.. jupyter-execute::

    cosi_p = [1., -1.] * np.sqrt(1. - sin2i_p)
    delta_omega_p = np.arctan(np.tan(chi_p) / cosi_p) + [[0.], [180.]] * u.deg
    omega_p = (xi - delta_omega_p) % (360.*u.deg)

    print(f'omega_p:')
    for omg_p in omega_p.flatten():
        print(f'{omg_p.to(u.deg):8.2f}')

.. jupyter-execute::

    i_p_all = np.linspace(0.*u.deg, 180.*u.deg, 181)

    delta_omega_p = (np.arctan(np.tan(chi_p) / np.cos(i_p_all))
                     + [[0.], [180.]] * u.deg)
    omega_p_all = (xi - delta_omega_p) % (360.*u.deg)

    ii_ccw = (i_p_all <= 90.*u.deg)
    ii_cw =  (i_p_all >  90.*u.deg)

    omega_p_all = np.concatenate((omega_p_all[:,ii_ccw],
                                  omega_p_all[::-1,ii_cw]), axis=1)

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot(i_p_all, omega_p_all.to(u.deg).T, c='C0')

    plt.plot([i_p.to(u.deg), i_p.to(u.deg)], omega_p.to(u.deg), 'k.')
    plt.plot([1., 1., 0.] * i_p[1].to(u.deg),
             [0., 1., 1.] * omega_p[0,1].to(u.deg), ':k')
    plt.plot([1., 1., 0.] * i_p[0].to(u.deg),
             [0., 1., 1.] * omega_p[1,0].to(u.deg), ':k')
    plt.plot([0., 1.] * i_p[0].to(u.deg),
             [1., 1.] * omega_p[0,0].to(u.deg), ':k')
    plt.plot([0., 1.] * i_p[1].to(u.deg),
             [1., 1.] * omega_p[1,1].to(u.deg), ':k')

    plt.xlim(0., 180.)
    plt.ylim(0., 360.)

    plt.xlabel(r"pulsar's orbital inclination $i_\mathrm{p}$")
    plt.ylabel(r"pulsar's longitude of ascending node $\Omega_\mathrm{p}$")

    plt.show()


The lens velocity
-----------------

Finally, with :math:`s` known, only two possible values remain for the
lens velocity.

.. math::

    v_\mathrm{lens,\parallel} = s \left( v_\mathrm{eff,\parallel,p,sys}
                                         + \sqrt{ d_\mathrm{eff} } C \right),
    \qquad \mathrm{with} \qquad
    v_\mathrm{eff,\parallel,p,sys}
        = d_\mathrm{eff} \left[ \mu_\mathrm{p,sys,\alpha\ast} \sin( \xi )
                              + \mu_\mathrm{p,sys,\delta}     \cos( \xi )
                         \right].

.. jupyter-execute::

    v_lens = s * (v_eff_p_sys + np.sqrt(d_eff) * dveff_c)

    print(f'v_lens: {v_lens.to(u.km/u.s):8.2f}')

.. jupyter-execute::

    s_all = [0., 1.]

    v_lens_all = s_all * (v_eff_p_sys + np.sqrt(d_eff) * dveff_c)

.. jupyter-execute::

    plt.figure(figsize=(7., 6.))

    plt.plot(s_all, v_lens_all.to(u.km/u.s).T)

    ylims = plt.gca().get_ylim()

    plt.plot(s, v_lens.to(u.km/u.s), 'k.')
    plt.plot([0., 1., 1.] * s, [1., 1., -10.] * v_lens.to(u.km/u.s), ':k')

    plt.xlim(0., 1.)
    plt.ylim(ylims)

    plt.xlabel(r'fractional screen-pulsar distance $s$')
    plt.ylabel(r'lens velocity $v_\mathrm{lens,\!\!\parallel}$ (km/s)')

    plt.show()
