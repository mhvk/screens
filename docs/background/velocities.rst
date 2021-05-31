************************
Scintillation velocities
************************


Consider a pulsar in a binary, observed through a single thin scattering screen
that acts as a one-dimensional lens. The goal of these derivations is to
examine how scintillometric measurements from Earth can constrain the physical
parameters of the pulsar system and the screen.


Scintillometric observables
===========================

The basic scintillometric observable is the curvature
:math:`\eta( t, \lambda )` of the parabola in a secondary spectrum, taken at a
reference time :math:`t` and wavelength :math:`\lambda`. In terms of physical
properties of the system, it can be expressed as

.. math::

    \eta = \frac{ \lambda^2 d_\mathrm{eff} }{ 2 c v_\mathrm{eff,\parallel}^2 },

where :math:`c` is the speed of light, and :math:`d_\mathrm{eff}` and
:math:`v_\mathrm{eff,\parallel}` are the effective distance and the effective
velocity, respectively, to be defined below. Isolating the unknowns in this
equation and taking the square root shows that curvature measurements constrain
the quantity

.. math::

    \frac{ \lambda }{ \sqrt{ 2 c \eta } }
        = \frac{ \left| v_\mathrm{eff,\parallel} \right| }
               { \sqrt{ d_\mathrm{eff} } },

which we will refer to as the scaled effective velocity. This also shows that
simultaneous curvature measurements taken at different wavelengths can be
combined, leaving a time-series of scaled effective velocities as the relevant
scintillometric observable to be modelled or fitted.


The effective distance
======================

The distances from Earth to the pulsar and the scattering screen are denoted by
:math:`d_\mathrm{p}` and :math:`d_\mathrm{s}`, respectively. These two physical
distances are combined into the effective distance

.. math::

    d_\mathrm{eff} = \frac{ d_\mathrm{p} d_\mathrm{s} }
                          { d_\mathrm{p} - d_\mathrm{s} }
                   = \frac{ 1 - s }{ s } d_\mathrm{p}.

Here, :math:`s` is the fractional pulsar--screen distance
(with :math:`0 \leq s \leq 1`), given by

.. math::

    s = 1 - \frac{ d_\mathrm{s} }{ d_\mathrm{p} }.

[figure: sketch with distances]


The effective velocity
======================

The scattering of the pulsar's radiation by the screen gives rise to an
interference pattern of bright and dark fringes in space. Because the pulsar,
the screen, and Earth all move with respect to one another, an Earth-based
observation will probe different points in this pattern over time.

.. TODO: [add figure showing interference pattern in observing plane]

The sky-plane velocity (i.e., the velocity component that is perpendicular
to the direct line of sight) of the interference pattern relative to Earth,
known as the effective velocity, is given by

.. math::

    \vec{v}_\mathrm{eff} = \frac{ 1 }{ s } \vec{v}_\mathrm{lens,sky}
        - \frac{ 1 - s }{ s } \vec{v}_\mathrm{p,sky}
        - \vec{v}_\mathrm{\oplus,sky}.

Here, :math:`\vec{v}_\mathrm{p,sky}`, :math:`\vec{v}_\mathrm{lens,sky}`, and
:math:`\vec{v}_\mathrm{\oplus,sky}` are the sky-plane velocities of the pulsar,
the screen, and Earth, respectively, all specified relative to the Solar
System's barycentre.
The pulsar's sky-plane velocity can be split into a systemic component
:math:`\vec{v}_\mathrm{p,sys,sky}` (corresponding to the system's proper
motion), specified relative to the Solar System's barycentre, and an orbital
component :math:`\vec{v}_\mathrm{p,orb,sky}`, specified relative to the
pulsar system's barycentre: :math:`\vec{v}_\mathrm{p,sky} =
\vec{v}_\mathrm{p,sys,sky} + \vec{v}_\mathrm{p,orb,sky}`.

.. TODO: [add figure explaining the 1/s, -(1-s)/s, and -1 factors]

Since we consider a one-dimensional lens, only velocity components parallel to
the line of lensed images (marked by subscript ':math:`\parallel`' below) have
any effect on the shift of the interference pattern. Thus, the equation for the
effective velocity, considering only the relevant velocity components, becomes

.. math::

    v_\mathrm{eff,\parallel} = \frac{ 1 }{ s } v_\mathrm{lens,\parallel}
        - \frac{ 1 - s }{ s } v_\mathrm{p,\parallel} - v_{\oplus,\parallel}.
    \label{eq_v_eff} \tag{1}

Again, the pulsar's term can be split into a systemic and an orbital component:
:math:`v_\mathrm{p,\parallel} = v_\mathrm{p,sys,\parallel} +
v_\mathrm{p,orb,\parallel}`.

In the following, we examine how the different velocities in equation
:math:`\ref{eq_v_eff}` that contribute to :math:`v_\mathrm{eff,\parallel}`
depend on the physical parameters of the pulsar system and the scattering
screen. The first of these velocities, :math:`v_\mathrm{lens,\parallel}`,
is a free parameter in the problem, leaving us to examine
:math:`v_\mathrm{p,sys,\parallel}`, :math:`v_\mathrm{p,orb,\parallel}` and
:math:`v_{\oplus,\parallel}`.


The pulsar's systemic motion
============================

The pulsar's systemic velocity in the plane of the sky
:math:`\vec{v}_\mathrm{p,sys,sky}` can be found from the system's proper motion
:math:`\vec{\mu}` following

.. math::

    \vec{v}_\mathrm{p,sys,sky} = d_\mathrm{p} \vec{\mu}.

The component of this velocity that is parallel to the line of images formed by
the lens is then given by
    
.. math::

    v_\mathrm{p,sys,\parallel} = d_\mathrm{p}
                                    \left[ \mu_{\alpha\ast} \sin( \xi )
                                         + \mu_\delta \cos( \xi )
                                    \right].

Here, :math:`\mu_{\alpha\ast}` is proper motion's right-ascension component
(including the :math:`\cos( \delta_\mathrm{p} )` term, with
:math:`\delta_\mathrm{p}` being the source's declination), :math:`\mu_\delta`
is its declination component, and :math:`\xi` is the position angle of the line
of lensed images, measured from the celestial north through east, with
:math:`0^\circ \leq \xi < 180^\circ`. Because the angle :math:`\xi` is
restricted to this range, it technically refers to the position angle of the
*eastern* half of the line of lensed images.

[figure: sky plane with screen and proper motion vector]


The pulsar's orbital motion -- circular orbit
=============================================

Let's first consider the case of a pulsar in a circular orbit.


The :math:`xyz` coordinate system and the orbital inclination
-------------------------------------------------------------

We introduce the right-handed orthonormal triad :math:`(\hat{x}, \hat{y},
\hat{z})` linked to the binary's barycentre, with :math:`\hat{x}` and
:math:`\hat{y}` in the plane of the sky, :math:`\hat{x}` pointing to the
pulsar's ascending node (the point on the orbit that intersects the sky-plane,
where the pulsar moves away from the observer), and :math:`\hat{z}` pointing
away from the observer. Since this is a right-handed coordinate system,
if :math:`\hat{y}` points upward as viewed by the observer, then
:math:`\hat{x}` points to the left (and the :math:`x` coordinate increases
towards the left).

The orbital inclination is parameterised by the angle :math:`i_\mathrm{p}`
between the direction towards the observer :math:`-\hat{z}` and the binary's
orbital specific angular momentum
:math:`\vec{h}_\mathrm{p} = \vec{r}_\mathrm{p} \times \vec{v}_\mathrm{p}`,
where :math:`\vec{r}_\mathrm{p}` and :math:`\vec{v}_\mathrm{p}` denote the
pulsar's position and velocity, respectively. This angle is naturally
restricted to the range :math:`0^\circ \le i_\mathrm{p} < 180^\circ`. As per
the standard convention for orbits outside the Solar System, inclinations of
:math:`i_\mathrm{p} < 90^\circ` correspond to counterclockwise rotation on the
sky and inclinations of :math:`i_\mathrm{p} > 90^\circ` correspond to clockwise
rotation on the sky, with :math:`i_\mathrm{p} = 90^\circ` for an edge-on orbit
(:math:`\vec{h}_\mathrm{p}` anti-parallel to :math:`\hat{y}`).

[figure: xy plane with orbit]

[figure: yz plane with orbit and inclination]

**Left:** observer's view, looking in the direction of :math:`\hat{z}`.
**Right:** side view, looking in the direction of :math:`\hat{x}`.


The pulsar's position and velocity in :math:`xyz` coordinates
-------------------------------------------------------------

In this :math:`xyz` coordinate system,
the pulsar's position and velocity as function of the its orbital phase
:math:`\phi_\mathrm{p} = 2 \pi ( t - t_\mathrm{asc,p} ) / P_\mathrm{orb,p}`,
measured from the ascending node of the pulsar's orbit, are given by

.. math::

    \vec{r}_\mathrm{p} &= a_\mathrm{p}
        \left[   \cos( \phi_\mathrm{p} ),
               - \cos( i_\mathrm{p} ) \sin( \phi_\mathrm{p} ),
                 \sin( i_\mathrm{p} ) \sin( \phi_\mathrm{p} )
        \right], \\
    \vec{v}_\mathrm{p} &= v_\mathrm{0,p}
        \left[ - \sin( \phi_\mathrm{p} ),
               - \cos( i_\mathrm{p} ) \cos( \phi_\mathrm{p} ),
                 \sin( i_\mathrm{p} ) \cos( \phi_\mathrm{p} )
        \right].

Here, :math:`t_\mathrm{asc,p}` is the pulsar's time of ascending node passage,
:math:`P_\mathrm{orb,p}` is the binary's orbital period,
:math:`a_\mathrm{p}` is the semi-major axis of the pulsar's orbit, and
:math:`v_\mathrm{0,p} = 2 \pi a_\mathrm{p} / P_\mathrm{orb,p}` is the
mean orbital speed of the pulsar. Pulsar timing studies normally constrain
:math:`t_\mathrm{asc,p}` and :math:`P_\mathrm{orb,p}`, as well as the pulsar
orbit's projected semi-major axis :math:`a_\mathrm{p} \sin( i_\mathrm{p} )`
and the hence pulsar's radial-velocity amplitude :math:`K_\mathrm{p}
= v_\mathrm{0,p} \sin( i_\mathrm{p} ) = 2 \pi a_\mathrm{p} \sin( i_\mathrm{p} )
/ P_\mathrm{orb,p}`.


The orbit's orientation on the sky and the sky-plane velocity
-------------------------------------------------------------

The orientation of the pulsar's orbit on the sky is parameterised by its
longitude of ascending node :math:`\Omega_\mathrm{p}`, measured from the
celestial north through east.

[figure: sky plane with orbit]

In the equatorial coordinate system, the pulsar's orbital sky-plane velocity is
:math:`\vec{v}_\mathrm{p,orb,sky} = (v_\mathrm{p,\alpha\ast},
v_\mathrm{p,\delta}, 0)` with

.. math::

    v_\mathrm{p,\alpha\ast} &= \vec{v}_\mathrm{p} \cdot \hat{\alpha}
        = - v_\mathrm{0,p}
            \left[ \sin( \Omega_\mathrm{p} ) \sin( \phi_\mathrm{p} )
                - \cos( \Omega_\mathrm{p} ) \cos( i_\mathrm{p} )
                    \cos( \phi_\mathrm{p} )
            \right], \\
    v_\mathrm{p,\delta} &= \vec{v}_\mathrm{p} \cdot \hat{\delta}
        = - v_\mathrm{0,p}
            \left[ \cos( \Omega_\mathrm{p} ) \sin( \phi_\mathrm{p} )
                + \sin( \Omega_\mathrm{p} ) \cos( i_\mathrm{p} )
                    \cos( \phi_\mathrm{p} )
            \right].


Projecting the sky-plane velocity onto the line of lensed images
----------------------------------------------------------------

The component of the pulsar's orbital sky-plane velocity
:math:`\vec{v}_\mathrm{p,orb,sky}` that is parallel to the line of images
formed by the lens is then given by
    
.. math::

    \begin{align}
    v_\mathrm{p,orb,\parallel}
        &= \left[ v_\mathrm{p,\alpha\ast} \sin( \xi )
                + v_\mathrm{p,\delta} \cos( \xi )
           \right] \\
        &= - v_\mathrm{0,p}
            \left[ \cos( \Delta\Omega_\mathrm{p} ) \sin( \phi_\mathrm{p} )
                 - \sin( \Delta\Omega_\mathrm{p} ) \cos( i_\mathrm{p} )
                     \cos( \phi_\mathrm{p} )
            \right].
        \label{eq_v_p_orb_parallel} \tag{2}
    \end{align}

where :math:`\Delta\Omega_\mathrm{p} = \xi - \Omega_\mathrm{p}` is the angle
of the screen measured from the ascending node of the pulsar orbit.

[figure: sky plane with delta_omega_p]


The velocity modulation's amplitude and phase offset
----------------------------------------------------

Equation :math:`\ref{eq_v_p_orb_parallel}` for the pulsar's orbital sky-plane
velocity's screen component :math:`v_\mathrm{p,orb,\parallel}` describes a
sinusoid as a function of orbital phase :math:`\phi_\mathrm{p}`. Via some
trigonometry, this equation can be rewritten as

.. math::

    v_\mathrm{p,orb,\parallel}
      = - v_\mathrm{0,p} b_\mathrm{p} \sin( \phi_\mathrm{p} - \chi_\mathrm{p} )
      = - \frac{ K_\mathrm{p} }{ \sin( i_\mathrm{p} ) } b_\mathrm{p}
            \sin( \phi_\mathrm{p} - \chi_\mathrm{p} ).

Here, the parameter :math:`b_\mathrm{p}` modifying the sinusoid's amplitude
(with :math:`0 \leq b_\mathrm{p} \leq 1`) is given by

.. math::

    b_\mathrm{p}^2 = \cos^2( \Delta\Omega_\mathrm{p} )
                   + \sin^2( \Delta\Omega_\mathrm{p} ) \cos^2( i_\mathrm{p} )
                   = \frac{ 1 - \sin^2( i_\mathrm{p} ) }
                          { 1 - \sin^2( i_\mathrm{p} ) \cos^2( \xi ) }.


The sinusoid's phase offset :math:`\chi_\mathrm{p}` is given by

.. math::

    \tan( \chi_\mathrm{p} ) = \frac{ \sin( \Delta\Omega_\mathrm{p} ) }
                                   { \cos( \Delta\Omega_\mathrm{p} ) }
                              \cos( i_\mathrm{p} )
                            = \tan( \Delta\Omega_\mathrm{p} )
                              \cos( i_\mathrm{p} ).


Earth's motion around the Sun
=============================


Earth's velocity projected onto the line of lensed images
---------------------------------------------------------

Earth's sky-plane velocity :math:`\vec{v}_\mathrm{\oplus,sky}` is its velocity,
relative to the Solar System's barycentre, in the plane perpendicular to the
line of sight towards the source. It can be found in the same way as the
pulsar's orbital sky-plane velocity :math:`\vec{v}_\mathrm{p,orb,sky}`, using
an :math:`xyz` coordinate system with the same orientation, but linked to the
Solar System's barycentre, and substituting the subscript ':math:`\mathrm{p}`'
with the subscript ':math:`\oplus`' in the above derivations. Thus, under the
simplifying assumption that Earth's orbit around the Solar System's barycentre
is circular, the component of Earth's sky-plane velocity along the line of
lensed images is given by

.. math::

    v_{\oplus,\parallel} = - v_{0,\oplus} b_\oplus
        \sin( \phi_\oplus - \chi_\oplus ),

with

.. math::

    v_{0,\oplus} = \frac{ 2 \pi a_\oplus }{ P_\mathrm{orb,\oplus} },
    \qquad
    b_\oplus^2 = \frac{ 1 - \sin^2( i_\oplus ) }
                      { 1 - \sin^2( i_\oplus ) \cos^2( \xi ) },
    
.. math::

    \phi_\oplus = 2 \pi \frac{ t - t_\mathrm{asc,\oplus} }
                             { P_\mathrm{orb,\oplus} },
    \qquad
    \tan( \chi_\oplus ) = \tan( \Delta\Omega_\oplus ) \cos( i_\oplus ),
    \qquad
    \Delta\Omega_\oplus = \xi - \Omega_\oplus.


Earth's orbital orientation
---------------------------

In contrast to the pulsar, all of Earth's orbital parameters
(:math:`P_\mathrm{orb,\oplus}`, :math:`a_\oplus`, :math:`i_\oplus`,
:math:`\Omega_\oplus`, :math:`t_\mathrm{asc,\oplus}`) are known. The
orientation of Earth's orbit with respect to the line of sight,
parameterised by :math:`i_\oplus` and :math:`\Omega_\oplus`,
can be derived from the pulsar system's ecliptic coordinates
:math:`(\lambda_\mathrm{p}, \beta_\mathrm{p})`.

[figure: ecliptic plane with orbit]

[figure: inside ecliptic plane with orbit and inclination]

**Left:** top-down view, looking in the direction of
:math:`-\vec{h}_\oplus`.
**Right:** side view, looking in the direction of :math:`\hat{x}`.

The inclination of Earth's orbital plane with respect to the line of sight
:math:`i_\oplus` is defined in the same way as the pulsar's orbital
inclination: it is the angle between the :math:`-\hat{z}` axis (pointing from
the Solar System's barycentre to the direction opposite of the pulsar) and
the Earth's orbital specific angular momentum vector :math:`\vec{h}_\oplus`.
It is given by

.. math::
    
    i_\oplus = \beta_\mathrm{p} + 90^\circ.

The restriction on the pulsar's ecliptic longitude :math:`-90^\circ \le
\beta_\mathrm{p} \le 90^\circ` leads to the expected range of allowed
inclinations :math:`0^\circ \le i_\oplus \le 180^\circ`. The convention for the
sense of rotation is also the same: :math:`i_\oplus < 90^\circ` for
counterclockwise rotation when viewing in the :math:`\hat{z}` direction and
:math:`i_\oplus > 90^\circ` for clockwise rotation.

Earth's ascending node with respect to the line of sight is the point on the
orbit where Earth passes through the observing plane in the direction of the
pulsar. In this context, the longitude of ascending node :math:`\Omega_\oplus`
is equivalent to the position angle of Earth's ascending node with respect to
the coordinates of the pulsar system:

.. math::

    \Omega_\oplus = \mathcal{P}( X_\mathrm{p}, X_\mathrm{asc,\oplus} ).

Here, :math:`\mathcal{P}( X_1, X_2 )` yields the position angle (east of north)
from position :math:`X_1` to position :math:`X_2` (for details on this
computation, see, e.g., the `Wikipedia article on position angle
<https://en.wikipedia.org/wiki/Position_angle>`_), :math:`X_\mathrm{p} =
(\alpha_\mathrm{p}, \delta_\mathrm{p})` denotes the equatorial coordinates of
the pulsar system, and :math:`X_\mathrm{asc,\oplus} =
(\alpha_\mathrm{asc,\oplus}, \delta_\mathrm{asc,\oplus})` is the equatorial
coordinates of Earth's ascending node. The latter can be found from its
ecliptic coordinates :math:`(\lambda_\mathrm{asc,\oplus},
\beta_\mathrm{asc,\oplus}) = (\lambda_\mathrm{p} - 90^\circ, 0)`.

.. TODO: [maybe include figure here showing Omega_earth]

Finally, assuming Earth's orbit is circular, the time of Earth's passage
through the ascending node is given by

.. math::
    
    t_\mathrm{asc,\oplus} = t_\mathrm{eqx} + P_\mathrm{orb,\oplus}
        \frac{ \lambda_\mathrm{asc,\oplus} }{ 360^\circ },

where :math:`t_\mathrm{eqx}` is the time of the March equinox and
:math:`\lambda_\mathrm{asc,\oplus} = \lambda_\mathrm{p} - 90^\circ` is the
ecliptic longitude of Earth's ascending node.


A model for scaled effective velocity
=====================================

Combining the different terms in equation :math:`\ref{eq_v_eff}` contributing
to :math:`v_\mathrm{eff,\parallel}` gives

.. math::

    v_\mathrm{eff,\parallel} =
        \underbrace{
            \frac{ 1 }{ s } v_\mathrm{lens,\parallel}
        }_\textrm{lens motion}
      - \underbrace{
            \frac{ 1 - s }{ s } d_\mathrm{p}
                \left[ \mu_{\alpha\ast} \sin( \xi )
                     + \mu_\delta \cos( \xi )
                \right]
        }_\textrm{pulsar's systemic motion}
      + \underbrace{
            \frac{ 1 - s }{ s } v_\mathrm{0,p} b_\mathrm{p}
                \sin( \phi_\mathrm{p} - \chi_\mathrm{p} )
        }_\textrm{pulsar's orbital motion}
      + \underbrace{ \vphantom{ \frac{ 1 }{ s } }
            v_{0,\oplus} b_\oplus \sin( \phi_\oplus - \chi_\oplus )
        }_\textrm{Earth's orbital motion}.


This shows that the scaled effective velocity can be written as the normed sum
of two sinusoids and a constant offset:

.. math::

    \frac{ \left| v_\mathrm{eff,\parallel} \right| }{ \sqrt{d_\mathrm{eff}} }
      = \left| A_\mathrm{p} \sin( \phi_\mathrm{p} - \chi_\mathrm{p} )
             + A_\oplus \sin( \phi_\oplus - \chi_\oplus ) + C
        \right|,

with

.. math::

    A_\mathrm{p} &= \frac{ 1 - s }{ s }
                    \frac{ v_\mathrm{0,p} }{ \sqrt{ d_\mathrm{eff} } }
                    b_\mathrm{p}
                  = \frac{ \sqrt{ d_\mathrm{eff} } }{ d_\mathrm{p} }
                    \frac{ K_\mathrm{p} }{ \sin( i_\mathrm{p} ) }
                    b_\mathrm{p},

    A_\oplus &= \frac{ v_{0,\oplus} }{ \sqrt{ d_\mathrm{eff} } } b_\oplus,

    C &= \frac{ 1 }{ s }
            \frac{ v_\mathrm{lens,\parallel} }{ \sqrt{ d_\mathrm{eff} } }
       - \frac{ 1 - s }{ s }
            \frac{ v_\mathrm{p,sys,\parallel} }{ \sqrt{ d_\mathrm{eff} } }
       = \frac{ 1 }{ s }
            \frac{ v_\mathrm{lens,\parallel} }{ \sqrt{ d_\mathrm{eff} } }
       - \sqrt{ d_\mathrm{eff} }
            \left[ \mu_{\alpha\ast} \sin( \xi )
                 + \mu_\delta \cos( \xi )
            \right].

.. TODO: [cf. C in infer_phys_pars, has plusminus]
