************************************
Single screen synthetic data example
************************************

This tutorial describes how to generate synthetic data corresponding to a
single one-dimensional scattering screen. The schematic below shows roughly
what the screen in this examplewill look like.

.. plot::

    from astropy import units as u
    import matplotlib.pyplot as plt
    from screens.visualization import make_sketch

    theta = [-4., -1., 0., 2.] << u.mas

    plt.figure(figsize=(12., 3.))
    make_sketch(theta)
    plt.show()


Preliminaries
=============

Start with some standard imports.

.. plot::
    :include-source:
    :context:

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from astropy import units as u
    from astropy import constants as const

Import a colormap to use for phases. Preferably import the colormap from
the local file ``hue_cycle_cmap``, available for download
:download:`here <./hue_cycle_cmap.py>`.
The ``hsv`` colormap is used as fallback.

.. plot::
    :include-source:
    :context:

    try:
        from hue_cycle_cmap import cmap as phasecmap
    except ImportError:
        phasecmap = cm.hsv

Also define a function to make two-dimensional phase-intensity colorbars,
for plotting the electric field.

.. plot::
    :include-source:
    :context:

    def phase_intensity_colorbar(fig, ax, phasecmap, ampmax=1.):

        cbar = fig.colorbar(cm.ScalarMappable(), ax=ax, aspect=7.5)
        cbar_pos = fig.axes[-1].get_position()
        cbar.remove()

        nph = 36
        nalpha = 256
        phases = np.linspace(-np.pi, np.pi, nph, endpoint=False) + np.pi/nph
        alphas = np.linspace(0., 1., nalpha, endpoint=False) + 0.5/nalpha
        phasegrid, alphagrid = np.meshgrid(phases, alphas)

        cax = fig.add_axes(cbar_pos)
        cax.imshow(phasegrid, alpha=alphagrid,
                   origin='lower', aspect='auto', interpolation='none',
                   cmap=phasecmap, extent=[-np.pi, np.pi, 0., ampmax])
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
        cax.yaxis.tick_right()
        cax.yaxis.set_label_position('right')
        cax.set_xticks([-np.pi, 0., np.pi])
        cax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        cax.set_xlabel('phase (rad)')
        cax.set_ylabel('normalized intensity')


Setting up a scattering screen
==============================

Set up the screen by defining angles of the scattering points parallel to
:math:`\theta_\parallel` and perpendicular to :math:`\theta_\perp` the
direction of effective velocity of the line of sight along the screen.
For this example, we choose the one-dimensional screen to be perfectly alligned
with the effective velocity vector, so the perpedicular angles are all zero.

.. plot::
    :include-source:
    :context:

    th_par = [-4., -1., 0., 2.] << u.mas
    th_perp = np.zeros_like(th_par)

Create a complex magnification for each of the scattering points (defining the
amplitude and phase of the lens image). Normalise them so the amplitudes add up
to unity.

.. plot::
    :include-source:
    :context:

    magnification = [-0.1 - 0.1j,
                     0.7 - 0.3j,
                     1.,
                     0.3 + 0.3j]
    magnification /= np.sqrt((np.abs(magnification)**2).sum())

Have a look at the lens, using a scatter plot where the size of the points
shows the amplitude of the magnifications and their colour shows the phase.

.. plot::
    :include-source:
    :context:

    plt.figure(figsize=(12., 3.))
    plt.scatter(th_par, np.zeros_like(th_par),
                s=np.abs(magnification)*2000., c=np.angle(magnification),
                cmap=phasecmap, vmin=-np.pi, vmax=np.pi)

    plt.xlabel(rf"$\theta_\parallel$ ({th_par.unit.to_string('latex')})")
    plt.ylabel(rf"$\theta_\perp$ ({th_par.unit.to_string('latex')})")

    cbar = plt.colorbar(aspect=7.5)
    cbar.set_label('phase (rad)')
    cbar.set_ticks([-np.pi, -np.pi/2., 0., np.pi/2., np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    plt.show()


Set up observing parameters
===========================

Set the parameters that describe the observation:
the central observing frequency :math:`f_\mathrm{obs}`,
the bandpass :math:`\Delta f`,
the observation length :math:`\Delta t`,
the number of frequency channels :math:`n_f`,
and the number of time bins :math:`n_t`.

.. plot::
    :include-source:
    :context: close-figs

    fobs = 316. * u.MHz
    delta_f = 2. * u.MHz
    delta_t = 90. * u.minute
    nf = 200
    nt = 180

Set up a grid of observing frequencies and times.
Then make the frequency grid a row vector and the time grid a column vector,
so they will be broadcast against each other correctly.

.. plot::
    :include-source:
    :context:

    f = (fobs + np.linspace(-0.5*delta_f, 0.5*delta_f, nf, endpoint=False)
         + 0.5*delta_f/nf)
    t = np.linspace(0.*u.minute, delta_t, nt, endpoint=False) + 0.5*delta_t/nt

    f, t = np.meshgrid(f, t, sparse=True)

Already define an ``extent`` for plotting the electric field and dynamic
spectrum.

.. plot::
    :include-source:
    :context:

    ds_extent = (t[0][0].value - 0.5*(t[1][0].value - t[0][0].value),
                 t[-1][0].value + 0.5*(t[1][0].value - t[0][0].value),
                 f[0][0].value - 0.5*(f[0][1].value - f[0][0].value),
                 f[0][-1].value + 0.5*(f[0][1].value - f[0][0].value))


Generate the electric field
===========================

Set the parameters of the system: the effective distance :math:`d_\mathrm{eff}`
and the effective proper motion :math:`\mu_\mathrm{eff}`.

.. plot::
    :include-source:
    :context:

    d_eff = 0.5 * u.kpc
    mu_eff = 50. * u.mas / u.yr

Create electric fields for each of the scattering points, given by

.. math::

    E_i(f, t) = \mu_k \exp \left[ j f \frac{d_\mathrm{eff}}{2 c}
                                  \theta_i^2 \right]

.. plot::
    :include-source:
    :context:
    
    th_par_t = th_par[:,np.newaxis,np.newaxis] + mu_eff * t
    theta_t_squared = th_par_t**2 + th_perp[:,np.newaxis,np.newaxis]**2
    tau_t = (((d_eff / (2*const.c)) * theta_t_squared)
             .to(u.s, equivalencies=u.dimensionless_angles()))

    phasor = np.exp(1j * (f * tau_t * u.cycle).to_value(u.rad))
    efields = phasor * magnification[:,np.newaxis,np.newaxis]

.. note::

    The ``screens`` package has a built-in function to quickly generate a cube
    of electric fields from a one-dimensional lens.

    .. code-block:: python

        from screens.fields import dynamic_field

        efields = dynamic_field(th_par, th_perp, magnification,
                                d_eff, mu_eff, f, t)

Have a look at the electric fields associated with the individual scattered
images.

.. plot::
    :include-source:
    :context: close-figs

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12., 8.))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    for i in range(efields.shape[0]):
        ax = axes.flat[i]
        ax.imshow(np.angle(efields[i,...]).T,
                  alpha=(np.abs(magnification[i])
                         / np.max(np.abs(magnification))),
                  origin='lower', aspect='auto', interpolation='none',
                  cmap=phasecmap, extent=ds_extent, vmin=-np.pi, vmax=np.pi)
        ax.set_title(rf"$\theta_\parallel = {th_par[i].value:.0f}$"
                    rf" {th_par.unit.to_string('latex')}")
        ax.set_xlabel(rf"time $t$ ({t.unit.to_string('latex')})")
        ax.set_ylabel(rf"frequency $f$ ({f.unit.to_string('latex')})")

    phase_intensity_colorbar(fig, axes, phasecmap,
                             ampmax=np.max(np.abs(magnification)))

    plt.show()

The electric fields corresponding to the individual scattering points still
have to be summed to create the electric field at the telescope.

.. plot::
    :include-source:
    :context: close-figs

    efield = efields.sum(axis=0)

Plot the combined electric field.

.. plot::
    :include-source:
    :context:

    fig = plt.figure(figsize=(12., 8.))
    ax = plt.subplot(111)
    plt.imshow(np.angle(efield).T,
               alpha=(np.abs(efield).T / np.max(np.abs(efield))),
               origin='lower', aspect='auto', interpolation='none',
               cmap=phasecmap, extent=ds_extent, vmin=-np.pi, vmax=np.pi)
    plt.title('electric field')
    plt.xlabel(rf"time $t$ ({t.unit.to_string('latex')})")
    plt.ylabel(rf"frequency $f$ ({f.unit.to_string('latex')})")

    phase_intensity_colorbar(fig, ax, phasecmap,
                             ampmax=np.max(np.abs(efield)))

    plt.show()


Create the dynamic spectrum
===========================

The dynamic spectrum is the square modulus of the summed electric field.

.. plot::
    :include-source:
    :context: close-figs

    dynspec = np.abs(efield)**2


Now, show the dynamic spectrum.

.. plot::
    :include-source:
    :context:

    # Plot dynamic spectrum
    plt.figure(figsize=(12., 8.))
    plt.imshow(dynspec.T,
               origin='lower', aspect='auto', interpolation='none',
               cmap='Greys', extent=ds_extent, vmin=0.)
    plt.title('dynamic spectrum')
    plt.xlabel(rf"time $t$ ({t.unit.to_string('latex')})")
    plt.ylabel(rf"frequency $f$ ({f.unit.to_string('latex')})")

    cbar = plt.colorbar()
    cbar.set_label('normalized intensity')

    plt.show()

