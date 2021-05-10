************************
Using the Screen1D Class
************************

This tutorial describes how to generate synthetic data corresponding to a
single-dish observation of a pulsar whose radiation is scattered by a single
one-dimensional screens. It explains how to use the :py:mod:`screens.screen`
module to quickly set up one-dimensional scattering screens and generate
synthetic observations of the pulsar through such screens.

The combined codeblocks in this tutorial can be downloaded as a Python script
and as a Jupyter notebook:

:Python script:
    :jupyter-download:script:`screen1d.py <screen1d>`
:Jupyter notebook:
    :jupyter-download:notebook:`screen1d.ipynb <screen1d>`


Preliminaries
=============

Imports.

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from astropy import units as u
    from astropy.coordinates import (
        CartesianRepresentation, CylindricalRepresentation,
        UnitSphericalRepresentation
    )
    
    from screens.screen import Source, Screen1D, Telescope
    from screens.fields import phasor

Define a handy function to create extents to use with imshow.

.. jupyter-execute::

    def axis_extent(x):
        x = x.ravel().value
        dx = x[1]-x[0]
        return x[0]-0.5*dx, x[-1]+0.5*dx


Set the system's parameters
===========================

Set the parameters of the system: the distances to the pulsar
:math:`d_\mathrm{p}` and to the screen :math:`d_\mathrm{s}`, and the velocity
of the pulsar :math:`v_\mathrm{p}` (we'll assume the scattering screen and
Earth to be at rest).

.. jupyter-execute::

    d_p = 0.5 * u.kpc
    d_s = 0.25 * u.kpc
    v_p = -300. * u.km/u.s

Set the positions of the lensed images in the lens plane, their complex
magnifications, and the angle of the line of images in the reference frame.

.. jupyter-execute::

    scr1_pos = np.array([-1., -0.25, 0., 0.5]) << u.au
    
    scr1_magnification = np.array([-0.1 - 0.1j,
                                    0.7 - 0.3j,
                                    1.,
                                    0.3 + 0.3j])
    scr1_magnification /= np.sqrt((np.abs(scr1_magnification)**2).sum())

    scr1_angle = 67. * u.deg


Set up the system using :py:mod:`screens.screen`
================================================

Create the scattering screen using the :py:class:`~screens.screen.Screen1D`
class. This requires setting a normal vector ``normal`` that defines the
direction of the line of images, setting the positions ``p`` of the images
along the line defined by the normal, and the velocities ``v`` of the images
along that line (in this case all images have the same velocity, zero).

.. jupyter-execute::

    scr1_normal = CylindricalRepresentation(1., scr1_angle, 0.).to_cartesian()

    scr1 = Screen1D(normal=scr1_normal, p=scr1_pos, v=0.*u.km/u.s,
                    magnification=scr1_magnification)

Create the pulsar using the :py:class:`~screens.screen.Source` class
and the telescope using the :py:class:`~screens.screen.Telescope` class.
Set their positions and velocities. and the (scaled) brightness pf the pulsar
using the ``magnification`` attribute.

.. jupyter-execute::

    pulsar = Source(pos=CartesianRepresentation([0., 0., 0.]*u.AU),
                    vel=CartesianRepresentation(v_p.value, 0., 0.,
                                                unit=u.km/u.s))

    telescope = Telescope(pos=CartesianRepresentation([0., 0., 0.]*u.AU),
                          vel=CartesianRepresentation(0., 0., 0.,
                                                      unit=u.km/u.s))

Let's have a quick look at the objects we just created.

.. jupyter-execute::

    print(pulsar)
    print(telescope)
    print(scr1)


Using the :py:meth:`~screens.screen.Screen.observe` method
==========================================================

We can use the :py:meth:`~screens.screen.Screen.observe` method of
``telescope``, for example to simulate a direct observation of the pulsar
(i.e., ignoring the screen for now):

.. jupyter-execute::

    telescope.observe(pulsar, distance=d_p)

This returns another :py:class:`~screens.screen.Telescope` object, but one that
has ``source`` and ``distance`` attributes.

At this point, it's useful to be aware of the inheritance of the classes in the
:py:mod:`screens.screen` module:

.. inheritance-diagram:: screens.screen
    :top-classes: screens.screen.Source
    :parts: 1

There are a few things to make note of:

- All objects have the same parent class: :py:class:`~screens.screen.Source`.
- The :py:meth:`~screens.screen.Screen.observe` method is carried by the
  :py:class:`~screens.screen.Screen` class, of which
  :py:class:`~screens.screen.Telescope` is a subclass.
- :py:class:`~screens.screen.Screen1D` is also a subclass of
  :py:class:`~screens.screen.Screen`, so it can also use the 
  :py:meth:`~screens.screen.Screen.observe` method
  (i.e., the lenses on a screen can observe a source).
- :py:class:`~screens.screen.Screen` is a subclass of
  :py:class:`~screens.screen.Source`, so a screen (or rather, the images on it)
  can also act a source of radiation that can be observed.


.. jupyter-execute::

    obs1 = telescope.observe(scr1.observe(pulsar, distance=d_p-d_s),
                             distance=d_s)


Making the dynamic spectrum
===========================

Define the observing frequencies and times. Make sure they can be broadcast
against one another correctly.

.. jupyter-execute::

    t = np.linspace(0, 90*u.min, 180)[:, np.newaxis]
    f = np.linspace(315*u.MHz, 317*u.MHz, 200)

Compute the dynamic wavefield (using the :py:func:`screens.fields.phasor`
function) and then the dynamic spectrum.

.. jupyter-execute::

    # Create dynamic spectrum using delay for each path.
    tau_t = (obs1.tau[:, np.newaxis, np.newaxis]
        + obs1.taudot[:, np.newaxis, np.newaxis] * t)

    ph = phasor(f, tau_t)
    dynwave = ph * obs1.brightness[:, np.newaxis, np.newaxis]

    # Calculate and show dynamic spectrum.
    dynspec = np.abs(dynwave.sum(0))**2

Plot the dynamic spectrum.

.. jupyter-execute::

    plt.figure(figsize=(12., 8.))

    plt.imshow(dynspec.T, cmap='Greys',
               extent=axis_extent(t) + axis_extent(f), vmin=0.,
               origin='lower', interpolation='none', aspect='auto')
    plt.xlabel(t.unit.to_string('latex'))
    plt.ylabel(f.unit.to_string('latex'))

    cbar = plt.colorbar()
    cbar.set_label('normalized intensity')


Making the secondary spectrum
=============================

Compute the conjugate spectrum, the conjugate variables, and then the secondary
spectrum.

.. jupyter-execute::

    # And the conjugate spectrum, and secondary spectrum.
    conspec = np.fft.fft2(dynspec)
    conspec /= conspec[0, 0]
    conspec = np.fft.fftshift(conspec)

    tau = np.fft.fftshift(np.fft.fftfreq(f.size, f[1]-f[0])).to(u.us)
    fd = np.fft.fftshift(np.fft.fftfreq(t.size, t[1]-t[0])).to(u.mHz)

    secspec = np.abs(conspec)**2

Plot the secondary spectrum.

.. jupyter-execute::

    plt.figure(figsize=(12., 8.))

    plt.imshow(secspec.T,
               origin='lower', aspect='auto', interpolation='none',
               cmap='Greys', extent=axis_extent(fd) + axis_extent(tau),
               norm=LogNorm(vmin=1.e-4, vmax=1.))
    plt.xlim(-5., 5.)
    plt.ylim(-15., 15.)
    plt.xlabel(r"differential Doppler shift $f_\mathrm{{D}}$ "
               rf"({fd.unit.to_string('latex')})")
    plt.ylabel(r"relative geometric delay $\tau$ "
               rf"({tau.unit.to_string('latex')})")

    cbar = plt.colorbar()
    cbar.set_label('normalized power')

    plt.show()


Visualize the system
====================

.. jupyter-execute::

    def unit_vector(c):
        return c.represent_as(UnitSphericalRepresentation).to_cartesian()


    ZHAT = CartesianRepresentation(0., 0., 1., unit=u.one)


    def plot_screen(ax, s, d, color='black', **kwargs):
        d = d.to_value(u.kpc)
        x = np.array(ax.get_xlim3d())
        y = np.array(ax.get_ylim3d())[:, np.newaxis]
        ax.plot_surface([[-2.1, 2.1]]*2, [[-2.1]*2, [2.1]*2], d*np.ones((2, 2)),
                        alpha=0.1, color=color)
        x = ax.get_xticks()
        y = ax.get_yticks()[:, np.newaxis]
        ax.plot_wireframe(x, y, np.broadcast_to(d, (x+y).shape),
                        alpha=0.2, color=color)
        spos = s.normal * s.p if isinstance(s, Screen1D) else s.pos
        ax.scatter(spos.x.to_value(u.AU), spos.y.to_value(u.AU),
                d, c=color, marker='+')
        if spos.shape:
            for pos in spos:
                zo = np.arange(2)
                ax.plot(pos.x.to_value(u.AU)*zo, pos.y.to_value(u.AU)*zo,
                        np.ones(2) * d, c=color, linestyle=':')
                upos = pos + (ZHAT.cross(unit_vector(pos))
                            * ([-1.5, 1.5] * u.AU))
                ax.plot(upos.x.to_value(u.AU), upos.y.to_value(u.AU),
                        np.ones(2) * d, c=color, linestyle='-')
        elif s.vel.norm() != 0:
            dp = s.vel * 5 * u.day
            ax.quiver(spos.x.to_value(u.AU), spos.y.to_value(u.AU), d,
                    dp.x.to_value(u.AU), dp.y.to_value(u.AU), np.zeros(1),
                    arrow_length_ratio=0.05)


.. jupyter-execute::

    plt.figure(figsize=(8., 12.))
    ax = plt.subplot(111, projection='3d')
    ax.set_box_aspect((1, 1, 2))
    ax.set_axis_off()
    ax.set_xlim3d(-4, 4)
    ax.set_ylim3d(-4, 4)
    ax.set_xticks([-2, -1, 0, 1., 2])
    ax.set_yticks([-2, -1, 0, 1., 2])
    ax.set_zticks([0, d_s.value, d_p.value])
    plot_screen(ax, telescope, 0*u.kpc, color='blue')
    plot_screen(ax, scr1, d_s, color='red')
    plot_screen(ax, pulsar, d_p, color='green')
    # Connect origins
    ax.plot(np.zeros(3), np.zeros(3),
            [0., d_s.value, d_p.value], color='black')

    path_shape = obs1.tau.shape  # Also trigger calculation of pos, vel.
    tpos = obs1.pos
    scat1 = obs1.source.pos
    ppos = obs1.source.source.pos
    x = np.vstack(
        [np.broadcast_to(getattr(pos, 'x').to_value(u.AU), path_shape).ravel()
        for pos in (tpos, scat1, ppos)])
    y = np.vstack(
        [np.broadcast_to(getattr(pos, 'y').to_value(u.AU), path_shape).ravel()
        for pos in (tpos, scat1, ppos)])
    z = np.vstack(
        [np.broadcast_to(d, path_shape).ravel()
        for d in (0., d_s.value, d_p.value)])
    for _x, _y, _z in zip(x.T, y.T, z.T):
        ax.plot(_x, _y, _z, color='black', linestyle=':')
        ax.scatter(_x[1], _y[1], _z[1], marker='o',
                color='red')
