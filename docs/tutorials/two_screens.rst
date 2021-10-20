***********************************
Simulating a multiple-screen system
***********************************

This tutorial describes how to generate synthetic data corresponding to a
single-dish observation of a pulsar whose radiation is scattered by multiple
one-dimensional screens using the :py:mod:`screens.screen` module. It explains
how to set up :py:class:`~screens.screen.Screen1D` objects from patterns seen
in observations, such that the synthetic data roughly match the observations.
For the basics of how to use the :py:class:`~screens.screen.Screen1D` class and
the :py:meth:`~screens.screen.Screen.observe` method, please refer to the
:doc:`preceding tutorial <screen1d>`.

The numerical values in the example explored in this tutorial correspond to a
model for the scattering of pulsar PSR B0834+06, specifically `Hengrui Zhu's
solution <https://eor.cita.utoronto.ca/penwiki/User:Hzhu#B0834_Paper_Status>`_,
which is based on that of `Liu et al. (2016)
<https://ui.adsabs.harvard.edu/abs/2016MNRAS.458.1289L/abstract>`_.

The combined codeblocks in this tutorial can be downloaded as a Python script
and as a Jupyter notebook:

:Python script:
    :jupyter-download:script:`two_screens.py <two_screens>`
:Jupyter notebook:
    :jupyter-download:notebook:`two_screens.ipynb <two_screens>`


Preliminaries
=============

Imports.

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from astropy import units as u
    from astropy import constants as const
    from astropy.time import Time
    from astropy.coordinates import (
        SkyCoord, SkyOffsetFrame, EarthLocation, CartesianRepresentation,
        CylindricalRepresentation, UnitSphericalRepresentation
    )

    from screens.screen import Source, Screen1D, Telescope
    from screens.fields import phasor

Create a random number generator.

.. jupyter-execute::

    rng = np.random.default_rng(seed=12345)

Define a handy function to help create an `extent` for
:py:func:`matplotlib.pyplot.imshow`.

.. jupyter-execute::

    def axis_extent(x):
        x = x.ravel().value
        dx = x[1]-x[0]
        return x[0]-0.5*dx, x[-1]+0.5*dx

Define a matrix to transform between coordinate systems.

.. jupyter-execute::

    xyz2yzx = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])


The pulsar
==========

Set the pulsar's distance, sky coordinates, and proper motion.

.. jupyter-execute::

    d_p = 0.620 * u.kpc

    psr_coord = SkyCoord('08h37m5.644606s +06d10m15.4047s',
                         distance=d_p,
                         pm_ra_cosdec=2.16 * u.mas / u.yr,
                         pm_dec=51.64 * u.mas / u.yr)

    psr_frame = SkyOffsetFrame(origin=psr_coord)

Get the pulsar's velocity in the correct format.

.. jupyter-execute::

    vel_psr = (psr_coord
               .transform_to(psr_frame)
               .velocity
               .to_cartesian()
               .transform(xyz2yzx))

Create the :py:class:`~screens.screen.Source` object for the pulsar.

.. jupyter-execute::

    pulsar = Source(vel=vel_psr)


The telescope
=============

Set the time of the observation and the location of the telescope. Use these to
get the telescope's velocity. The velocity needs to be given as a

.. jupyter-execute::

    tel_loc = EarthLocation('66°45′10″W', '18°20′48″N')

    t_obs = Time(53712.29719907, format='mjd', scale='tai')
    
    vel_tel = (tel_loc
               .get_gcrs(t_obs)
               .transform_to(psr_frame)
               .velocity
               .to_cartesian()
               .transform(xyz2yzx))

Create the :py:class:`~screens.screen.Telescope` object.

.. jupyter-execute::

    telescope = Telescope(vel=vel_tel)


The screens
===========


The lens properties
-------------------

Distances of the two screens from Earth.

.. jupyter-execute::

    d_s1 = 0.389 * u.kpc
    d_s2 = 0.415 * u.kpc

The screen angles, defined as the position angle of the line of lensed images,
measured eastward from the celestial north.

.. jupyter-execute::

    xi1 = 154.8 * u.deg
    xi2 =  46.1 * u.deg

Lens velocities along the line of lensed images (i.e., the component of the
lens velocity in the direction defined by the angles :math:`\xi_1` and
:math:`\xi_2`).

.. jupyter-execute::

    v_lens1 = 23.1 * u.km / u.s
    v_lens2 = -3.3 * u.km / u.s


The positions of main screen's images
-------------------------------------

For screen 1 (the screen responsible for the main parabola in the secondary
spectrum), we want to derive the positions of the images on the screen from the
:math:`f_\mathrm{D}` coordinates of the apexes of the inverted arclets.

.. jupyter-execute::

    fd1 = [
        -15.93352884, -15.05376344, -14.46725318, -13.58748778,
        -13.00097752, -12.41446725, -11.82795699,  -9.77517107,
         -8.30889541,  -5.37634409,  -3.61681329,  -2.15053763,
         -1.27077224,  -0.09775171,   1.07526882,   1.95503421,
          4.5943304 ,   5.4740958 ,   7.52688172,   9.28641251,
         10.45943304,  15.15151515,
    ] * u.mHz

These could be converted to :math:`\theta` angles using the main parabola's
curvature parameter :math:`\eta`, but since we have already set the screen's
distance and velocity, it's better to do the conversion self-consistently using
the screen's effective velocity.

First, get the component of the pulsar's and the telescope's (i.e., Earth's)
sky-plane velocity in the direction of the line of lensed images
(see :doc:`gen_velocities` for further explanation).

.. jupyter-execute::

    lens1_frame = SkyOffsetFrame(origin=psr_coord, rotation=xi1)

    v_psr1 = psr_coord.transform_to(lens1_frame).velocity.d_z

    v_tel1 = (tel_loc
              .get_gcrs(t_obs)
              .transform_to(lens1_frame)
              .velocity
              .d_z)

Then, compute effective velocity associated with the main screen.

.. jupyter-execute::

    s1 = 1. - d_s1 / d_p
    v_eff1 = 1. / s1 * v_lens1 - (1. - s1) / s1 * v_psr1 - v_tel1

As a sanity check, we can verify that the curvature :math:`\eta` corresponds to
the value measured from the secondary spectrum.

.. jupyter-execute::

    nu_obs = 318. * u.MHz
    lambda_obs = const.c / nu_obs

    d_eff1 = (1. - s1) / s1 * d_p

    eta1 = lambda_obs**2 * d_eff1 / (2. * const.c * v_eff1**2)

    eta1.to(u.s**3)

Then, convert the listed :math:`f_\mathrm{D}` coordinates to angles
:math:`\theta`, and subsequently to positions on the screen (i.e., coordinates
along the line of lensed images).

.. jupyter-execute::

    theta1 = (fd1 / v_eff1 * lambda_obs
             ).to(u.mas, equivalencies=u.dimensionless_angles())

    pos1 = (theta1 * d_s1).to(u.au, equivalencies=u.dimensionless_angles())


The magnifications of main screen's images
------------------------------------------

The magnifications of the images on the main screen will be derived from the
normalized brightness of the points along the main parabola with the
:math:`f_\mathrm{D}` coordinates listed above. We set random angles for the
unknown intrinsic phase due to the lens.

.. jupyter-execute::

    brightness1 = [
         1.203809  ,  1.65880546,  1.60188394,  1.45380177,  1.37484462,
         0.98746569,  1.21659379,  8.98523653,  8.50580556,  6.47967157,
        22.03764475, 26.32474627, 28.04993397, 27.7825562 , 22.63727646,
        21.20465725, 40.38307175, 18.76022889, 10.79893695,  6.31275872,
         5.01528948,  0.21360035,
    ] * u.dimensionless_unscaled

    phase1 = rng.random(len(brightness1)) * 2.*np.pi

    magnification1 = brightness1 / brightness1.max() * np.exp(1j*phase1)


Constructing the main screen
----------------------------

Create the :py:class:`~screens.screen.Screen1D` object for the main screen.

.. jupyter-execute::

    normal1 = CylindricalRepresentation(1., 90.*u.deg - xi1, 0.).to_cartesian()

    screen1 = Screen1D(normal=normal1,
                       p=pos1,
                       v=v_lens1,
                       magnification=magnification1)


Constructing the secondary screen
---------------------------------

For the secondary screen, we manually set the position and magnification of the
single image.

.. jupyter-execute::

    pos2 = [9.1652957] * u.au
    magnification2 = 0.1

Create the :py:class:`~screens.screen.Screen1D` object for the second screen.

.. jupyter-execute::

    normal2 = CylindricalRepresentation(1., 90.*u.deg - xi2, 0.).to_cartesian()

    screen2 = Screen1D(normal=normal2,
                       p=pos2,
                       v=v_lens2,
                       magnification=magnification2)


Generating observations
=======================

Use the :py:meth:`~screens.screen.Screen.observe` method to generate two sets
of optical paths: one of radiation scattered only by the main screen (resulting
in the main parabola) and one of radiation scattered by both screens (yielding
the millisecond feature).

.. jupyter-execute::

    obs1 = telescope.observe(
        source=screen1.observe(source=pulsar, distance=d_p-d_s1),
        distance=d_s1)

    obs2 = telescope.observe(
        source=screen1.observe(
            source=screen2.observe(source=pulsar, distance=d_p-d_s2),
            distance=d_s2-d_s1),
        distance=d_s1)


Making the dynamic spectrum
===========================

Define the observing frequencies and times. Make one a column vector and the
other a row vector, so they will be broadcast against one another correctly.

.. jupyter-execute::

    t = np.linspace(0, 45*u.min, 300)[:, np.newaxis]
    f = np.linspace(318.*u.MHz, 319.*u.MHz, 3000)

The :py:class:`~screens.screen.Screen1D` class assumes that the linear features
that cause the images on the lens continue indefinitely. Hence, to restrict the
extent of the lens that causes the millisecond feature, we have to use a little
hack: we can select optical paths in ``obs2`` based on their positions at the
main screen. For these positions to be available, they need to be computed
(triggered by the first line).

.. jupyter-execute::

    obs2._paths
    bool_on_lens2 = obs2.source.pos.x.squeeze() < 7. * u.au

Find the geometric delays as a function of time from the ``tau`` and ``taudot``
attributes of ``obs1`` and ``obs2``. Create a single list with all optical
paths, using only the ones from ``obs2`` that were selected above.

.. jupyter-execute::

    tau0 = np.hstack([obs1.tau.ravel(),
                      obs2.tau.ravel()[bool_on_lens2]])
    taudot = np.hstack([obs1.taudot.ravel(),
                        obs2.taudot.ravel()[bool_on_lens2]])

    tau_t = (tau0[:, np.newaxis, np.newaxis]
            + taudot[:, np.newaxis, np.newaxis] * t)

Compute the dynamic wavefield and then the dynamic spectrum.

.. jupyter-execute::

    ph = phasor(f, tau_t)

    brightness = np.hstack([obs1.brightness.ravel(),
                            obs2.brightness.ravel()[bool_on_lens2]])

    dynwave = ph * brightness[:, np.newaxis, np.newaxis]

    dynspec = np.abs(dynwave.sum(0))**2

Plot the dynamic spectrum.

.. jupyter-execute::

    plt.figure(figsize=(12., 8.))

    plt.imshow(dynspec.T,
               origin='lower', aspect='auto', interpolation='none',
               cmap='Greys', extent=axis_extent(t) + axis_extent(f), vmin=0.)
    plt.xlabel(rf"time $t$ ({t.unit.to_string('latex')})")
    plt.ylabel(rf"frequency $f$ ({f.unit.to_string('latex')})")

    cbar = plt.colorbar()
    cbar.set_label('normalized intensity')


Making the secondary spectrum
=============================

Compute the conjugate spectrum, the conjugate variables, and then the secondary
spectrum.

.. jupyter-execute::

    conspec = np.fft.fft2(dynspec)
    conspec /= conspec[0, 0]
    conspec = np.fft.fftshift(conspec)

    tau = np.fft.fftshift(np.fft.fftfreq(f.size, f[1]-f[0])).to(u.us)
    fd = np.fft.fftshift(np.fft.fftfreq(t.size, t[1]-t[0])).to(u.mHz)

    secspec = np.abs(conspec)**2

Plot the secondary spectrum.

.. jupyter-execute::

    plt.figure(figsize=(12., 8.))

    plt.imshow(secspec.T.value,
               origin='lower', aspect='auto', interpolation='none',
               cmap='Greys', extent=axis_extent(fd) + axis_extent(tau),
               norm=LogNorm(vmin=1.e-9, vmax=1.))

    plt.xlim(-50., 50.)
    plt.ylim(0., 1300.)
    plt.xlabel(r"differential Doppler shift $f_\mathrm{{D}}$ "
               rf"({fd.unit.to_string('latex')})")
    plt.ylabel(r"relative geometric delay $\tau$ "
               rf"({tau.unit.to_string('latex')})")

    cbar = plt.colorbar()
    cbar.set_label('normalized power')

    plt.show()

The arclet apexes, to be overplotted on the observed data.

.. jupyter-execute::

    plt.figure(figsize=(12., 8.))

    plt.scatter((nu_obs * taudot).to(u.mHz), tau0.to(u.us),
                c=np.abs(brightness).value, s=5, cmap='Blues',
                norm=LogNorm(vmin=1.e-4, vmax=1.))

    plt.xlim(-50., 50.)
    plt.ylim(0., 1300.)
    plt.xlabel(r"differential Doppler shift $f_\mathrm{{D}}$ "
               rf"({fd.unit.to_string('latex')})")
    plt.ylabel(r"relative geometric delay $\tau$ "
               rf"({tau.unit.to_string('latex')})")

    cbar = plt.colorbar()
    cbar.set_label('normalized power')

    plt.show()
