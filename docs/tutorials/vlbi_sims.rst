***********************************
Simulating VLBI Data
***********************************

This tutorial describes how to generate synthetic data corresponding to
a VLBI observation of a pulsar whose radiation is scattered by a single
one-dimensional screen using the screens.screen module.

This simulation is based around the results of `Hengrui Zhu's work <https://arxiv.org/abs/2208.06884>`_
on PSR B0834+06

For the basics of how to use the Screen1D class and the observe()
method, please refer to the preceding tutorial.

The code used in this example can be downloaded from:

:Python script:
    :jupyter-download-script:`vlbi_sims.py <vlbi_sims>`
:Jupyter notebook:
    :jupyter-download-notebook:`vlbi_sims.ipynb <vlbi_sims>`


Import
======

Import some useful functions for simulating screens.

.. jupyter-execute::

    import numpy as np
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, SymLogNorm
    
    import astropy.constants as const
    import astropy.units as u
    from astropy.coordinates import (
        CartesianRepresentation,
        CylindricalRepresentation,
        EarthLocation,
        SkyCoord,
        SkyOffsetFrame,
    )
    from astropy.time import Time
    
    from screens.fields import phasor
    from screens.screen import Screen1D, Source, Telescope
    from screens.visualization import axis_extent

Simulation Function
===================

For convenience we have collected the simulation code into a single
function.

.. jupyter-execute::

    def simulate(observationPars, screenPars, pulsarPars, startTime):
        """
        Simulation Code
    
        Parameters
        ----------
    
        observationPars : dict
            dictionary of observation parameters
                time : Quanity Array
                    1D array of times since start of observation
                freq: Quantity Array
                    1D array of channel frequencies
                dishNames: Array
                    array of dish identifiers
                dishLocations : list[EarthLocation]
                    list of dish locations
    
        screenPars : dict
            dictionary of screen parameters
                screenDistance : Quantity
                    distance to screen from Earth
                screenOrientation : Quantity
                    orientation of line of images defined E of N
                screenSpeed : Quantity
                    the speed of the screen along the line of images
                imageNumber : int
                    the number of images to simulate
    
        pulsarPars : dict
            dictionary of pulsar parameters
                pulsarDistance : Quantity
                    distance to pulsar from Earth
                properMotion : Quanity Array
                    pulsar proper motion in [ra,dec] in angle/time
                pulsarRA : Quantity
                    pulsar position in Right Ascension
                pulsarDec : Quanity
                    pulsar position in Declination
    
        startTime : astropy.time.core.Time
            start time of the obervation
        """
    
        ## Convert time and freq for use in screens
        t = np.copy(observationPars["time"])[:, np.newaxis]
        f = np.copy(observationPars["freq"])
    
        ## Calculate useful derived quanities
        lam = const.c / observationPars["freq"].mean()
        effectiveDistance = (
            pulsarPars["pulsarDistance"]
            * screenPars["screenDistance"]
            / (pulsarPars["pulsarDistance"] - screenPars["screenDistance"])
        )
    
        fd = np.fft.fftshift(np.fft.fftfreq(t.shape[0],d=t[1]-t[0]).to(u.mHz))
        tau = np.fft.fftshift(np.fft.fftfreq(f.shape[0],d=f[1]-f[0]).to(u.us))
    
        ## Determine furthest image observable in data (tau limit)
        thetaMaxTau = np.sqrt(
            0.8 * 2 * tau.max() * const.c / effectiveDistance
        )
        offsetMaxTau = thetaMaxTau * screenPars["screenDistance"]
    
        ## Create pulsar frame
        psrCoord = SkyCoord(ra=pulsarPars["pulsarRA"], dec=pulsarPars["pulsarDec"])
        psrFrame = SkyOffsetFrame(origin=psrCoord)
        pulsarVelocity = (pulsarPars["pulsarDistance"] * pulsarPars["properMotion"]).to(
            u.km / u.s, equivalencies=u.dimensionless_angles()
        )
        pulsarVelocity = np.concatenate((pulsarVelocity, np.zeros(1) * u.km / u.s))
        pulsar = Source(vel=CartesianRepresentation(pulsarVelocity))
    
        ## Create Screen
        screenOffsets = (
            np.random.uniform(-1, 1, screenPars["imageNumber"]) * u.dimensionless_unscaled
        )
        screenOffsets[0] *= 0
        screenMagnification = np.exp(
            1j * np.random.uniform(-np.pi, np.pi, screenPars["imageNumber"])
        ) * np.exp(-np.power(screenOffsets / 10, 2) / 2)
        screenMagnification /= np.sqrt(np.sum(np.abs(screenMagnification) ** 2))
        screenOffsets *= offsetMaxTau
        
        screenNormal = CylindricalRepresentation(
            1.0, 90 * u.deg - screenPars["screenOrientation"], 0.0
        ).to_cartesian()
        
        screen = Screen1D(
            normal=screenNormal,
            p=screenOffsets,
            v=screenPars["screenSpeed"],
            magnification=screenMagnification,
        )
    
        ##observe pulsar with screen
        observeScreenPulsar = screen.observe(
            source=pulsar,
            distance=pulsarPars["pulsarDistance"] - screenPars["screenDistance"],
        )
    
        ##Lists to store
        wavefields = []
        etas = []
        UVW = []
    
        ##Determine Earth core position to correct positions
        earthCorePosition = EarthLocation(x=0 * u.m, y=0 * u.m, z=0 * u.m).get_gcrs(
            startTime + observationPars["time"].mean()
        )
        earthCorePosition = earthCorePosition.transform_to(psrFrame).cartesian
        earthCorePosition = (
            np.array(
                [
                    earthCorePosition.y.to_value(u.m),
                    earthCorePosition.z.to_value(u.m),
                    earthCorePosition.x.to_value(u.m),
                ]
            )
            * u.m
        )
        ## Loop over all dishes
        for name in observationPars["dishLocations"].keys():
            ## convert dish location to gcrs at the middle of the observation
            earthPosition = observationPars["dishLocations"][name].get_gcrs(
                startTime + observationPars["time"].mean()
            )
            ##Transform to pulsar frame
            earthPosition = earthPosition.transform_to(psrFrame).cartesian
    
            ## Get dish velocity
            earthVelocity = earthPosition.differentials["s"]
            earthVelocity = (
                np.array(
                    [
                        earthVelocity.d_y.to_value(u.km / u.s),
                        earthVelocity.d_z.to_value(u.km / u.s),
                        earthVelocity.d_x.to_value(u.km / u.s),
                    ]
                )
                * u.km
                / u.s
            )
    
            ## dish position relative to earth center in UVW
            earthPosition = (
                np.array(
                    [
                        earthPosition.y.to_value(u.m),
                        earthPosition.z.to_value(u.m),
                        earthPosition.x.to_value(u.m),
                    ]
                )
                * u.m
            )
            earthPosition -= earthCorePosition
            UVW.append(earthPosition)
    
            ## Create telescope
            telescope = Telescope(
                pos=CartesianRepresentation(earthPosition),
                vel=CartesianRepresentation(earthVelocity),
            )
            ## observe screen with telescope
            observation = telescope.observe(
                source=observeScreenPulsar, distance=screenPars["screenDistance"]
            )
    
            ##Create wavefield
            brightness = observation.brightness[:, np.newaxis, np.newaxis]
            tau0 = observation.tau[:, np.newaxis, np.newaxis]
            taudot = observation.taudot[:, np.newaxis, np.newaxis]
            tau_t = tau0 + taudot * t
            ph = phasor(f, tau_t)
            wavefields.append(np.sum(ph * brightness, axis=0).T)
    
            ##calculate curvature
            parallelVelocity = np.sum((
                telescope.vel
                + pulsar.vel * effectiveDistance / pulsarPars["pulsarDistance"]
            ).to_cartesian().xyz*screenNormal.xyz)
            parallelVelocity -= (
                screenPars["screenSpeed"] * effectiveDistance / screenPars["screenDistance"]
            )
            eta = (
                (effectiveDistance * lam**2)
                / (2 * const.c * parallelVelocity**2)
            ).to(u.s**3)
            etas.append(eta.to_value(u.s**3))
        etas = np.array(etas) * u.s**3
    
        ## Create visibilities
        baselineID = []
        baselines = []
        spectra = []
        for i, name1 in enumerate(observationPars["dishLocations"].keys()):
            for j, name2 in enumerate(observationPars["dishLocations"].keys()):
                if j >= i:
                    spec = wavefields[i] * np.conjugate(wavefields[j])
                    spectra.append(spec)
                    baselineID.append(256 * (i + 1) + j + 1)
                    baselines.append((UVW[j] - UVW[i]).to_value(u.km))
        spectra = np.array(spectra)
        wavefields = np.array(wavefields)
        baselineID = np.array(baselineID)
        baselines = np.array(baselines) * u.km
        return (spectra, etas, baselineID, baselines, wavefields)

Parameters
==========

Define simulation parameters

Pulsar
------

Parameters for the pulsar. In this simulation we use the parameters from
pulsar B0834+06.

.. jupyter-execute::

    pulsarDistance = .620 * u.kpc
    properMotion = np.array([2.16, 51.64]) * u.mas / u.year
    pulsarRA = ((8*u.hour+37*u.min+5.6485930*u.s) * (360*u.deg/(24*u.hour))).to(u.deg)
    pulsarDec = 6 * u.deg+10*u.arcmin+16.06361*u.arcsec
    pulsarPars = {
        "pulsarDistance": pulsarDistance,
        "properMotion": properMotion,
        "pulsarRA": pulsarRA,
        "pulsarDec": pulsarDec,
    }

Screen
------

Parameters for the interstellar screen. 100 images were placed on the
screen to produce nice dynamic and conjugate spectra. Other screen
parameters are based on Hengrui Zhu’s work.

.. jupyter-execute::

    imageNumber = 100
    screenDistance = .389*u.kpc
    screenOrientation = 154.8*u.deg
    screenSpeed = 23.1*u.km/u.s
    screenPars = {
        "screenDistance": screenDistance,
        "screenOrientation": screenOrientation,
        "screenSpeed": screenSpeed,
        "imageNumber": imageNumber,
    }

Observation
-----------

Observation specific parameters. For this simulation we use the Green
Bank Telescope, and the dearly missed Arecibo, and simulate 1 hour of
data on MJD 53675 for a 1 MHz band from 318 MHz to 319 MHz

.. jupyter-execute::

    dishLocations = {
        "AR": EarthLocation.of_site('arecibo'),
        "GBT" : EarthLocation.of_site('GBT'),
    }
    startTime = Time(53675,format="mjd")
    time = np.linspace(0, 60, 512) * u.min
    freq = np.linspace(318,319,1024)*u.MHz
    observationPars = {
        "time": time,
        "freq": freq,
        "dishLocations": dishLocations,
    }

Simulation
==========

Simulation of the dynamic and visibility spectra using the above
parameters. The spectra are labeled using the baselineIDs defined by
256*(dish1ID)+dish2ID, where dish1ID and dish2ID are the positions of
the the dishes in the dishLocations disctionary (starting at 1). Also
incuded are the curvatures at each station and the underlying
wavefields for diagnostic purposes.

.. jupyter-execute::

    spectra, etas, baselineIDs, baselines, wavefields = simulate(
        observationPars, screenPars, pulsarPars, startTime
    )
    dishes0 = baselineIDs // 256
    dishes1 = baselineIDs % 256
    dishNames = [name for name in dishLocations.keys()]

Looking at the the resulting spectra, we see that the
visiblity is predominantly positive, real, and very similar to the
dynamic spectra. The imaginary part is much small and contains the
normal cross hatch of positive and negative features along opposite
diagonals.

.. jupyter-execute::

    grid = plt.GridSpec(nrows=2,ncols=2)
    plt.figure(figsize=(6,6))
    plt.subplot(grid[0,0])
    plt.imshow(spectra[0].real,origin='lower',aspect='auto',extent=axis_extent(time,freq),vmin=-4,vmax=4,cmap='bwr')
    plt.ylabel(r'$\nu~\left(\rm{MHz}\right)$')
    plt.xticks([])
    plt.title(r'$I_{name}$'.replace('name',dishNames[0]))
    plt.colorbar()
    plt.subplot(grid[0,1])
    plt.imshow(spectra[2].real,origin='lower',aspect='auto',extent=axis_extent(time,freq),vmin=-4,vmax=4,cmap='bwr')
    plt.yticks([])
    plt.xticks([])
    plt.title(r'$I_{name}$'.replace('name',dishNames[1]))
    plt.colorbar()
    plt.subplot(grid[1,0])
    plt.imshow(spectra[1].real,origin='lower',aspect='auto',extent=axis_extent(time,freq),vmin=-4,vmax=4,cmap='bwr')
    plt.ylabel(r'$\nu~\left(\rm{MHz}\right)$')
    plt.xlabel(r'$t~\left(\rm{min}\right)$')
    plt.title(r'$Re\left(V_{name1,name2}\right)$'.replace('name1',dishNames[0]).replace('name2',dishNames[1]))
    plt.colorbar()
    plt.subplot(grid[1,1])
    plt.imshow(spectra[1].imag,origin='lower',aspect='auto',extent=axis_extent(time,freq),vmin=-1,vmax=1,cmap='bwr')
    plt.yticks([])
    plt.xlabel(r'$t~\left(\rm{min}\right)$')
    plt.title(r'$Im\left(V_{name1,name2}\right)$'.replace('name1',dishNames[0]).replace('name2',dishNames[1]))
    plt.colorbar()
