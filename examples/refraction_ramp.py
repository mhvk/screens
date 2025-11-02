# Licensed under the GPLv3 - see LICENSE
"""Explore the interaction between a screen and a linear refraction ramp.

Setup is psr -> ramp -> screen -> observer.

Displayed is the (doppler-delay) wavefield space.

Black cross direct line of sight
Blue: only through screen 1
Grey: through ramp and screens

Adjustable are the pulsar velocity (less useful) and the angles of the
screens (direction in which the line of scattering points or
refraction is oriented, relative to the pulsar), the bending angle
gradient, as well as the screen velocity.  All velocies are relative
to the observer.

"""
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import CylindricalRepresentation
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button

from screens.screen import Source, Screen1D, Telescope

# Pulsar properties.
d_psr = 300 * u.pc
vpsr_init = 300 * u.km/u.s
# Screen 1, closest to observer, with lots of points.
d_s1 = 100 * u.pc
p1 = np.linspace(-1.0, 1.0, 41) << u.au
xi1_init = 30*u.deg
v1_init = 0.*u.km/u.s
t1 = 0.5 * u.one  # Transparency - 50% of light passes through.
m1 = np.exp(-0.5*(p1/(0.5*u.au))**2)
m1[::10] *= 10
m1 *= np.sqrt((1-t1**2) / np.sum(np.abs(m1)**2))
# Screen 2, a refraction ramp closer to pulsar.
d_s2 = 200*u.pc
xi2_init = 45*u.deg
p2 = 0 * u.au
v2 = 0 * u.km/u.s
dalpha_dp_init = 5*u.mas/u.au

# Standard units to assume for plot.
tau_unit = u.us
taudot_unit = u.us/u.day


def observations(xi1=xi1_init, v1=v1_init,
                 xi2=xi2_init, dalpha_dp=dalpha_dp_init,
                 vpsr=vpsr_init):
    """Create observations for given orientations and velocities.

    Used both to generate initial points, and to update values interactively.

    Returns a tuple with the following:
        obs0: direct from pulsar
        obs1: via screen 1
        obs2: via screen 2
        obs12: via both screens
    """
    vel_psr = CylindricalRepresentation(
        vpsr, 0.*u.deg, 0.*u.km/u.s).to_cartesian()
    # Duplicate entries to have different brightnesses.
    # TODO: implement overall magnification in .observe?
    pulsar0 = Source(vel=vel_psr, magnification=t1)
    pulsar1 = Source(vel=vel_psr, magnification=1.)
    pulsar2 = Source(vel=vel_psr, magnification=t1)
    pulsar12 = Source(vel=vel_psr)
    telescope = Telescope()
    normal1 = CylindricalRepresentation(1., xi1, 0.).to_cartesian()
    screen1 = Screen1D(normal=normal1, p=p1, v=v1, magnification=m1)
    normal2 = CylindricalRepresentation(1., xi2, 0.).to_cartesian()
    dp_dalpha = 1/dalpha_dp if dalpha_dp != 0 else 1e20 / dalpha_dp.unit
    screen2 = Screen1D(normal=normal2, p=p2, v=v2, dp_dalpha=dp_dalpha)

    obs0 = telescope.observe(source=pulsar0, distance=d_psr)

    obs1 = telescope.observe(
        source=screen1.observe(source=pulsar1, distance=d_psr-d_s1),
        distance=d_s1)

    obs2 = telescope.observe(
        source=screen2.observe(source=pulsar2, distance=d_psr-d_s2),
        distance=d_s2)

    obs12 = telescope.observe(
        source=screen1.observe(
            source=screen2.observe(source=pulsar12, distance=d_psr-d_s2),
            distance=d_s2-d_s1),
        distance=d_s1)

    return obs0, obs1, obs2, obs12


# Get initial setup.
obs0, obs1, obs2, obs12 = observations()
# Check that total brightness is OK, and set color scale range.
all_mag = np.hstack([obs.brightness.ravel()
                     for obs in (obs0, obs1, obs2, obs12)])
all_b = np.abs(all_mag)
# assert np.isclose(np.sum(all_b**2), 1.)
vmin = all_b.min()*0.3
vmax = 1.

# Create initial plot.
fig, ax = plt.subplots(figsize=(12., 8.))

scs = []
for obs, marker, size, cmap in (
        (obs0, "x", 30, "Greys"),
        (obs1, "o", 20, "Blues"),
        (obs2, "o", 20, "Reds"),
        (obs12, "o", 10, "Greys"))[::-1]:
    tau = obs.tau.to_value(tau_unit).ravel()
    taudot = obs.taudot.to_value(taudot_unit).ravel()
    sc = ax.scatter(taudot, tau, marker=marker, s=size,
                    c=np.abs(obs.brightness).value, cmap=cmap,
                    norm=LogNorm(vmin=vmin, vmax=vmax))
    scs.insert(0, sc)

fig.colorbar(mappable=sc, label=r"$|\mu|$", fraction=0.1)

ax.set_xlabel(rf"$\dot{{\tau}}$ ({taudot_unit:latex_inline})")
ax.set_ylabel(rf"$\tau$ ({tau_unit:latex_inline})")

# Adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.2)
# Add sliders.
ax_xi1 = fig.add_axes([0.5, 0.02, 0.3, 0.03])
xi1_slider = Slider(ax=ax_xi1, label=r'$\xi_{1}$ (deg)',
                    valmin=0., valmax=180, valinit=xi1_init.to_value(u.deg))
ax_v1 = fig.add_axes([0.1, 0.02, 0.3, 0.03])
v1_slider = Slider(ax=ax_v1, label=r'$v_{1}$ (km/s)',
                   valmin=-50., valmax=50, valinit=0)
ax_xi2 = fig.add_axes([0.5, 0.06, 0.3, 0.03])
xi2_slider = Slider(ax=ax_xi2, label=r'$\xi_{2}$ (deg)',
                    valmin=0., valmax=180, valinit=xi2_init.to_value(u.deg))
ax_dalpha_dp = fig.add_axes([0.1, 0.06, 0.3, 0.03])
dalpha_dp_slider = Slider(ax=ax_dalpha_dp, label=r'$d\alpha/dp$ (mas/au)',
                          valmin=-20., valmax=20,
                          valinit=dalpha_dp_init.to_value("mas/au"))
ax_vpsr = fig.add_axes([0.1, 0.1, 0.7, 0.03])
vpsr_slider = Slider(ax=ax_vpsr, label=r'$v_\mathrm{psr}$ (km/s)',
                     valmin=0., valmax=500,
                     valinit=vpsr_init.to_value(u.km/u.s))


def update(val):
    """Update different scatter parts for new parameters."""
    all_obs = observations(
        xi1=xi1_slider.val * u.deg,
        v1=v1_slider.val * u.km/u.s,
        xi2=xi2_slider.val * u.deg,
        dalpha_dp=dalpha_dp_slider.val * u.mas/u.au,
        vpsr=vpsr_slider.val * u.km/u.s)
    for obs, sc in zip(all_obs, scs):
        tau = obs.tau.to_value(tau_unit).ravel()
        taudot = obs.taudot.to_value(taudot_unit).ravel()
        sc.set_offsets(np.vstack([taudot, tau]).T)


xi1_slider.on_changed(update)
v1_slider.on_changed(update)
xi2_slider.on_changed(update)
dalpha_dp_slider.on_changed(update)
vpsr_slider.on_changed(update)


# Add reset button to get back to initial values.
resetax = fig.add_axes([0.85, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    xi1_slider.reset()
    v1_slider.reset()
    xi2_slider.reset()
    dalpha_dp_slider.reset()
    vpsr_slider.reset()


button.on_clicked(reset)

fig.show()
