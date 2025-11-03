# Licensed under the GPLv3 - see LICENSE
"""Explore the interaction between a screen and a linear refraction ramp.

Setup is psr -> single-ramp-point -> screen -> observer.
with properties like those observed for PSR B0834+06

Displayed are the (doppler-delay) wavefield space and sky view.

Black cross direct line of sight
Blue: only through screen 1
Red:  only through screen 2
Grey: through both ramp and screens

"""
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import (
    CartesianRepresentation, CylindricalRepresentation
)
from astropy.units import Quantity as Q, Unit as U
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button

from screens.screen import Source, Screen1D, Telescope

u.set_enabled_equivalencies(u.dimensionless_angles())

# Properties for PSR B0834+06, taken from Zhu et al. 2023.
d_psr = Q(620, "pc")
vel_psr = CartesianRepresentation(
    (Q([2.2, 51.6, 0], "mas/yr") * d_psr).to("km/s"))
# Screen 1, closest to observer, with lots of points.
d_s1 = Q(389, "pc")
p1 = Q(np.linspace(-10, 10, 41), "au")
xi1_init = Q(155, "deg")
v1_init = Q(23, "km/s")  # sign flipped??
t1 = Q(0.3)  # Transparency - 30% of light passes through.
sig1 = Q(4, "au")
m1 = np.exp(-0.5*(p1/sig1)**2)
m1[5::10] *= (10**np.sign(p1))[5::10]
m1 *= np.sqrt((1-t1**2) / np.sum(np.abs(m1)**2))
# Screen 2, a refraction ramp closer to pulsar.
d_s2 = Q(415, "pc")
xi2_init = Q(136-90, "deg")  # Zhu gives angle of line of images!
p2_init = Q(10, "au")
v2_init = Q(-3, "km/s")
dalpha_dp_init = Q(-100, "mas/au")  # to start on correct side.
t2 = Q(0.9)  # Transparency - 90% of light passes through.
m2 = np.sqrt(1-t2**2)

# Standard units to assume for plot.
tau_unit = U("us")
taudot_unit = U("us/day")

tau_lims = [0, 1300]
nu_obs = Q(318.5, "MHz")
taudot_lims = (Q([-50, 50], "mHz") / nu_obs).to_value(taudot_unit)


def observations(
        xi1=xi1_init, v1=v1_init,
        xi2=xi2_init, p2=p2_init, v2=v2_init, dalpha_dp=dalpha_dp_init,
):
    """Create observations for given orientations and velocities.

    Used both to generate initial points, and to update values interactively.

    Returns a tuple with the following:
        obs0: direct from pulsar
        obs1: via screen 1
        obs2: via screen 2
        obs12: via both screens
    """
    # print(f"{xi1=}, {v1=}, {xi2=}, {p2=}, {v2=}, {dalpha_dp=}")
    # Duplicate entries to have different brightnesses.
    # TODO: implement overall magnification in .observe?
    pulsar0 = Source(vel=vel_psr, magnification=t1*t2)
    pulsar1 = Source(vel=vel_psr, magnification=t2)
    pulsar2 = Source(vel=vel_psr, magnification=t1)
    pulsar12 = Source(vel=vel_psr)
    telescope = Telescope()
    normal1 = CylindricalRepresentation(1., Q(90, "deg") - xi1, 0.).to_cartesian()
    screen1 = Screen1D(normal=normal1, p=p1, v=v1, magnification=m1)
    normal2 = CylindricalRepresentation(1., Q(90, "deg") - xi2, 0.).to_cartesian()
    dp_dalpha = 1/dalpha_dp if dalpha_dp != 0 else 1e20 / dalpha_dp.unit
    screen2 = Screen1D(normal=normal2, p=p2, v=v2, dp_dalpha=dp_dalpha,
                       magnification=m2)

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
assert np.isclose(np.sum(all_b**2), 1.)
vmin = all_b.min() * 0.5
vmax = 1.

# Create initial plot.
fig, ax = plt.subplots(figsize=(12., 8.))

# Adjust the main plot to make room for the sliders and sky view
fig.subplots_adjust(left=0.07, bottom=0.2, right=0.7, top=0.95)
ax_sky = fig.add_axes([0.75, 0.2, 0.2, 0.75])

scs = []
skys = []
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
    theta_ray = (obs.source.pos.xyz[:2]
                 / obs.distance).to_value("mas")
    sky = ax_sky.scatter(*theta_ray, marker=marker, s=size,
                         c=np.abs(obs.brightness).value, cmap=cmap,
                         norm=LogNorm(vmin=vmin, vmax=vmax))
    skys.insert(0, sky)

ax.set_xlim(*taudot_lims)
ax.set_ylim(*tau_lims)
ax_sky.set_xlim(20, -10)
ax_sky.set_aspect("equal")

fig.colorbar(mappable=sc, label=r"$|\mu|$", fraction=0.08)

ax.set_xlabel(rf"$\dot{{\tau}}$ ({taudot_unit:latex_inline})")
ax.set_ylabel(rf"$\tau$ ({tau_unit:latex_inline})")

# Add sliders.
ax_xi1 = fig.add_axes([0.5, 0.02, 0.3, 0.03])
xi1_slider = Slider(ax=ax_xi1, label=r'$\xi_{1}$ (deg)',
                    valmin=0., valmax=180, valinit=xi1_init.to_value("deg"))
ax_v1 = fig.add_axes([0.1, 0.02, 0.3, 0.03])
v1_slider = Slider(ax=ax_v1, label=r'$v_{1}$ (km/s)',
                   valmin=-50., valmax=50, valinit=v1_init.to_value("km/s"))
ax_xi2 = fig.add_axes([0.5, 0.06, 0.3, 0.03])
xi2_slider = Slider(ax=ax_xi2, label=r'$\xi_{2}$ (deg)',
                    valmin=0., valmax=180, valinit=xi2_init.to_value("deg"))
ax_v2 = fig.add_axes([0.1, 0.06, 0.3, 0.03])
v2_slider = Slider(ax=ax_v2, label=r'$v_{2}$ (km/s)',
                   valmin=-50., valmax=50, valinit=v2_init.to_value("km/s"))
ax_dalpha_dp = fig.add_axes([0.1, 0.1, 0.3, 0.03])
dalpha_dp_slider = Slider(ax=ax_dalpha_dp, label=r'$d\alpha/dp$ (mas/au)',
                          valmin=-150., valmax=150,
                          valinit=dalpha_dp_init.to_value("mas/au"))
ax_p2 = fig.add_axes([0.5, 0.1, 0.3, 0.03])
p2_slider = Slider(ax=ax_p2, label=r'$p_{2}$ (au)',
                   valmin=-20., valmax=20, valinit=p2_init.to_value("au"))


def update(val):
    """Update different scatter parts for new parameters."""
    all_obs = observations(
        xi1=Q(xi1_slider.val, "deg"),
        v1=Q(v1_slider.val, "km/s"),
        p2=Q(p2_slider.val, "au"),
        v2=Q(v2_slider.val, "km/s"),
        xi2=Q(xi2_slider.val, "deg"),
        dalpha_dp=Q(dalpha_dp_slider.val, "mas/au"),
    )
    for obs, sc, sky in zip(all_obs, scs, skys):
        tau = obs.tau.to_value(tau_unit).ravel()
        taudot = obs.taudot.to_value(taudot_unit).ravel()
        sc.set_offsets(np.vstack([taudot, tau]).T)
        theta_ray = (obs.source.pos.xyz[:2]
                     / obs.distance).to_value("mas")
        sky.set_offsets(theta_ray.T)


xi1_slider.on_changed(update)
v1_slider.on_changed(update)
xi2_slider.on_changed(update)
p2_slider.on_changed(update)
v2_slider.on_changed(update)
dalpha_dp_slider.on_changed(update)


# Add reset button to get back to initial values.
resetax = fig.add_axes([0.85, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    xi1_slider.reset()
    v1_slider.reset()
    xi2_slider.reset()
    p2_slider.reset()
    v2_slider.reset()
    dalpha_dp_slider.reset()


button.on_clicked(reset)

fig.show()
