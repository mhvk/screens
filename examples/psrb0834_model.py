# Licensed under the GPLv3 - see LICENSE
"""Explore the interaction between screens like in PSR B0834+06.

This uses a regular primary screen and a single point on a second
screen, for which one can set a linear refraction ramp, with the
order: psr -> single-ramp-point -> screen -> observer.

All properties are like those observed for PSR B0834+06, as inferred
from Zhu et al. 2023.

Displayed are the (doppler-delay) wavefield space and sky view.

Black cross direct line of sight
Blue: only through screen 1
Red:  only through screen 2
Grey: through both screens

"""
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import CartesianRepresentation
from astropy.units import Quantity as Q, Unit as U
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button

from screens.screen import Source, Screen1D, Telescope

# Properties for PSR B0834+06, taken from Zhu et al. 2023.
d_psr = Q(620, "pc")
vel_psr = CartesianRepresentation(
    (Q([2.2, 51.6, 0], "mas/yr") * d_psr).to("km/s", u.dimensionless_angles()))
# Screen 1, closest to observer, with lots of points.
d_s1 = Q(389, "pc")
p1 = Q(np.linspace(-10, 10, 41), "au")
xi1_init = Q(155, "deg")
v1_init = Q(23, "km/s")  # sign flipped relative to Zhu et al.?
t1 = Q(0.3)  # Transparency - 30% of light passes through.
sig1 = Q(4, "au")  # Brightness distribution of points
m1 = np.exp(-0.5*(p1/sig1)**2)
m1[5::10] *= (10**np.sign(p1))[5::10]  # Mark some for visualization.
m1 *= np.sqrt((1-t1**2) / np.sum(np.abs(m1)**2))
# Screen 2, a refraction ramp closer to pulsar.
d_s2 = Q(415, "pc")
xi2_init = Q(136-90, "deg")  # Zhu gives angle of line of images!
p2_init = Q(10, "au")
v2_init = Q(-3, "km/s")
# w=0.7 au for β=24 mas, hence α = βdₑ/dₛ~73 mas, so could be -0.008 au/mas.
dp_dalpha_init = Q(0, "au/mas")
t2 = Q(0.9)  # Transparency - 90% of light passes through.
m2 = np.sqrt(1-t2**2)

# Standard units to assume for plot.
tau_unit = U("us")
taudot_unit = U("us/day")

# Set limits like in Fig. 2 of Zhu et al. 2023.
tau_lims = [0, 1300]
nu_obs = Q(318.5, "MHz")
taudot_lims = (Q([-50, 50], "mHz") / nu_obs).to_value(taudot_unit)


def observations(
        xi1=xi1_init, v1=v1_init,
        xi2=xi2_init, p2=p2_init, v2=v2_init, dp_dalpha=dp_dalpha_init,
):
    """Create observations for given orientations and velocities.

    Used both to generate initial points, and to update values interactively.

    Returns a tuple with the following:
        obs0: direct from pulsar
        obs1: via screen 1
        obs2: via screen 2
        obs12: via both screens
    """
    # Duplicate entries to have different brightnesses.
    pulsar0 = Source(vel=vel_psr, magnification=t1*t2)
    pulsar1 = Source(vel=vel_psr, magnification=t2)
    pulsar2 = Source(vel=vel_psr, magnification=t1)
    pulsar12 = Source(vel=vel_psr)
    telescope = Telescope()
    # Angles are E of N.
    normal1 = CartesianRepresentation(np.sin(xi1), np.cos(xi1), 0)
    screen1 = Screen1D(normal=normal1, p=p1, v=v1, magnification=m1)
    normal2 = CartesianRepresentation(np.sin(xi2), np.cos(xi2), 0)
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


def get_plot_data(all_obs):
    return [(
        obs.tau.to_value(tau_unit).ravel(),
        obs.taudot.to_value(taudot_unit).ravel(),
        (obs.source.pos.xyz[:2]
         / obs.distance).to_value("mas", u.dimensionless_angles())
    ) for obs in all_obs]


def get_s2_line(obs):
    return ((obs.source.p
             / obs.distance).to("mas", u.dimensionless_angles())
            * (n := obs.source.normal).xyz[:2, np.newaxis]
            + Q([-30, 30], "mas")
            * n.cross(CartesianRepresentation(0, 0, 1)).xyz[:2, np.newaxis])


# Get initial setup.
obs0, obs1, obs2, obs12 = all_obs = observations()

# Check that total brightness is OK, and set color scale range.
all_mag = np.hstack([obs.brightness.ravel() for obs in all_obs])
all_b = np.abs(all_mag)
assert np.isclose(np.sum(all_b**2), 1.)
vmin = all_b.min() * 0.5
vmax = 1.

# Create initial plot, adjusting it to have room for sliders and sky view.
fig, ax = plt.subplots(figsize=(12., 8.))
fig.subplots_adjust(left=0.07, bottom=0.2, right=0.7, top=0.95)
ax.set_xlim(*taudot_lims)
ax.set_ylim(*tau_lims)
ax.set_xlabel(rf"$\dot{{\tau}}$ ({taudot_unit:latex_inline})")
ax.set_ylabel(rf"$\tau$ ({tau_unit:latex_inline})")

ax_sky = fig.add_axes([0.75, 0.2, 0.2, 0.75])
ax_sky.set_xlim(20, -10)
ax_sky.set_aspect("equal")
ax_sky.set_xlabel(r"$\Delta\alpha$")
ax_sky.set_ylabel(r"$\Delta\delta$", labelpad=-5)

s2_line = ax_sky.plot(*get_s2_line(obs2), "r:")

scs = []
skys = []
# Reverse order so non-interaction points plotted on top.
for obs, (tau, taudot, theta_ray), marker, size, cmap in zip(
        all_obs[::-1],
        get_plot_data(all_obs)[::-1],
        "ooox",
        (10, 20, 20, 30),
        ("Greys", "Reds", "Blues", "Greys"),
):
    sc = ax.scatter(taudot, tau, marker=marker, s=size,
                    c=np.abs(obs.brightness).value, cmap=cmap,
                    norm=LogNorm(vmin=vmin, vmax=vmax))
    scs.insert(0, sc)
    sky = ax_sky.scatter(*theta_ray, marker=marker, s=size,
                         c=np.abs(obs.brightness).value, cmap=cmap,
                         norm=LogNorm(vmin=vmin, vmax=vmax))
    skys.insert(0, sky)

# Add colorbar.
cb = fig.colorbar(mappable=scs[-1], fraction=0.1)
cb.set_label(r"$|\mu|$", labelpad=-15)

# Add sliders and a reset button
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
ax_dp_dalpha = fig.add_axes([0.1, 0.1, 0.3, 0.03])
dp_dalpha_slider = Slider(ax=ax_dp_dalpha, label=r'$dp/d\alpha$ (au/mas)',
                          valmin=-0.15, valmax=0.15,
                          valinit=dp_dalpha_init.to_value("au/mas"))
ax_p2 = fig.add_axes([0.5, 0.1, 0.3, 0.03])
p2_slider = Slider(ax=ax_p2, label=r'$p_{2}$ (au)',
                   valmin=-11, valmax=11, valinit=p2_init.to_value("au"))
resetax = fig.add_axes([0.85, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def update(val):
    """Update plot data for new parameters."""
    all_obs = observations(
        xi1=Q(xi1_slider.val, "deg"),
        v1=Q(v1_slider.val, "km/s"),
        p2=Q(p2_slider.val, "au"),
        v2=Q(v2_slider.val, "km/s"),
        xi2=Q(xi2_slider.val, "deg"),
        dp_dalpha=Q(dp_dalpha_slider.val, "au/mas"),
    )
    s2_line[0].set_data(get_s2_line(all_obs[2]))
    for (tau, taudot, theta_ray), sc, sky in zip(
            get_plot_data(all_obs), scs, skys
    ):
        sc.set_offsets(np.stack([taudot, tau], axis=-1))
        sky.set_offsets(theta_ray.T)


def reset(event):
    xi1_slider.reset()
    v1_slider.reset()
    xi2_slider.reset()
    p2_slider.reset()
    v2_slider.reset()
    dp_dalpha_slider.reset()


xi1_slider.on_changed(update)
v1_slider.on_changed(update)
xi2_slider.on_changed(update)
p2_slider.on_changed(update)
v2_slider.on_changed(update)
dp_dalpha_slider.on_changed(update)
button.on_clicked(reset)

fig.show()
