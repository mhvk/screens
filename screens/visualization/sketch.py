# Licensed under the GPLv3 - see LICENSE
"""Sketch the scattering screen geometry.

Use interactively with,
plt.ion()
run -i visualization/sketch.py <fig no>

Or create multiple figure files with

python3 visualization/sketch.py 'fig{}.png' 1 2 3
"""
import sys
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt


def rotate(position, angle):
    if len(position) > 2:
        results = []
        for pos in zip(position[::2], position[1::2]):
            results += list(rotate(pos, angle))
        return results
    x, y = position
    return (x * np.cos(angle) - y * np.sin(angle),
            x * np.sin(angle) + y * np.cos(angle))


def offset(position, off):
    if len(position) > 2:
        results = []
        for pos in zip(position[::2], position[1::2]):
            results += list(offset(pos, off))
        return results
    return (position[0] + off[0], position[1] + off[1])


def scale(position, scl):
    if len(position) > 2:
        results = []
        for pos in zip(position[::2], position[1::2]):
            results += list(scale(pos, scl))
        return results
    return (position[0] * scl, position[1] * scl)


class Telescope(object):
    def __init__(self, angle=0.*u.deg, f_ratio=0.4):
        self.angle = angle
        self.f_ratio = f_ratio

    def __call__(self):
        # x**2 + (fl-y)**2 = (fl+y)**2 -> x**2 = 4*fl*y
        dish_x = np.linspace(-0.5, 0.5, 51)
        dish_y = dish_x**2 / (4.*self.f_ratio)
        dish_x, dish_y = rotate((dish_x, dish_y), self.angle)
        support_x = np.array([0., 0.])
        support_y = np.array([0., -0.2])
        support_x, support_y = rotate((support_x, support_y), self.angle)
        support_x = np.hstack((support_x.value, support_x[-1].value))
        support_y = np.hstack((support_y.value, support_y[-1].value-0.6))
        return dish_x, dish_y, support_x, support_y

    def foot(self):
        _, _, x, y = self()
        return x[-1], y[-1]


def circle(n_points=61):
    with u.add_enabled_equivalencies(u.dimensionless_angles()):
        circle = np.exp(np.linspace(0., 1., n_points)*u.cycle*1j)
    return circle.real, circle.imag


def arrow(length=1., head_size=0.15):
    return (np.array([0., 0.]), np.array([length, 0.]),
            np.array([-head_size, 0., head_size]),
            np.array([length-head_size, length, length-head_size]))


def bar(length=1, head_size=0.1, head_length=None):
    half = length / 2.
    if head_length is None:
        head_length = half
    return (np.array([-half, -half + head_length]), np.array([-0., 0.]),
            np.array([-half, -half]), np.array([-head_size, head_size]),
            np.array([half, half - head_length]), np.array([-0., 0.]),
            np.array([half, half]), np.array([-head_size, head_size]))


class NeutronStar(object):
    def __init__(self, max_field_radii=[2.5, 4.],
                 open_field_radii=[1.2, 2.5],
                 light_cylinder_radius=500.):
        self.max_field_radii = max_field_radii
        self.open_field_radii = open_field_radii
        self.light_cylinder_radius = light_cylinder_radius

    def __call__(self):
        results = list(circle())
        for max_radius in self.max_field_radii:
            angle = np.arccos(np.sqrt(1./max_radius)
                              * u.dimensionless_unscaled)
            a = np.linspace(-1., 1., 61) * angle
            r = max_radius * np.cos(a)**2
            results += [r*np.cos(a), r*np.sin(a), -r*np.cos(a), r*np.sin(a)]

        a = np.arccos(np.sqrt(np.array(self.open_field_radii)
                              / self.light_cylinder_radius
                              * u.dimensionless_unscaled))
        bzz_x, bzz_y = (self.open_field_radii * np.cos(a),
                        self.open_field_radii * np.sin(a))
        results += [bzz_x, bzz_y, -bzz_x, bzz_y, -bzz_x, -bzz_y, bzz_x, -bzz_y]

        return results


def make_sketch(theta, beta=0.5, screen_y_scale=1.e7, mu_eff=50.*u.mas/u.yr,
                tels=slice(0, 1), scatters=slice(0, None),
                earth=True, pulsar=True, screens=True, direct=None,
                velocity=None, scales=False, distances=False,
                rotation=0.*u.deg, ax=None):
    """Draw a schematic of the thin-screen model of for scintillation.

    Parameters
    ----------
    theta : `~astropy.units.Quantity`
        Angles of the scattering images from the core image.
    beta : float, optional
        Fractional distance of scattering screen, measured from the pulsar
        to Earch, ``1 - d_lens / d_psr``. Default: 0.5
    screen_y_scale : float, optional
        Vertical magnification factor of scattering screen. Default: 1.e7
    mu_eff : `~astropy.units.Quantity`, optional
        Proper motion used for direction and scaling of arrow that
        indicates the pulsar's motion. Default: 50. mas/yr
    tels : slice, optional
        Slice object to select which telescopes to draw and use.
    scatters : slice, optinal
        Slice object to select which scattering points to draw and use.
    earth : bool, optional
        Whether or not to draw Earth and telescopes.
    pulsar : bool, optional
        Whether or not to draw the pulsar.
    screens : bool, optional
        Whether or not to draw the scintillation screen.
    direct : bool, optional
        Whether or not to draw direct line-or-sight beam.  Default:
        infer from whether theta includes 0.
    velocity : bool, optional
        Whether or not to include an arrow to indicate the pulsar's motion.
        Default: infer from ``mu_eff``
    scales : bool, optional
        Whether or not to include scale bars indicating physical sizes.
    distances : bool, optional
        Whether or not to include scale bars indicating distances and s.
    rotation : `~astropy.units.Quantity`, optional
        Angle to rotate entire sketch. Default: 0. deg
    ax : `~matplotlib.axes.Axes`, optional
        The Axes object in which to draw the sketch.
    """

    # Infer options
    if direct is None:
        direct = 0. in theta
    if velocity is None:
        velocity = mu_eff != 0.

    # Earth and telescopes
    earth_pos = circle()
    locations = [170., 100., 250.] * u.degree
    tel_size = 0.5
    tel_pos = []
    for location in locations:
        tel = Telescope(angle=(180.*u.deg-location))
        xf, yf = tel.foot()
        tel_pos += [offset(rotate(scale(offset(tel(), (-xf, -yf)), tel_size),
                                  rotation + location - 90.*u.deg),
                           rotate((1., 0.), rotation + location))]

    # Pulsar
    ns = NeutronStar()
    ns_x = -44.
    ns_size = 0.3
    ns_offset = rotate((ns_x, 0.), rotation)
    ns_pos = offset(rotate(scale(ns(), ns_size),
                           rotation - (np.sign(mu_eff)
                                       + (mu_eff == 0.))*60.*u.deg),
                    ns_offset)
    arrow_size = 1.5 * np.abs(mu_eff.to_value(u.mas/u.yr) / 50.)
    ns_vel = offset(rotate(offset(arrow(arrow_size), (0., 1.4)),
                           rotation + 270.*u.deg + np.sign(mu_eff)*45.*u.deg),
                    ns_offset)

    # Scattering points
    if direct:
        # Remove direct line of sight from input angles.
        # TODO: not strictly the same if the telescope is not at y=0!
        theta = theta[theta != 0]
    lens_x = (1. - beta) * ns_x
    lens_y = lens_x * np.tan(theta).value * screen_y_scale
    lens_pos = []
    for l_y in lens_y:
        lens_pos += [rotate([lens_x, l_y], rotation)]

    # Direct lines from pulsar to telescopes
    shortening = np.array([0.025, beta, 0.992])
    tel_centers = [(t[2][0], t[3][0]) for t in tel_pos]
    p2t_pos = []

    for tel_x, tel_y in tel_centers:
        p2t_pos += [[ns_offset[0] + shortening * (tel_x - ns_offset[0]),
                     ns_offset[1] + shortening * (tel_y - ns_offset[1])]]

    # Lines from pulsar to screen to telescopes
    p2s2t_pos = []
    for lens in lens_pos:
        p2s2t = []
        for p2t in p2t_pos:
            p = tuple(p.copy() for p in p2t)
            p[0][1] = lens[0]
            p[1][1] = lens[1]
            p[0][0] = (ns_offset[0]
                       + shortening[0]/shortening[1]
                       * (lens[0] - ns_offset[0]))
            p[1][0] = (ns_offset[1]
                       + shortening[0]/shortening[1]
                       * (lens[1] - ns_offset[1]))
            p2s2t += [p]
        p2s2t_pos += [p2s2t]

    # Scattering screen
    screen_size = 12.
    screen_pos = []
    for a, f, p in zip([0.5, 0.2, 0.3],
                       [1., 0.7, 1.6] * u.cycle,
                       [0.1, 0.4, 0.5] * u.cycle):
        y = (np.linspace(-screen_size/2., screen_size/2., 361)
             * u.dimensionless_unscaled)
        x = lens_x + a * np.cos(f*y + p)
        screen_pos += rotate([x, y], rotation)

    if ax is None:
        ax = plt.gca()
    ax.axison = False
    ax.set_aspect('equal')

    # Draw Earth and its telescopes
    if earth:
        ax.plot(*(_v.value for _v in earth_pos), color='blue')

    # Draw Pulsar
    if pulsar:
        ax.plot(*(_v.value for _v in ns_pos), color='black')

    # Draw pulsar's velocity arrow
    if velocity:
        ax.plot(*(_v.value for _v in ns_vel), color='black')

    # Draw telescopes
    if tels:
        ax.plot(*(_v.value for _t in tel_pos[tels] for _v in _t),
                color='black')

    # Draw direct line-of-sight beams
    if direct:
        ax.plot(*(_v.value for p2t in p2t_pos[tels] for _v in p2t),
                linestyle='dashed', color='black')

    # Draw scattering screen and scattered beams
    if screens:
        ax.plot(*(_v.value for _v in screen_pos), color='grey')
        ax.plot(*(_v.value for p2s2t in p2s2t_pos[scatters]
                  for _p in p2s2t[tels] for _v in _p),
                linestyle='dotted', color='black')

    # Draw physical scale bars
    if scales:
        earth_bar = offset(rotate(bar(2.), rotation + 90.*u.degree),
                           rotate((1.3, 0.), rotation))
        ax.plot(*(_v.value for _v in earth_bar), color='black')
        ax.annotate(xy=rotate((1.5, 0.), rotation), text='12000 km',
                    verticalalignment='center',
                    rotation=rotation.to(u.deg).value, rotation_mode='anchor')
        off_x = ns_x - ns_size * 5
        ns_bar = offset(rotate(bar(ns_size * 2), rotation + 90.*u.degree),
                        rotate((off_x, 0.), rotation))
        ax.plot(*(_v.value for _v in ns_bar), color='black')
        ax.annotate(xy=rotate((off_x - 0.3, 0.), rotation), text='~20 km',
                    verticalalignment='center',
                    horizontalalignment='right',
                    rotation=rotation.to(u.deg).value, rotation_mode='anchor')
        screen_bar = offset(rotate(bar(screen_size-0.2, head_length=1.5),
                                   rotation + 90*u.degree),
                            rotate((lens_x + 0.7, 0.), rotation))
        ax.plot(*(_v.value for _v in screen_bar), color='black')
        ax.annotate(xy=rotate((lens_x + 1., -screen_size/2), rotation),
                    text='~10 AU',
                    verticalalignment='center',
                    rotation=rotation.to(u.deg).value, rotation_mode='anchor')
        distance_bar = offset(rotate(bar(-ns_x, head_length=7.),
                                     rotation),
                              rotate((ns_x / 2., -5.), rotation))
        ax.plot(*(_v for _v in distance_bar), color='black')
        ax.annotate(xy=rotate((ns_x / 3., -5.), rotation), text='~1 kpc',
                    verticalalignment='center',
                    rotation=rotation.to(u.deg).value, rotation_mode='anchor')

    # Draw scale bars for distances
    if distances:
        bar_offset = 2.
        screen_thickness = 0.125
        d_p_bar = offset(rotate(bar(-ns_x), rotation),
                         rotate((ns_x / 2.,
                                 -(screen_size/2. + bar_offset)),
                                rotation))
        ax.plot(*(_v for _v in d_p_bar), color='black')
        ax.annotate(xy=rotate((ns_x / 2.,
                               -(screen_size/2. + bar_offset + 0.25)),
                              rotation),
                    text=r'$ d_\mathrm{p} $', ha='right', va='top')
        d_s_bar_length = -lens_x - screen_thickness
        d_s_bar = offset(rotate(bar(d_s_bar_length), rotation),
                         rotate((-d_s_bar_length / 2.,
                                 -(screen_size/2. + bar_offset - 1.)),
                                rotation))
        ax.plot(*(_v for _v in d_s_bar), color='black')
        ax.annotate(xy=rotate((-d_s_bar_length / 2.,
                               -(screen_size/2. + bar_offset - 1.25)),
                              rotation),
                    text=r'$ d_\mathrm{s} = (1 - s) d_\mathrm{p} $',
                    ha='left', va='bottom')
        d_ps_bar_length = (-ns_x) - (-lens_x) - screen_thickness
        d_ps_bar = offset(rotate(bar(d_ps_bar_length), rotation),
                          rotate((ns_x + d_ps_bar_length / 2.,
                                  -(screen_size/2. + bar_offset - 1.)),
                                 rotation))
        ax.plot(*(_v for _v in d_ps_bar), color='black')
        ax.annotate(xy=rotate((ns_x - (-d_ps_bar_length) / 2.,
                               -(screen_size/2. + bar_offset - 1.25)),
                              rotation),
                    text=r'$ s d_\mathrm{p} $', ha='left', va='bottom')

    return ax


if __name__ == '__main__':
    if len(sys.argv) < 3:
        figures = [None] if len(sys.argv) <= 1 else sys.argv[1]
        filename = None
    else:
        filename = sys.argv[1]
        figures = sys.argv[2:]

    for figure in figures:

        # defaults
        beta = 1. - 1. / 1.7
        theta = [0.08863083, -0.08863083, -0.22044899, 0.16087047] << u.rad
        screens, direct, velocity, scales = True, True, False, False

        plt.figure(figsize=(12., 3.))

        if figure == '1':
            # Just one telescope with a direct line of sight.
            tels = slice(0, 1)
            scatters = slice(0)
            screens = False
        elif figure == '2':
            # One telescope with a scattering screen.
            tels = slice(0, 1)
            scatters = slice(1, None)
        elif figure in ('3', '4'):
            # One telescope with a single scattered line
            tels = slice(0, 1)
            scatters = slice(1, 2)
            if figure == '4':
                velocity = True
        elif figure == '5':
            # One telescope with two opposite scatterers (no direct l.o.s.)
            tels = slice(0, 1)
            scatters = slice(0, 2)
            direct = False
            velocity = True
        elif figure == '9':
            beta = 1. - 1. / 1.1
            theta = [0.05743676, -0.05743676, -0.14399642, 0.10461666] << u.rad
            tels = slice(0, 1)
            scatters = slice(1, None)
        else:
            tels = slice(None)
            scatters = slice(1, None)
            scales = True

        make_sketch(theta=theta, beta=beta, screen_y_scale=1., tels=tels,
                    scatters=scatters, screens=screens, direct=direct,
                    velocity=velocity, scales=scales)

        if filename:
            plt.savefig(filename.format(figure))
            plt.clf()
        else:
            plt.draw()
