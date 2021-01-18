# Licensed under the GPLv3 - see LICENSE
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
            angle = np.arccos(np.sqrt(1./max_radius) *
                              u.dimensionless_unscaled)
            a = np.linspace(-1., 1., 61) * angle
            r = max_radius * np.cos(a)**2
            results += [r*np.cos(a), r*np.sin(a), -r*np.cos(a), r*np.sin(a)]

        a = np.arccos(np.sqrt(np.array(self.open_field_radii) /
                              self.light_cylinder_radius *
                              u.dimensionless_unscaled))
        bzz_x, bzz_y = (self.open_field_radii * np.cos(a),
                        self.open_field_radii * np.sin(a))
        results += [bzz_x, bzz_y, -bzz_x, bzz_y, -bzz_x, -bzz_y, bzz_x, -bzz_y]

        return results


def sketch(th, th_scale=2., screens=True, direct=True, velocity=True,
           scales=False, filename=None):
    """Draw a schematic of the thin-screen model of for scintillation.

    Parameters
    ----------
    th : `~astropy.units.Quantity`
        Angles of the scattering images from the core image.
    screens : bool, optional
        Whether or not to draw the scintillation screen.
    direct : bool, optional
        Whether or not to draw direct line-or-sight beam.
    velocity : bool, optional
        Whether or not to include an arrow to indicate the pulsar's motion.
    scales : bool, optional
        Whether or not to include scale bars indicating physical sizes.
    filename : string, optional
        Path to output file.
    """

    screen_y_from_th = th_scale * np.delete(th.value, np.where(th == 0))

    # Earth and telescopes
    earth_pos = circle()
    locations = [170., 100., 250.] * u.degree
    tel_size = 0.5
    tel_pos = []
    for location in locations:
        tel = Telescope(angle=(180.*u.deg-location))
        xf, yf = tel.foot()
        tel_pos += [offset(rotate(scale(offset(tel(), (-xf, -yf)), tel_size),
                                  location-90.*u.deg),
                           rotate((1., 0.), location))]

    # Pulsar
    ns = NeutronStar()
    ns_size = 0.3
    ns_offset = (-44., 0.)
    ns_pos = offset(rotate(scale(ns(), ns_size), 60.*u.degree), ns_offset)
    arrow_size = 1.5
    ns_vel = offset(rotate(offset(arrow(arrow_size), (0., 1.4)),
                           225.*u.degree), ns_offset)

    screen_x = ns_offset[0] / 1.7
    screen_y = screen_y_from_th
    shortening = np.array([0.025, 0.5, 0.992])
    tel_centers = [(t[2][0], t[3][0]) for t in tel_pos]
    p2t_pos = []

    # Lines from screen to telescopes
    for tel_x, tel_y in tel_centers:
        p2t_pos += [[ns_offset[0] + shortening * (tel_x - ns_offset[0]),
                     ns_offset[1] + shortening * (tel_y - ns_offset[1])]]

    # Lines from pulsar to screen to telescopes
    p2s2t_pos = []
    for s_y in screen_y:
        p2s2t = []
        for p2t in p2t_pos:
            p = tuple(p.copy() for p in p2t)
            p[0][1] = screen_x
            p[1][1] = s_y
            p[1][0] = s_y * shortening[0]/shortening[1]
            p2s2t += [p]
        p2s2t_pos += [p2s2t]

    # Scattering screen
    screen_pos = []
    for a, f, p in zip([0.5, 0.2, 0.3],
                       [1., 0.7, 1.6] * u.cycle,
                       [0.1, 0.4, 0.5] * u.cycle):
        y = np.linspace(-6.5, 5.5, 361) * u.dimensionless_unscaled
        x = screen_x + a * np.cos(f*y + p)
        screen_pos += [x, y]

    plt.clf()
    plt.figure(figsize=(12., 3.))
    plt.gca().axison = False
    plt.xlim(-46, 2)
    plt.ylim(-6.5, 5.5)

    # Settings
    tels = slice(0, 1)
    scatters = slice(0, None)

    # Draw Earth and its telescopes
    plt.plot(*(_v.value for _v in earth_pos), color='blue')

    # Draw Pulsar
    plt.plot(*(_v.value for _v in ns_pos), color='black')

    # Draw pulsar's velocity arrow
    if velocity:
        plt.plot(*(_v.value for _v in ns_vel), color='black')

    # Draw telescopes
    if tels:
        plt.plot(*(_v.value for _t in tel_pos[tels] for _v in _t),
                 color='black')

    # Draw direct line-of-sight beam
    if direct:
        plt.plot(*(_v.value for p2t in p2t_pos[tels] for _v in p2t),
                 linestyle='dashed', color='black')

    # Draw scattering screen and scattered beams
    if screens:
        plt.plot(*(_v.value for _v in screen_pos), color='grey')
        plt.plot(*(_v.value for p2s2t in p2s2t_pos[scatters]
                   for _p in p2s2t[tels] for _v in _p),
                 linestyle='dotted', color='black')

    # Draw physical scale bars
    if scales:
        earth_bar = offset(rotate(bar(2.), 90.*u.degree), (1.3, 0.))
        plt.plot(*(_v.value for _v in earth_bar), color='black')
        plt.annotate(xy=(1.5, 0.), s='12000 km',
                     verticalalignment='center')
        off_x = ns_offset[0] - ns_size * 5
        ns_bar = offset(rotate(bar(ns_size * 2), 90.*u.degree),
                        (off_x, 0.))
        plt.plot(*(_v.value for _v in ns_bar), color='black')
        plt.annotate(xy=(off_x - 0.3, 0.), s='~20 km',
                     verticalalignment='center',
                     horizontalalignment='right')
        screen_bar = offset(rotate(bar(11.8, head_length=1.5),
                                   90*u.degree), (screen_x + 0.7, -0.5))
        plt.plot(*(_v.value for _v in screen_bar), color='black')
        plt.annotate(xy=(screen_x + 1., -6.), s='~10 AU',
                     verticalalignment='center')
        distance_bar = offset(bar(-ns_offset[0], head_length=7.),
                              (ns_offset[0] / 2., -5.))
        plt.plot(*(_v for _v in distance_bar), color='black')
        plt.annotate(xy=(ns_offset[0] / 3., -5.), s='~1 kpc',
                     verticalalignment='center')

    if filename:
        plt.savefig(filename)
    else:
        plt.draw()
