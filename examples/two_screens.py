import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import (
    CartesianRepresentation, CylindricalRepresentation)

from screens.screen import Screen, Screen1D


dp = 1.5*u.kpc
d2 = 1.0*u.kpc
d1 = 0.5*u.kpc
zhat = CartesianRepresentation(0, 0, 1)


pulsar = Screen(CartesianRepresentation([0.25, 0.25, 0.]*u.AU),
                vel=CartesianRepresentation(100., 0., 0., unit=u.km/u.s))
telescope = Screen(CartesianRepresentation([0.25, 0., 0.]*u.AU))

s1 = Screen1D(CylindricalRepresentation(1., -30*u.deg, 0.).to_cartesian(),
              [0, -1.2, 0.7]*u.AU)
s2 = Screen1D(CylindricalRepresentation(1., 90*u.deg, 0.).to_cartesian(),
              [0, 1, -0.5]*u.AU)


def plot_screen(ax, s, d, color='black', **kwargs):
    d = d.to_value(u.kpc)
    x = np.array(ax.get_xlim3d())
    y = np.array(ax.get_ylim3d())[:, np.newaxis]
    ax.plot_surface([-2.1, 2.1], [[-2.1], [2.1]], d*np.ones((2, 2)),
                    alpha=0.1, color=color)
    x = ax.get_xticks()
    y = ax.get_yticks()[:, np.newaxis]
    ax.plot_wireframe(x, y, np.broadcast_to(d, (x+y).shape),
                      alpha=0.2, color=color)
    multi_d = s.pos.shape and s.pos.shape[0] > 1
    spos = s.pos[0] if multi_d else s.pos
    ax.scatter(spos.x.to_value(u.AU), spos.y.to_value(u.AU),
               d, c=color, marker='o')
    if s.pos.shape and s.pos.shape[0] > 1:
        for spos in s.pos[1:]:
            ax.plot([0, spos.x.to_value(u.AU)], [0, spos.y.to_value(u.AU)],
                    np.ones(2) * d, c=color, linestyle=':')
            upos = spos + (zhat.cross(spos/spos.norm())
                           * ([-1.5, 1.5] * u.AU))
            ax.plot(upos.x.to_value(u.AU), upos.y.to_value(u.AU),
                    np.ones(2) * d, c=color, linestyle='-')


if __name__ == '__main__':
    ax = plt.gca(projection='3d')
    ax.set_box_aspect((1, 1, 2))
    ax.set_axis_off()
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_xticks([-2, -1, 0, 1., 2])
    ax.set_yticks([-2, -1, 0, 1., 2])
    ax.set_zticks([0, 0.5, 1., 1.5])
    plot_screen(ax, telescope, 0*u.kpc, color='blue')
    plot_screen(ax, s1, d1, color='red')
    plot_screen(ax, s2, d2, color='orange')
    plot_screen(ax, pulsar, dp, color='green')
    ax.plot(np.zeros(4), np.zeros(4),
            [0., d1.value, d2.value, dp.value], color='black')
    tpos = telescope.pos
    rt = tpos.norm()
    rthat = tpos / rt if rt > 0 else CartesianRepresentation(1, 0, 0)
    uthat = zhat.cross(rthat)
    ppos = pulsar.pos
    for s1pos in s1.pos[1:]:
        r1 = s1pos.norm()
        rho1 = r1 / d1
        r1hat = s1pos / r1
        u1hat = zhat.cross(r1hat)
        for s2pos in s2.pos[1:]:
            r2 = s2pos.norm()
            rho2 = r2 / d2
            r2hat = s2pos / r2
            u2hat = zhat.cross(r2hat)
            s21 = 1-d1/d2
            sp1 = 1-d1/dp
            sp2 = 1-d2/dp
            #     alpha1  beta1                 alpha2 beta2
            Axy = np.array(
                [[uthat.x, -rthat.x, -u1hat.x, 0., 0., 0.],
                 [uthat.y, -rthat.y, -u1hat.y, 0., 0., 0.],
                 [uthat.x, -rthat.x, 0., -s21*r1hat.x, -u2hat.x, 0.],
                 [uthat.y, -rthat.y, 0., -s21*r1hat.y, -u2hat.y, 0.],
                 [uthat.x, -rthat.x, 0., -sp1*r1hat.x, 0., -sp2*r2hat.x],
                 [uthat.y, -rthat.y, 0., -sp1*r1hat.y, 0., -sp2*r2hat.y]])
            theta1 = (s1pos - tpos) / d1
            theta2 = (s2pos - tpos) / d2
            thetap = (ppos - tpos) / dp
            Bxy = u.Quantity(
                [theta1.x,
                 theta1.y,
                 theta2.x,
                 theta2.y,
                 thetap.x,
                 thetap.y])
            Axyinv = np.linalg.inv(Axy)
            sig0, alpha0, sig1, alpha1, sig2, alpha2 = Axyinv @ Bxy
            scat1 = s1pos + sig1 * d1 * u1hat
            scat2 = s2pos + sig2 * d2 * u2hat
            ds1t = zhat + (scat1-tpos)/d1
            ds21 = zhat + (scat2-scat1)/(d2-d1)
            dps2 = zhat + (ppos-scat2)/(dp-d2)
            print("ς₀={!s}, α₀={!s} (asin(cross)={})".format(
                sig0, alpha0, np.arcsin(
                    (ds1t.cross(zhat) / ds1t.norm())
                    .dot(uthat)).to(alpha0.unit, u.dimensionless_angles())))
            print("ς₁={!s}, α₁={!s} (asin(cross)={})".format(
                sig1, alpha1, np.arcsin(
                    ds1t.cross(ds21).norm()
                    / ds1t.norm()/ds21.norm()).to(alpha1.unit,
                                                  u.dimensionless_angles())))
            print("ς₂={!s}, α₂={!s} (asin(cross)={})".format(
                sig2, alpha2, np.arcsin(
                    ds21.cross(dps2).norm()
                    / ds21.norm()/dps2.norm()).to(alpha2.unit,
                                                  u.dimensionless_angles())))
            print(f"{ds1t.dot(r1hat)=!s}, {ds1t.dot(u1hat)=!s}")
            print(f"{ds21.dot(r1hat)=!s}, {ds21.dot(u1hat)=!s}")
            print(f"{ds21.dot(r2hat)=!s}, {ds21.dot(u2hat)=!s}")
            print(f"{dps2.dot(r2hat)=!s}, {dps2.dot(u2hat)=!s}")
            x = [getattr(pos, 'x').to_value(u.AU)
                 for pos in (tpos, scat1, scat2, ppos)]
            y = [getattr(pos, 'y').to_value(u.AU)
                 for pos in (tpos, scat1, scat2, ppos)]
            z = [0., d1.value, d2.value, dp.value]
            ax.plot(x, y, z, color='black', linestyle='-.')
            ax.scatter(x[1:3], y[1:3], z[1:3], marker='o',
                       color=['red', 'orange'])
    plt.tight_layout()
