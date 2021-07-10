"""Set of functions to plot labelled angles, vectors, ellipses, etc.

adapted from Daniel Baker
"""

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def blank_diagram(fig_width=8.,
                  fig_height=8.,
                  bg_color="transparent",
                  color="black",
                  box=True):
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes((0., 0., 1., 1.))
    ax.set_xlim(0., 2.)
    ax.set_ylim(0., 2.)
    if not bg_color == "transparent":
        ax.set_facecolor(bg_color)

    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.tick_params(labelbottom=False,
                   labeltop=False,
                   labelleft=False,
                   labelright=False)

    if box:
        ax.spines[:].set_color(color)
        ax.spines[:].set_linewidth(4)
    else:
        ax.spines[:].set_color('none')

    return fig, ax


def coord_cross(ax,
                x='x',
                y='y',
                xoff=.05,
                yoff=.05,
                flipx=False,
                flipy=False):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    delx = xlim[1] - xlim[0]
    dely = xlim[1] - xlim[0]
    ax.annotate(
        "",
        (xlim[0] + xoff * delx, (ylim[1] + ylim[0]) / 2.),
        (xlim[1] - xoff * delx, (ylim[1] + ylim[0]) / 2.),
        arrowprops=dict(arrowstyle="-"),
    )
    if not flipx:
        ax.text(xlim[1] - xoff * delx / 2., (ylim[1] + ylim[0]) / 2.,
                x,
                horizontalalignment="center",
                verticalalignment="center")
    else:
        ax.text(xlim[0] + xoff * delx / 2., (ylim[1] + ylim[0]) / 2.,
                x,
                horizontalalignment="center",
                verticalalignment="center")

    ax.annotate(
        "",
        ((xlim[1] + xlim[0]) / 2., ylim[0] + yoff * dely),
        ((xlim[1] + xlim[0]) / 2., ylim[1] - yoff * dely),
        arrowprops=dict(arrowstyle="-"),
    )
    if not flipy:
        ax.text((xlim[1] + xlim[0]) / 2.,
                ylim[1] - yoff * dely / 2.,
                y,
                horizontalalignment="center",
                verticalalignment="center")
    else:
        ax.text((xlim[1] + xlim[0]) / 2.,
                ylim[0] + yoff * dely / 2.,
                y,
                horizontalalignment="center",
                verticalalignment="center")


def label_angle(ax,
                th_name=r'$\theta$',
                th0=0. * u.deg,
                th1=180. * u.deg,
                rad=1.,
                color='black',
                va=None, vh=None,
                th0_arrow=False, th1_arrow=False, other_direction=False):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    center = np.array([(xlim[0] + xlim[1]) / 2., (ylim[0] + ylim[1]) / 2.])
    th1_mod = np.mod(th1.to_value(u.deg), 360.) * u.deg
    th0_mod = np.mod(th0.to_value(u.deg), 360.) * u.deg
    ax.add_patch(
        patches.Arc(xy=center,
                    width=2. * rad,
                    height=2. * rad,
                    theta1=th0.to_value(u.deg),
                    theta2=th1.to_value(u.deg),
                    angle=0.,
                    color=color))
    if th1_mod > th0_mod:
        label_pos = center + rad * np.array(
            [np.cos((th0_mod + th1_mod) / 2.),
             np.sin((th0_mod + th1_mod) / 2.)])
    else:
        label_pos = center + rad * np.array([
            np.cos(180. * u.deg + (th0_mod + th1_mod) / 2.),
            np.sin(180. * u.deg + (th0_mod + th1_mod) / 2.)
        ])
    if va is None:
        if label_pos[1] - center[1] > 0.:
            va = 'bottom'
        else:
            va = 'top'
    if vh is None:
        if label_pos[0] - center[0] > 0.:
            vh = 'left'
        else:
            vh = 'right'
    ax.text(x=label_pos[0],
            y=label_pos[1],
            s=th_name,
            horizontalalignment=vh,
            verticalalignment=va,
            color=color)
    if th1_arrow:
        startX = center[0]+rad*np.cos(th1)
        startY = center[1]+rad*np.sin(th1)
        startDX = -.000001*rad*np.sin(th1)
        startDY = .000001*rad*np.cos(th1)
        if th1_mod < th0_mod:
            startDX *= -1.
            startDY *= -1.
        if other_direction:
            startDX *= -1.
            startDY *= -1.
        ax.arrow(startX-startDX, startY-startDY, startDX, startDY,
                 color=color, width=0., head_width=.045, head_length=.045,
                 length_includes_head=True)
    if th0_arrow:
        startX = center[0]+rad*np.cos(th0)
        startY = center[1]+rad*np.sin(th0)
        startDX = -.000001*rad*np.sin(th0)
        startDY = .000001*rad*np.cos(th0)
        if th1_mod < th0_mod:
            startDX *= -1.
            startDY *= -1.
        if other_direction:
            startDX *= -1.
            startDY *= -1.
        ax.arrow(startX-startDX, startY-startDY, startDX, startDY,
                 color=color, width=0., head_width=.045, head_length=.045,
                 length_includes_head=True)


def rot_lin(ax, th=0.*u.deg, s=0.9,
            name='line', arrow="-", ls='-', color='black', va=None, vh=None):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    delx = xlim[1] - xlim[0]
    dely = xlim[1] - xlim[0]
    center = np.array([(xlim[0] + xlim[1]) / 2., (ylim[0] + ylim[1]) / 2.])
    p0 = center + np.array([np.cos(th) * delx / 2.,
                            np.sin(th) * dely / 2.]) * s
    p1 = center + np.array([
        np.cos(th + 180. * u.deg) * delx / 2.,
        np.sin(th + 180. * u.deg) * dely / 2.
    ]) * s
    ax.annotate(
        "",
        p0,
        p1,
        arrowprops=dict(arrowstyle=arrow, linestyle=ls, color=color),
    )
    if va is None:
        if p0[1] - center[1] > 0.:
            va = 'bottom'
        else:
            va = 'top'
    if vh is None:
        if p0[0] - center[0] < 0.:
            vh = 'left'
        else:
            vh = 'right'
    ax.text(p0[0],
            p0[1],
            name,
            horizontalalignment=vh,
            verticalalignment=va,
            color=color)


def rot_vec(ax, th=0.*u.deg, s=0.9,
            name='line', arrow="-", ls='-', color='black', va=None, vh=None):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    delx = xlim[1] - xlim[0]
    dely = xlim[1] - xlim[0]
    center = np.array([(xlim[0] + xlim[1]) / 2., (ylim[0] + ylim[1]) / 2.])
    p0 = center + np.array([np.cos(th) * delx / 2.,
                            np.sin(th) * dely / 2.]) * s
    p1 = center
    ax.annotate(
        "",
        p0,
        p1,
        arrowprops=dict(arrowstyle=arrow, linestyle=ls, color=color),
    )
    if va is None:
        if p0[1] - center[1] > 0.:
            va = 'bottom'
        else:
            va = 'top'
    if vh is None:
        if p0[0] - center[0] < 0.:
            vh = 'left'
        else:
            vh = 'right'
    ax.text(p0[0],
            p0[1],
            name,
            horizontalalignment=vh,
            verticalalignment=va,
            color=color)


def elpse(ax, e=0., a0=1., omg=0.*u.deg, center_focus=True, color='black'):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    center = np.array([(xlim[0] + xlim[1]) / 2., (ylim[0] + ylim[1]) / 2.])
    if center_focus:
        offset = e*a0*np.array([np.cos(omg), np.sin(omg)])
        center -= offset
    ax.add_patch(patches.Ellipse(center,
                                 width=2.*a0, height=2.*a0*np.sqrt(1.-e**2),
                                 angle=omg.to_value(u.deg),
                                 facecolor='none', edgecolor=color))


def circl(ax, a0=1., omg=0.*u.deg, incl=0.*u.deg, color='black'):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    center = np.array([(xlim[0] + xlim[1]) / 2., (ylim[0] + ylim[1]) / 2.])
    ax.add_patch(patches.Ellipse(center,
                                 width=2.*a0, height=2.*a0*np.cos(incl),
                                 angle=omg.to_value(u.deg),
                                 facecolor='none', edgecolor=color))
