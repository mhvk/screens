from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm as cm
import numpy as np
from astropy import units as u
from screens.fields import dynamic_field
from screens.secspec import SecondarySpectrum
from screens.visualization import make_sketch, phasecmap


def generate_magnification(th):

    # Complex magnifications amplitudes of the scattering points
    sig = 1.5*u.mas
    a = 0.5*np.exp(-0.5*(th/sig)**2).to_value(1)
    np.random.seed(2)
    magnification = a * (np.random.normal(size=(th.size, 2)).view('c16')
                         .squeeze(-1))
    magnification[np.where(th == 0)] = 1
    magnification /= np.sqrt((np.abs(magnification)**2).sum())

    return magnification


def simple_figure(th_par, magnification,
                  d_eff=0.5*u.kpc, mu_eff=50.*u.mas/u.yr, fobs=316.*u.MHz,
                  delta_f=2.*u.MHz, delta_t=90.*u.minute, nf=200, nt=180,
                  **kwargs):

    # Observation frequencies and times
    f = (fobs + np.linspace(-0.5*delta_f, 0.5*delta_f, nf, endpoint=False)
         + 0.5*delta_f/nf)
    t = (np.linspace(0.*u.minute, delta_t, nt, endpoint=False)[:, np.newaxis]
         + 0.5*delta_t/nt)

    # Create electric field and dynamic spectrum
    th_perp = np.zeros_like(th_par)
    dynwave = dynamic_field(th_par, th_perp, magnification,
                            d_eff, mu_eff, f, t)
    axes = tuple(range(0, dynwave.ndim-2))
    dynwave = dynwave[...].sum(axes)
    dynspec = np.abs(dynwave)**2

    # Create conjugate and then secondary spectrum
    ss = SecondarySpectrum.from_dynamic_spectrum(dynspec, f=f, t=t, noise=0.,
                                                 d_eff=d_eff, mu_eff=mu_eff,
                                                 theta=th_par,
                                                 magnification=magnification)
    tau = ss.tau
    fd = ss.fd
    secspec = np.abs(ss.secspec)**2

    # Create dynamic wavefield
    dwft = np.fft.fft2(dynwave)
    dwft /= dwft[0, 0]
    dwft = np.fft.fftshift(dwft)

    # Figure
    fig = plt.figure(figsize=(12., 8.))
    plt.subplots_adjust(wspace=0., hspace=0.4)

    ds_extent = (t[0][0].value - 0.5*(t[1][0].value - t[0][0].value),
                 t[-1][0].value + 0.5*(t[1][0].value - t[0][0].value),
                 f[0].value - 0.5*(f[1].value - f[0].value),
                 f[-1].value + 0.5*(f[1].value - f[0].value))
    ss_extent = (fd[0][0].value - 0.5*(fd[1][0].value - fd[0][0].value),
                 fd[-1][0].value + 0.5*(fd[1][0].value - fd[0][0].value),
                 tau[0].value - 0.5*(tau[1].value - tau[0].value),
                 tau[-1].value + 0.5*(tau[1].value - tau[0].value))

    f_lims = (fobs.value - 0.5*delta_f.value, fobs.value + 0.5*delta_f.value)
    t_lims = (0., delta_t.value)

    tau_lims = (-15., 15.)
    fd_lims = (-5., 5.)

    # Plot dynamic spectrum
    ax1 = plt.subplot(231)
    im1 = ax1.imshow(dynspec.T,
                     origin='lower', aspect='auto', interpolation='none',
                     cmap='Greys', extent=ds_extent,
                     vmin=(0 if np.max(dynspec) > 2 else None))
    plt.title('dynamic spectrum')
    plt.xlabel(rf"time $t$ ({t.unit.to_string('latex')})")
    plt.ylabel(rf"frequency $f$ ({f.unit.to_string('latex')})")
    plt.xlim(t_lims)
    plt.ylim(f_lims)
    cbar = plt.colorbar(im1)
    cbar.set_label('normalized intensity')

    # Plot secondary spectrum
    ax2 = plt.subplot(234)
    im2 = ax2.imshow(secspec.T,
                     origin='lower', aspect='auto', interpolation='none',
                     cmap='Greys', extent=ss_extent,
                     norm=mcolors.LogNorm(vmin=1.e-4, vmax=1.))
    plt.title('secondary spectrum')
    plt.xlabel(r"differential Doppler shift $f_\mathrm{{D}}$"
               rf"({fd.unit.to_string('latex')})")
    plt.ylabel(r"relative geometric delay $\tau$ "
               rf"({tau.unit.to_string('latex')})")
    plt.xlim(fd_lims)
    plt.ylim(tau_lims)
    cbar = plt.colorbar(im2)
    cbar.set_label('normalized power')

    # Plot electric field
    bgcol = 'w'
    ax3 = plt.subplot(233)
    ax3.set_facecolor(bgcol)
    ax3.imshow(np.angle(dynwave).T,
               alpha=np.abs(dynwave).T / np.max(np.abs(dynwave).T),
               origin='lower', aspect='auto', interpolation='none',
               cmap=phasecmap, extent=ds_extent, vmin=-np.pi, vmax=np.pi)
    plt.title('dynamic wavefield')
    plt.xlabel(rf"time $t$ ({t.unit.to_string('latex')})")
    plt.ylabel(rf"frequency $f$ ({f.unit.to_string('latex')})")
    plt.xlim(t_lims)
    plt.ylim(f_lims)

    # create temporary colorbar, for axes positioning
    cbar = plt.colorbar(cm.ScalarMappable(), ax=ax3, aspect=5.5)
    cbar_pos = fig.axes[-1].get_position()
    cbar.remove()

    # create 2D colormap
    cax = fig.add_axes(cbar_pos)
    phases = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    alphas = np.linspace(0., 1., 256)
    phasegrid, alphagrid = np.meshgrid(phases, alphas)
    cax.set_facecolor(bgcol)
    cax.imshow(phasegrid, alpha=alphagrid,
               origin='lower', aspect='auto', interpolation='none',
               cmap=phasecmap,
               extent=[-np.pi, np.pi, 0., np.max(np.abs(dynwave))])
    cax.set_xticks([-np.pi, 0., np.pi])
    cax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position('right')
    cax.set_xlabel('phase (rad)')
    cax.set_ylabel('normalized intensity')

    # Plot wavefield
    ax4 = plt.subplot(236)
    im4 = ax4.imshow(np.abs(dwft).T,
                     origin='lower', aspect='auto', interpolation='none',
                     cmap='Greys', extent=ss_extent,
                     norm=mcolors.LogNorm(vmin=1.e-2, vmax=1.))
    plt.title('conjugate wavefield')
    plt.xlabel(r"differential Doppler shift $f_\mathrm{{D}}$"
               rf"({fd.unit.to_string('latex')})")
    plt.ylabel(r"relative geometric delay $\tau$ "
               rf"({tau.unit.to_string('latex')})")
    plt.xlim(fd_lims)
    plt.ylim(tau_lims)
    cbar = plt.colorbar(im4)
    cbar.set_label('normalized power')

    # Add sketch
    ax5 = plt.subplot(132)
    make_sketch(th_par, mu_eff=mu_eff, rotation=-90.*u.deg, ax=ax5, **kwargs)
    ax5.set_position([0.34, 0.05, 0.3, 0.9])

    plt.show()


if __name__ == '__main__':

    theta = np.linspace(-4.5, 4.5, 23) << u.mas
    magnification = generate_magnification(theta)

    simple_figure(theta, magnification)
