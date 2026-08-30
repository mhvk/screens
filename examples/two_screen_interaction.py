# Licensed under the GPLv3 - see LICENSE
"""Explore the interaction between two screens.

Setup is psr -> screen 2 -> screen 1 -> observer.

Displayed is the (doppler-delay) wavefield space.

Black cross direct line of sight
Blue: only through screen 1
Red:  only through screen 2 (only few points)
Grey: through both screens

Adjustable are the 
- pulsar velocity (less useful) and the 
- angles, distances, velocities, and physical scale (called y here) of the 
screens (direction in which the line of scattering points is oriented,
relative to the pulsar).  All velocies
are relative to the observer.

The two-screen interactions become just vertical lines (no tau-dot) when
one sets xi2 = 0 (i.e., screen closest to pulsar parallel to it, so no
motion along the waves), or xi1 = xi2.  Note that when one sets xi1=90,
the blue points no longer see any velocity, but the interacting points do.

Note: Pulsar/Earth velocity/orientation are adjustable

"""

import astropy.units as u
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import CylindricalRepresentation
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button

from screens.screen import Source, Screen1D, Telescope


#velocity, orientation and distance to the pulsar
d_p = 300 * u.pc
vpsr_init = 300. * u.km/u.s
dp_angle = 90. * u.deg

#velocity and orientation of the earth
vearth = 0. * u.km/u.s
e_angle = 0. * u.deg

#screen 1 params (closer to the Earth)
d_s1_init = 100 * u.pc 
xi1_init = 30*u.deg 
v1_init = 0.*u.km/u.s 

#screen 2 params (closer to the pulsar)
d_s2_init = 200.*u.pc
xi2_init = 45 *u.deg
v2_init = 0*u.km/u.s



tau_unit = u.us
taudot_unit = u.us/u.day

#scale factors that control the physical scale of the 1d screens
#they are just multipliers applied to the scales of 1AU 1d screens
#for instance scr2_init = 0.1 implies the screen spans -0.1 AU to 0.1 AU
scr1_init = 1.0
scr2_init = 1.0
taudot_arr = np.linspace(-2,2) * u.us * u.day**-1



#useful relations for the two 1d screens curvature measurement
deff1 = d_p * d_s1_init / (d_p - d_s1_init)
veff1 = (d_s1_init / (d_p - d_s1_init) * vpsr_init * ( +np.sin(xi1_init) * np.sin(dp_angle) + np.cos(xi1_init) * np.cos(dp_angle)) 
         - v1_init * d_p / (d_p - d_s1_init)
         + vearth * ( +np.sin(xi1_init) * np.sin(e_angle) + np.cos(xi1_init) * np.cos(e_angle))
        )
eta1 = ( deff1 * const.c / 2 / (veff1)**2 ).to(u.us * u.day**2 / u.us**2)


deff2 = d_p * d_s2_init / (d_p - d_s2_init)
veff2 = (vpsr_init * d_s2_init / (d_p - d_s2_init) * ( +np.sin(xi2_init) * np.sin(dp_angle) + np.cos(xi2_init) * np.cos(dp_angle)) 
         - v2_init * d_p / (d_p - d_s2_init)
         + vearth * ( +np.sin(xi2_init) * np.sin(e_angle) + np.cos(xi2_init) * np.cos(e_angle))
        )
eta2 = ( deff2 * const.c / 2 / (veff2)**2 ).to(u.us * u.day**2 / u.us**2)



#defining some useful quantities for interaction arc
Vex = -vearth * np.cos(e_angle) 
Vey = -vearth * np.sin(e_angle)
vpx = -vpsr_init * np.cos(dp_angle) 
vpy = -vpsr_init * np.sin(dp_angle)
e1 = xi1_init
e2 = xi2_init
ds2 = d_s2_init
ds1 = d_s1_init
vs1 = v1_init
vs2 = v2_init
s12 = 1 - d_s1_init / d_s2_init
s1p = 1 - d_s1_init / d_p
d = e2 - e1





## interaction arc plot for th2 -> 0
th1 = ( scr1_init * np.linspace(-1.0, 1.0, 61) << u.AU  * u.rad / ds1).to(u.rad).value
th2 = 0.

rho2 = ds2 * np.sin(d) * (s12 - s1p) / (s1p + (s12 - s1p)*np.cos(d)**2 ) * th1
rho1 = ds1 * np.sin(d) * np.cos(d) * (s12 - s1p) / (s1p + (s12 - s1p) * np.cos(d)**2 ) * th1 

tau_noth2 = ( 1/ const.c * ( ds1 * th1**2 / 2 + (ds1 * th1)**2 / 2 / (ds2 - ds1) + rho1**2 / 2 / ds1 + rho2**2 / 2/ (d_p-ds2)
                      + (rho1**2 + rho2**2 - 2 * rho1 * rho2 * np.cos(d) ) / 2 / (ds2 - ds1)
                      + (rho2 * ds1 * th1 * np.sin(d) ) / (ds2 - ds1) ) ).to(u.us)

fd_noth2 = ( ( Vex * rho1 * np.sin(e1) / ds1  
            - Vex * th1 * np.cos(e1) 
            - Vey * rho1 * np.cos(e1) / ds1
            - Vey * th1 * np.sin(e1) 
            + vpx * rho2 * np.sin(e2) / (d_p - ds2)
            - vpy * rho2 * np.cos(e2) / (d_p - ds2)
            - th1 * vs1 / (ds2 - ds1) * ds1
            + th1 * vs2 * np.cos(e1 - e2) / (ds2 - ds1) * ds1
            - th1 * vs1 
            + th2 * vs1 * np.cos(e1 - e2) / (ds2 - ds1) * ds2
            - th2 * vs2 / (ds2 - ds1) * ds2
            - th2 * vs2 / (d_p - ds2) * ds2
            + rho1 * vs2 * np.sin(d) / (ds2 - ds1)
            - rho2 * vs1 * np.sin(d) / (ds2 - ds1)
           ) / const.c ).to(u.us / u.day)



def observations(xi1=xi1_init, v1=v1_init,
                 xi2=xi2_init, v2=v2_init,
                 vpsr=vpsr_init,
                 scr1_scale = scr1_init, scr2_scale = scr2_init,
                 d_s1 = d_s1_init, d_s2 = d_s2_init):
    
    p1 = scr1_scale * np.linspace(-1.0, 1.0, 51) << u.AU
    t1 = 0.1
    m1 = np.exp(-0.5*(p1/(0.5*u.AU))**2)
    m1 *= t1
    
    
    p2 = scr2_scale * np.linspace(-1., 1., 51) << u.AU
    t2 = 0.1 
    m2 = np.exp(-0.5*(p2/(0.5*u.AU))**2)
    m2 *= t2    
    
    t12 = t1 * t2
    
    sum1 = np.sum( m1**2 )
    sum2 = np.sum( m2**2 )
    
    
    vel_psr = CylindricalRepresentation(vpsr, dp_angle, 0.*u.km/u.s).to_cartesian()
    vel_ear = CylindricalRepresentation(vearth, e_angle, 0.*u.km/u.s).to_cartesian()
    telescope = Telescope(vel = vel_ear)
    
    
    pulsar0 = Source(vel=vel_psr, magnification = 1.)
    pulsar1 = Source(vel=vel_psr, magnification=t2)
    pulsar2 = Source(vel=vel_psr, magnification=t1)
    pulsar12 = Source(vel=vel_psr, magnification = t12)
    
    normal1 = CylindricalRepresentation(1., xi1, 0.).to_cartesian()
    screen1 = Screen1D(normal=normal1, p=p1, v=v1, magnification=m1)
    normal2 = CylindricalRepresentation(1., xi2, 0.).to_cartesian()
    screen2 = Screen1D(normal=normal2, p=p2, v=v2, magnification=m2)

    obs0 = telescope.observe(source=pulsar0, distance=d_p)

    obs1 = telescope.observe(
        source=screen1.observe(source=pulsar1, distance=d_p-d_s1),
        distance=d_s1)

    obs2 = telescope.observe(
        source=screen2.observe(source=pulsar2, distance=d_p-d_s2),
        distance=d_s2)

    obs12 = telescope.observe(
        source=screen1.observe(
            source=screen2.observe(source=pulsar12, distance=d_p-d_s2),
            distance=d_s2-d_s1),
        distance=d_s1)

    return obs0, obs1, obs2, obs12

all_obs = obs0, obs1, obs2, obs12 = observations()
brightness = np.hstack([obs.brightness.ravel() for obs in all_obs])
mall = np.abs(brightness)
# assert np.isclose(np.sum(np.abs(brightness)**2), 1.)

fig, ax = plt.subplots(figsize=(10., 10.))

scs = []
for obs, marker, size, cmap in (
        (obs0, "x", 30, "Greys"),
        (obs1, "o", 20, "Blues"),
        (obs2, "o", 20, "Reds"),
        (obs12, "o", 10, "Greys"))[::-1]:
    tau = obs.tau.to_value(tau_unit).ravel()
    taudot = obs.taudot.to_value(taudot_unit).ravel()
    sc = ax.scatter(taudot, tau, marker=marker, s=size,
                    c=np.abs(obs.brightness), cmap=cmap,
                    norm=LogNorm(vmin=mall.min()*0.9, vmax=mall.max()))
    
    
    scs.insert(0, sc)

fig.colorbar(mappable=sc, label="magnification", fraction=0.1)

ax.set_xlabel(rf"differential delay $\dot{{\tau}}$ ({taudot_unit:latex_inline})")
ax.set_ylabel(rf"relative geometric delay $\tau$ ({tau_unit:latex_inline})")

line1, = ax.plot(taudot_arr , taudot_arr**2 * eta1, c = 'b', alpha = 0.3 )
line2, = ax.plot(taudot_arr , taudot_arr**2 * eta2, c = 'r', alpha = 0.3  )
line3, = ax.plot(fd_noth2 , tau_noth2, c = 'k', ls = '--', alpha = 0.3 )
ax.set_ylim(-0.5,40)
ax.set_xlim(-2.5, 2.5)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom = 0.3)
# Make a horizontal slider to control the frequency.


ax_vpsr = fig.add_axes([0.1, 0.16 + 0.01, 0.75, 0.03])
vpsr_slider = Slider(ax=ax_vpsr, label=r'$v_{psr}~[km/s]$',
                     valmin=0., valmax=500, valinit=vpsr_init.to_value(u.km/u.s))

ax_xi1 = fig.add_axes([0.1, 0.12 + 0.01, 0.3, 0.03])
xi1_slider = Slider(ax=ax_xi1, label=r'$\xi_{1}~[deg]$',
                    valmin=0., valmax=180, valinit=xi1_init.to_value(u.deg))

ax_xi2 = fig.add_axes([0.55, 0.12 + 0.01, 0.3, 0.03])
xi2_slider = Slider(ax=ax_xi2, label=r'$\xi_{2}~[deg]$',
                    valmin=0., valmax=180, valinit=xi2_init.to_value(u.deg))


ax_v1 = fig.add_axes([0.1, 0.08 + 0.01, 0.3, 0.03])
v1_slider = Slider(ax=ax_v1, label=r'$v_{1}~[km/s]$',
                   valmin=-60., valmax=60, valinit=v1_init.value)

ax_v2 = fig.add_axes([0.55, 0.08 + 0.01, 0.3, 0.03])
v2_slider = Slider(ax=ax_v2, label=r'$v_{2}~[km/s]$',
                   valmin=-60., valmax=60, valinit=v2_init.value)

ax_scr1 = fig.add_axes([0.1, 0.04 + 0.01, 0.3, 0.03])
scr1_slider = Slider(ax=ax_scr1, label=r'$y_1~[AU]$',
                     valmin=0., valmax=2, valinit=scr1_init)

ax_scr2 = fig.add_axes([0.55, 0.04 + 0.01, 0.3, 0.03])
scr2_slider = Slider(ax=ax_scr2, label=r'$y_2~[AU]$',
                     valmin=0., valmax=2, valinit=scr2_init)

ax_ds1 = fig.add_axes([0.1, 0.0 + 0.01, 0.3, 0.03])
ds1_slider = Slider(ax=ax_ds1, label=r'$d_{s,1}~[pc]$',
                     valmin=0., valmax=d_p.value, valinit=d_s1_init.to_value(u.pc))

ax_ds2 = fig.add_axes([0.55, 0.0 + 0.01, 0.3, 0.03])
ds2_slider = Slider(ax=ax_ds2, label=r'$d_{s,2}~[pc]$',
                     valmin=0., valmax=d_p.value, valinit=d_s2_init.to_value(u.pc))



def update(val):
    all_obs = observations(
        xi1=xi1_slider.val * u.deg,
        v1=v1_slider.val * u.km/u.s,
        xi2 = xi2_slider.val * u.deg,
        v2 = v2_slider.val * u.km/u.s,
        vpsr = vpsr_slider.val * u.km/u.s,
        scr1_scale = scr1_slider.val,
        scr2_scale = scr2_slider.val,
        d_s1 = ds1_slider.val * u.pc,
        d_s2 = ds2_slider.val * u.pc
        )
    
    
    for obs, sc in zip(all_obs, scs):
        tau = obs.tau.to_value(tau_unit).ravel()
        taudot = obs.taudot.to_value(taudot_unit).ravel()
        sc.set_offsets(np.vstack([taudot, tau]).T)
        
        

        
    #new curvature measuremenets
    deff1 = d_p * ds1_slider.val * u.pc / (d_p - ds1_slider.val * u.pc)
    veff1_new = (-vpsr_slider.val * u.km/u.s * ds1_slider.val * u.pc / (d_p - ds1_slider.val * u.pc) * ( -np.sin(xi1_slider.val * u.deg) * np.sin(dp_angle) - np.cos(xi1_slider.val * u.deg) * np.cos(dp_angle)) 
                 - v1_slider.val * u.km/u.s * d_p / (d_p - ds1_slider.val * u.pc) 
                - vearth * ( -np.sin(xi1_slider.val * u.deg) * np.sin(e_angle) - np.cos(xi1_slider.val * u.deg) * np.cos(e_angle))
                )
    eta1_new = ( deff1 * const.c / 2 / (veff1_new)**2 ).to(u.us * u.day**2 / u.us**2)

    deff2 = d_p * ds2_slider.val * u.pc / (d_p - ds2_slider.val * u.pc)
    veff2_new = (-vpsr_slider.val * u.km/u.s * ds2_slider.val * u.pc / (d_p - ds2_slider.val * u.pc) * ( -np.sin(xi2_slider.val * u.deg) * np.sin(dp_angle) - np.cos(xi2_slider.val * u.deg) * np.cos(dp_angle)) 
                 - v2_slider.val * u.km/u.s * d_p / (d_p - ds2_slider.val * u.pc) 
                - vearth * ( -np.sin(xi2_slider.val * u.deg) * np.sin(e_angle) - np.cos(xi2_slider.val * u.deg) * np.cos(e_angle))
                )
    eta2_new = ( deff2 * const.c / 2 / (veff2_new)**2 ).to(u.us * u.day**2 / u.us**2)
        
    
    line1.set_ydata(eta1_new * taudot_arr**2)
    line2.set_ydata(eta2_new * taudot_arr**2)
    
    
    ds1 = ds1_slider.val * u.pc
    ds2 = ds2_slider.val * u.pc
    Vex = -vearth * np.cos(e_angle) 
    Vey = -vearth * np.sin(e_angle)
    vpx = -vpsr_slider.val * u.km/u.s * np.cos(dp_angle) 
    vpy = -vpsr_slider.val * u.km/u.s * np.sin(dp_angle) 
    e1 = xi1_slider.val * u.deg
    e2 = xi2_slider.val * u.deg
    vs1 = v1_slider.val * u.km/u.s
    vs2 = v2_slider.val * u.km/u.s
    s12 = 1 - ds1 / ds2
    s1p = 1 - ds1 / d_p
    d = e2 - e1
    
    
    #interaction arc curvatures ( both for when th1->0 or th2->0)
     
    if scr1_slider.val >= scr2_slider.val:
        th1 = ( scr1_slider.val * np.linspace(-1.0, 1.0, 101) * u.AU / ds1 * u.rad).to(u.rad).value
        th2 = 0.
        rho2 = ds2 * np.sin(d) * (s12 - s1p) / (s1p + (s12 - s1p)*np.cos(d)**2 ) * th1
        rho1 = ds1 * np.sin(d) * np.cos(d) * (s12 - s1p) / (s1p + (s12 - s1p) * np.cos(d)**2 ) * th1 

        tau_noth3 = ( 1/ const.c * ( ds1 * th1**2 / 2 + (ds1 * th1)**2 / 2 / (ds2 - ds1) + rho1**2 / 2 / ds1 + rho2**2 / 2/ (d_p-ds2)
                          + (rho1**2 + rho2**2 - 2 * rho1 * rho2 * np.cos(d) ) / 2 / (ds2 - ds1)
                          + (rho2 * ds1 * th1 * np.sin(d) ) / (ds2 - ds1) ) ).to(u.us)
        
    elif scr1_slider.val < scr2_slider.val:
        th1 = 0.
        th2 = ( scr2_slider.val * np.linspace(-1.0, 1.0, 50) * u.AU / ds2 * u.rad).to(u.rad).value

        rho2 = ds2 * np.sin(d) * (s12 - s1p) / (s1p + (s12 - s1p)*np.cos(d)**2 ) * (-np.cos(d) * th2)
        rho1 = ds1 * np.sin(d) * s1p / (s1p + (s12 - s1p) * np.cos(d)**2 ) * th2 

        tau_noth3 = ( 1/ const.c * ( (ds2 * th2)**2 / 2 / (d_p - ds2) + rho1**2 / 2 / ds1 + rho2**2 / 2/ (d_p-ds2)
                                    + (ds2 * th2)**2 / 2 / (ds2 - ds1)
                          + (rho1**2 + rho2**2 - 2 * rho1 * rho2 * np.cos(d) ) / 2 / (ds2 - ds1)
                          - (rho1 * ds2 * th2 * np.sin(d) ) / (ds2 - ds1) ) ).to(u.us)

    
    
    fd_noth3 = ( ( Vex * rho1 * np.sin(e1) / ds1  
            - Vex * th1 * np.cos(e1) 
            - Vey * rho1 * np.cos(e1) / ds1
            - Vey * th1 * np.sin(e1) 
            + vpx * rho2 * np.sin(e2) / (d_p - ds2)
            - vpx * (ds2 * th2) * np.cos(e2) / (d_p - ds2)
            - vpy * rho2 * np.cos(e2) / (d_p - ds2)
            - vpy * (ds2 * th2) * np.sin(e2) / (d_p - ds2)
            - th1 * vs1 / (ds2 - ds1) * ds1
            + th1 * vs2 * np.cos(e1 - e2) / (ds2 - ds1) * ds1
            - th1 * vs1 
            + th2 * vs1 * np.cos(e1 - e2) / (ds2 - ds1) * ds2
            - th2 * vs2 / (ds2 - ds1) * ds2
            - th2 * vs2 / (d_p - ds2) * ds2
            + rho1 * vs2 * np.sin(d) / (ds2 - ds1)
            - rho2 * vs1 * np.sin(d) / (ds2 - ds1)
           ) / const.c ).to(u.us / u.day)
    
    line3.set_data(fd_noth3, tau_noth3)
        


xi1_slider.on_changed(update)
v1_slider.on_changed(update)
xi2_slider.on_changed(update)
v2_slider.on_changed(update)
vpsr_slider.on_changed(update)
scr1_slider.on_changed(update)
scr2_slider.on_changed(update)
ds1_slider.on_changed(update)
ds2_slider.on_changed(update)


resetax = fig.add_axes([0.85, 0.225, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    xi1_slider.reset()
    v1_slider.reset()
    xi2_slider.reset()
    v2_slider.reset()
    vpsr_slider.reset()
    scr1_slider.reset()
    scr2_slider.reset()
    ds1_slider.reset()
    ds2_slider.reset()

button.on_clicked(reset)

