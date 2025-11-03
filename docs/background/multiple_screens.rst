=====================================
Scattering by multiple linear screens
=====================================

When multiple screens are present, the number of paths from source to observer
become much larger. Below, we derive the paths taken for linear screens.  The
result is implemented in the :py:class:`~screens.screen.Screen1D` class, and
this class is used in the tutorial on :doc:`../tutorials/two_screens`.  To
help visualize how the resulting wavefield depends on the properties of the
screen, have a look at `examples/two_screen_interaction.py
<https://github.com/mhvk/screens/blob/main/examples/two_screen_interaction.py>`_.

Multiple screens
================

Consider screens with linear features between the telescope and the
pulsar.  Consider a coordinate system in which z is along the line of
sight (with direction :math:`\hat{z}`), and the lines on a given screen :math:`s`
describing the features are given by,

.. math::
   :label: eq_basic_trajectory

    d_{s}\hat{z} + p \hat{r} + s \hat{u}

where :math:`p \hat{r}` is a cylindrical radius from the line of sight to the line
(ie., :math:`\hat{r}\cdot\hat{z}=0`) and :math:`\hat{u}=\hat{z}\times\hat{r}` a
unit vector perpendicular to it in the plane of the screen.

Imagine now going from the observer to some point along a line on
a first screen.  It is easiest to work in terms of angles relative
to the observer, so we use :math:`\rho=p/d` and :math:`\varsigma=s/d` to write this
trajectory as,

.. math::
   :label: eq_basic_angular_trajectory_to_screen1

    d(\hat{z} + \rho_{1}\hat{r}_{1} + \varsigma_{1}\hat{u}_{1}).

When it hits the line, light can be bent only perpedicular to the
line, by an angle which we will label :math:`\alpha` (with positive :math:`\alpha` implying
bending closer to the line of sight; equal to :math:`\hat\alpha` in Simard &
Pen).  Hence, beyond the screen it will travel along

.. math::
   :label: eq_basic_angular_trajectory_to_screen2

    d(\hat{z} + \rho_{1}\hat{r}_{1} + \varsigma_{1}\hat{u}_{1}) - (d-d_{1})\alpha_{1}\hat{r}_{1}.

If it then hits a line on the second screen, it will again be bent,
and then follow,

.. math::
   :label: eq_basic_angular_trajectory_to_pulsar

    d(\hat{z} + \rho_{1}\hat{r}_{1} + \varsigma_{1}\hat{u}_{1}) - (d-d_{1})\alpha_{1}\hat{r}_{1} - (d-d_{2})\hat{r}_{2}.

In order to specify the full trajectory, we need to make sure that it
actually intersects the line on the second screen, and ends at the
pulsar, i.e.,

.. math::
   :label: eq_two_screen_constraints

    \begin{eqnarray}
    d_{2}(\hat{z} + \rho_{1}\hat{r}_{1} + \varsigma_{1}\hat{u}_{1}) - (d_{2}-d_{1})\alpha_{1}\hat{r}_{1}
     &=& d_{2}(\hat{z} + \rho_{2}\hat{r}_{2} + \varsigma_{2}\hat{u}_{2}),\\
    d_{p}(\hat{z} + \rho_{1}\hat{r}_{1} + \varsigma_{1}\hat{u}_{1}) - (d_{p}-d_{1})\alpha_{1}\hat{r}_{1} - (d_{p}-d_{2})\alpha_{2}\hat{r}_{2}
     &=& d_{p}\hat{z}.
    \end{eqnarray}

This can be simplified to,

.. math::
   :label: eq_two_screen_constraints_simplified

    \begin{eqnarray}
    \rho_{1}\hat{r}_{1} + \varsigma_{1}\hat{u}_{1} - (1-d_{1}/d_{2})\alpha_{1}\hat{r}_{1}
     &=& \rho_{2}\hat{r}_{2} + \varsigma_{2}\hat{u}_{2},\\
    \rho_{1}\hat{r}_{1} + \varsigma_{1}\hat{u}_{1} - (1-d_{1}/d_{p})\alpha_{1}\hat{r}_{1} - (1-d_{2}/d_{p})\hat{r}_{2}) &=& 0.
    \end{eqnarray}

Obviously, this can be extended to multiple screens, and for :math:`n`
screens one generally winds up with :math:`n` equations for two-dimensional
vectors with :math:`2n` unknowns, the :math:`n` bending angles :math:`\alpha_{i}` and the :math:`n`
angular offsets :math:`\varsigma_{i}` along the lines.

Direct solutions for one or two screens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If there is only a single screen, the situation is simple: we just
need to make sure the trajectory after the first screen ends at the
pulsar. Following the above simplification, one needs,

.. math::
   :label: eq_one_screen_constraint

    \rho_{1}\hat{r}_{1} + \varsigma_{1}\hat{u}_{1} - (1-d_{1}/d_{p})\alpha_{1}\hat{r}_{1} = 0.

Thus, taking inner products with :math:`\hat{r}_{1}` and :math:`\hat{u}_{1}`, one finds
:math:`\alpha_{1}=\rho_{1}d_{p}/(d_{p}-d_{1})` and :math:`\varsigma_{1}=0`, i.e., the ray will always be
bent along the point closes to the line of sight.

For two screens, the above still can be solved fairly easily by
substitution, though it becomes less obvious this is still useful.
We start by considering inner products with :math:`\hat{r}_{1}`
and :math:`\hat{u}_{1}`, giving the following four equations (writing
:math:`s_{12}\equiv1-d_{1}/d_{2}`, etc., for brevity),

.. math::

    \begin{eqnarray}
    \rho_{1} - s_{12}\alpha_{1} &=& \rho_{2}\hat{r}_{2}\cdot\hat{r}_{1} + \varsigma_{2}\hat{u}_{2}\cdot\hat{r}_{1}
               = \rho_{2}\cos \delta - \varsigma_{2}\sin \delta,\\
    \varsigma_{1} &=& \rho_{2}\hat{r}_{2}\cdot\hat{u}_{1} + \varsigma_{2}\hat{u}_{2}\cdot\hat{u}_{1}
        = \rho_{2} \sin \delta + \varsigma_{2} \cos \delta,\\
    \rho_{1} - s_{1p}\alpha_{1} &=& s_{2p}\alpha_{2}\hat{r}_{2}\cdot\hat{r}_{1}
               = s_{2p}\alpha_{2}\cos \delta\\
    \varsigma_{1} &=& s_{2p}\alpha_{2}\hat{r}_{2}\cdot\hat{u}_{1}
        = s_{2p}\alpha_{2}\sin \delta,
    \end{eqnarray}


where in the second equalities we used
:math:`\hat{r}_{2}\cdot\hat{r}_{1}=\hat{u}_{2}\cdot\hat{u}_{1}=\cos \delta` and
:math:`\hat{r}_{2}\cdot\hat{u}_{1}=-\hat{u}_{2}\cdot\hat{r}_{1}=\sin \delta`.

Eliminating :math:`s_{2p}\alpha_{2}` between the third and fourth,

.. math::

    \rho_{1} - s_{1p}\alpha_{1} = \varsigma_{1}/\tan \delta.

Solving for :math:`\alpha_{1}` and inserting this in the first,

.. math::

    \rho_{1} - (s_{12}/s_{1p})(\rho_{1}-\varsigma_{1}/\tan \delta) = \rho_{2}\cos \delta - \varsigma_{2}\sin \delta.

Collecting :math:`\rho_{1}` terms and inserting the second,

.. math::

    \rho_{1}(1 - s_{12}/s_{1p}) + (s_{12}/s_{1p})(\rho_{2} \sin \delta + \varsigma_{2} \cos \delta)/\tan \delta
    = \rho_{2}\cos \delta - \varsigma_{2}\sin \delta.

Bringing :math:`\varsigma_{2}` and :math:`\rho_{2}` terms together, this simplifies to

.. math::

    \varsigma_{2}(\sin \delta + (s_{12}/s_{1p}) \cos^{2}\delta/\sin \delta)
    = (\rho_{2}\cos \delta - \rho_{1})(1-s_{12}/s_{1p}),

and thus

.. math::

    \varsigma_{2}(1 - \cos^{2}\delta(1-s_{12}/s_{1p})) = (\rho_{2}\cos \delta -\rho_{1})(1-s_{12}/s_{1p})\sin \delta.

The other unknowns then follow from substitution.

General solution
~~~~~~~~~~~~~~~~

For :math:`n` screens, it has to hold for all screens :math:`i>1`, that,

.. math::
   :label: eq_screen_intersections

    \rho_{1}\hat{r}_{1} + \varsigma_{1}\hat{u}_{1}
    - \sum_{j=1}^{i-1} \alpha_{j}(1-d_{j}/d_{i})\hat{r}_{j}
    = \rho_{i}\hat{r}_{i} + \varsigma_{i}\hat{u}_{i}.

The same has to hold for the pulsar, and one can incorporate this by
giving it an index :math:`p` for which :math:`p-1=n` in the summation.  Above, we
effectively assumed :math:`\rho_{p}=\varsigma_{p}=0`, but in the general case one
will have a spatial offset :math:`\vec{r}_{p}` corresponding to an angular
offset :math:`\rho_{p}=r_{p}/d_{p}` in direction :math:`\hat{r}_{p}` (and :math:`\varsigma_{p}=0`).

One can also have the telescope be offset (e.g., from the solar system
barycentre), by some :math:`\vec{r}_{t}` with norm :math:`r_{t}` and direction
:math:`\hat{r}_{t}`.  Essentially the same equation will hold, with summation
starting at :math:`j=t=0`, :math:`d_{t}=0`, :math:`\rho_{ti}=r_{t}/d_{i}`, and :math:`\alpha_{t}` and :math:`\varsigma_{t}` angles towards
the line of sight and perpendicular to :math:`\hat{r}_{t}`, respectively.
Bringing unknowns to one side and the knowns to the other, and again
writing :math:`s_{ji}=1-d_{j}/d_{i}` (where all :math:`s_{ti}=1`), one finds,

.. math::
    :label: eq_general_constraints

    - (\varsigma_{i}\hat{u}_{i} - \varsigma_{t}\hat{u}_{t})
    - \sum_{j=0}^{i-1} \alpha_{j}s_{ji}\hat{r}_{j}
    = \rho_{i}\hat{r}_{i} - \rho_{t}\hat{r}_{t}

To solve this, we project on given :math:`x` and :math:`y` directions, i.e., use
:math:`\hat{r}_{x,y}` and :math:`\hat{u}_{x,y}` (in terms of angles
:math:`\phi_{i}` relative to :math:`\hat{x}`,
one has  :math:`\hat{r}_{i} = \cos \phi_{i} \hat{x} + \sin \phi_{i}\hat{y}`
and thus :math:`\hat{u}_{i} = -\sin \phi_{i} \hat{x} + \cos \phi_{i}\hat{y}`).
Furthermore, for brevity, we write
:math:`\vec{\theta} \equiv (p_{i}\hat{r}_{i} - p_{t}\hat{r}_{t}) / d_{i}`.
With that, the equations in matrix form are,

.. math::
   :label: eq_matrix_form

   \left(\begin{matrix}
    \hat{u}_{t,x} & -\hat{r}_{t,x} & -\hat{u}_{1,x} & 0 & \ldots & 0 & 0\\
    \hat{u}_{t,y} & -\hat{r}_{t,y} & -\hat{u}_{1,y} & 0 & \ldots & 0 & 0\\
    \hat{u}_{t,x} & -\hat{r}_{t,x} & 0 & -s_{12}\hat{r}_{1,x} & \ldots & 0 & 0\\
    \hat{u}_{t,y} & -\hat{r}_{t,y} & 0 & -s_{12}\hat{r}_{1,y} & \ldots & 0 & 0\\
     \vdots  & \vdots    &\vdots & \vdots & \ddots & \vdots &\vdots \\
    \hat{u}_{t,x} & -\hat{r}_{t,x} & 0 & -s_{1n}\hat{r}_{1,x} & \ldots & -\hat{u}_{n,x} & 0\\
    \hat{u}_{t,y} & -\hat{r}_{t,y} & 0 & -s_{1n}\hat{r}_{1,y} & \ldots & -\hat{u}_{n,y} & 0\\
    \hat{u}_{t,x} & -\hat{r}_{t,x} & 0 & -s_{1p}\hat{r}_{1,x} & \ldots & 0 & -s_{np}\hat{r}_{n,x}\\
    \hat{u}_{t,y} & -\hat{r}_{t,y} & 0 & -s_{1p}\hat{r}_{1,y} & \ldots & 0 & -s_{np}\hat{r}_{n,y}\\
   \end{matrix}\right)
   \left(\begin{matrix}
    \varsigma_{t}\\
    \alpha_{t}\\
    \varsigma_{1}\\
    \alpha_{1}\\
    \vdots\\
    \varsigma_{n-1}\\
    \alpha_{n-1}\\
    \varsigma_{n}\\
    \alpha_{n}
    \end{matrix}\right)
   = \left(\begin{matrix}
    \theta_{1,x}\\
    \theta_{1,y}\\
    \theta_{2,x}\\
    \theta_{2,y}\\
    \vdots\\
    \theta_{n,x}\\
    \theta_{n,y}\\
    \theta_{p,x}\\
    \theta_{p,y}\\
   \end{matrix}\right).

These can be solved by by inverting the matrix :math:`A`, i.e.,

.. math::
   :label: eq_angles_solution

    \left(\begin{matrix}
    \varsigma_{t}\\
    \alpha_{t}\\
    \varsigma_{1}\\
    \alpha_{1}\\
    \vdots\\
    \varsigma_{n-1}\\
    \alpha_{n-1}\\
    \varsigma_{n}\\
    \alpha_{n}
    \end{matrix}\right) = A^{-1}
    \left(\begin{matrix}
    \theta_{1,x}\\
    \theta_{1,y}\\
    \theta_{2,x}\\
    \theta_{2,y}\\
    \vdots\\
    \theta_{n,x}\\
    \theta_{n,y}\\
    \theta_{p,x}\\
    \theta_{p,y}\\
    \end{matrix}\right).

Velocities
~~~~~~~~~~

In principle, the telescope, screens and pulsar will all have velocities.  One
sees that the entries in the matrix involve only directions of the telescope and
screens and ratios of distances, which will change with time much more slowly
than everything else.  Hence, one can solve for the time derivatives of the
parameters by applying the matrix inverse to the time derivatives of the entries
of the right-hand side vector, which are simply the :math:`x` and :math:`y`
components of the proper motions of the screens and the pulsar relative to the
telescope, i.e.,

.. math::
   :label: eq_derivatives_solution

   \left(\begin{matrix}
    \dot{\varsigma}_{t}\\
    \dot{\alpha}_{t}\\
    \dot{\varsigma}_{1}\\
    \dot{\alpha}_{1}\\
    \vdots\\
    \dot{\varsigma}_{n-1}\\
    \dot{\alpha}_{n-1}\\
    \dot{\varsigma}_{n}\\
    \dot{\alpha}_{n}
   \end{matrix}\right)
   = A^{-1}
   \left(\begin{matrix}
    \mu_{1,x}\\
    \mu_{1,y}\\
    \mu_{2,x}\\
    \mu_{2,y}\\
    \vdots\\
    \mu_{n,x}\\
    \mu_{n,y}\\
    \mu_{p,x}\\
    \mu_{p,y}\\
   \end{matrix}\right).

Trajectories and time delays
----------------------------

The above :math:`\theta` and :math:`\mu` reflect the positions and proper
motions of the structures, not of trajectories taken by different rays.
Those are given by,

.. math::
   :label: eq_trajectories_solution

    \begin{eqnarray}
    \vec{\theta}^{r}_{i} &=& \theta^{s}_{i}\hat{r}_{i} + \varsigma_{i} \hat{u}_{i},\\
    \vec{\mu}^{r}_{i} &=& \mu^{s}_{i}\hat{r}_{i} + \dot{\varsigma}_{i} \hat{u}_{i}.
    \end{eqnarray}

The total time delay :math:`\tau` and its derivative :math:`\dot{\tau}` are given
by,

.. math::
   :label: eq_time_delays

    \begin{eqnarray}
    \tau &=& \sum_{i=0}^{n} \frac{d_{i+1}-d_{i}}{2c}
    \left|\vec{\theta}^{r}_{i+1}-\vec{\theta}^{r}_{i}\right|^{2},\\
    \dot{\tau} &=&
    \sum_{i=0}^{n} \frac{d_{i+1}-d_{i}}{c}
    \left(\vec{\theta}^{r}_{i+1}-\vec{\theta}^{r}_{i}\right)
    \cdot\left(\vec{\mu}^{r}_{i+1}-\vec{\mu}^{r}_{i}\right).
    \end{eqnarray}

Bending angle dependent positions
=================================

In the derivation above, the rays have to intersect the one dimensional lines on
each screen.  Physically, this corresponds to assuming the lensing structures
have widths much smaller than their separations.  In reality, where a ray
crosses through a lens will depend on the bending angle: very small angles are
produced right on top of the lens and far away from it, while the largest
bending angles occur where the electron column density gradient is steepest.

While the general bending-angle dependence of the crossing location depends on
the precise lens shape, and thus cannot be easily included, it turns out to be
nearly trivial to include a linear expansion, where the angular location of the
crossing point is given by,

.. math::
   :label: eq_position_with_gradient

   p(\alpha) = p(0) + \alpha \frac{dp}{d\alpha}
   \quad\Rightarrow\quad
   \rho(\alpha) = \rho(0) + \frac{\alpha}{d}\frac{dp}{d\alpha}
   = \rho(0) + \alpha \rho^{\prime},

where in the last equality we implicitly defined :math:`\rho^{\prime}\equiv
(1/d)(dp/d\alpha)`.
The case described so far then corresponds to the zeroth-order approximation,
with :math:`dp/d\alpha=0`.

Looking at the general solution (Eq. :eq:`eq_general_constraints`), one sees
that the only difference is that the :math:`\rho_{i}\hat{r}_{i}` term on the
right now no longer is constant.
Taking the :math:`\alpha`-dependent part to the left, one finds,

.. math::
   :label: eq_general_constraints_with_gradients

    - (\varsigma_{i}\hat{u}_{i} - \varsigma_{t}\hat{u}_{t})
    - \alpha_{i} \rho_{i}^{\prime} \hat{r}_{i}
    - \sum_{j=0}^{i-1} \alpha_{j}s_{ji}\hat{r}_{j}
    = \rho_{i}(0)\hat{r}_{i} - \rho_{t}\hat{r}_{t},

In matrix form, one then has,

.. math::
   :label: eq_matrix_form_with_gradients

   \left(\begin{matrix}
    \hat{u}_{t,x} & -\hat{r}_{t,x} & -\hat{u}_{1,x} & -\rho_{1}^{\prime}\hat{r}_{1,x} & \ldots & 0 & 0\\
    \hat{u}_{t,y} & -\hat{r}_{t,y} & -\hat{u}_{1,y} & -\rho_{1}^{\prime}\hat{r}_{1,y} & \ldots & 0 & 0\\
    \hat{u}_{t,x} & -\hat{r}_{t,x} & 0 & -s_{12}\hat{r}_{1,x} & \ldots & 0 & 0\\
    \hat{u}_{t,y} & -\hat{r}_{t,y} & 0 & -s_{12}\hat{r}_{1,y} & \ldots & 0 & 0\\
     \vdots  & \vdots    &\vdots & \vdots & \ddots & \vdots &\vdots \\
    \hat{u}_{t,x} & -\hat{r}_{t,x} & 0 & -s_{1n}\hat{r}_{1,x} & \ldots & -\hat{u}_{n,x} & -\rho_{n}^{\prime} \hat{r}_{n,x}\\
    \hat{u}_{t,y} & -\hat{r}_{t,y} & 0 & -s_{1n}\hat{r}_{1,y} & \ldots & -\hat{u}_{n,y} & -\rho_{n}^{\prime} \hat{r}_{n,y}\\
    \hat{u}_{t,x} & -\hat{r}_{t,x} & 0 & -s_{1p}\hat{r}_{1,x} & \ldots & 0 & -s_{np}\hat{r}_{n,x}\\
    \hat{u}_{t,y} & -\hat{r}_{t,y} & 0 & -s_{1p}\hat{r}_{1,y} & \ldots & 0 & -s_{np}\hat{r}_{n,y}\\
   \end{matrix}\right)
   \left(\begin{matrix}
    \varsigma_{t}\\
    \alpha_{t}\\
    \varsigma_{1}\\
    \alpha_{1}\\
    \vdots\\
    \varsigma_{n-1}\\
    \alpha_{n-1}\\
    \varsigma_{n}\\
    \alpha_{n}
    \end{matrix}\right)
   = \left(\begin{matrix}
    \theta_{1,x}\\
    \theta_{1,y}\\
    \theta_{2,x}\\
    \theta_{2,y}\\
    \vdots\\
    \theta_{n,x}\\
    \theta_{n,y}\\
    \theta_{p,x}\\
    \theta_{p,y}\\
   \end{matrix}\right).

Like before, one can use the same matrix inverse to calculate velocities.
The trajectories relative to the telescope now are given by,

.. math::
   :label: eq_trajectories_solution_with_gradients

    \begin{eqnarray}
    \theta^{r}_{i}
    = \left(\theta^{s}_{i} + \alpha_{i}\rho_{i}^{\prime}\right)\hat{r}_{i}
    + \varsigma_{i} \hat{u}_{i},\\
    \mu^{r}_{i}
    = \left(\mu^{s}_{i} + \dot{\alpha}_{i}\rho_{i}^{\prime}\right)\hat{r}_{i}
    + \dot{\varsigma}_{i} \hat{u}_{i}.
    \end{eqnarray}
