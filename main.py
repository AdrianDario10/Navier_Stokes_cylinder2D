import tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from pinn import PINN
from network import Network
from optimizer import L_BFGS_B



def mass_cons(network, xy):
    """
    Compute u_x and v_y
    Args:
        xy: network input variables as ndarray.
    Returns:
        (u_x, v_y) as ndarray.
    """

    xy = tf.constant(xy)
    x, y = [ xy[..., i, tf.newaxis] for i in range(xy.shape[-1]) ]
    with tf.GradientTape(persistent=True) as g:
      g.watch(x)
      g.watch(y)

      u_v_p = network(tf.concat([x, y], axis=-1))
      u = u_v_p[..., 0, tf.newaxis]
      v = u_v_p[..., 1, tf.newaxis]
      p = u_v_p[..., 2, tf.newaxis]
    u_x = g.batch_jacobian(u, x)[..., 0]
    v_y = g.batch_jacobian(v, y)[..., 0]

    return u_x.numpy(), v_y.numpy()

def u_0(xy):
    """
    Initial wave form.
    Args:
        tx: variables (t, x) as tf.Tensor.
    Returns:
        u(t, x) as tf.Tensor.
    """

    x = xy[..., 0, None]
    y = xy[..., 1, None]


    return    4*y*(1 - y) 


def contour(x, y, z, title, levels=100):
    """
    Contour plot.
    Args:
        grid: plot position.
        x: x-array.
        y: y-array.
        z: z-array.
        title: title string.
        levels: number of contour lines.
    """

    # get the value range
    vmin = np.min(z)
    vmax = np.max(z)
    # plot a contour
    font1 = {'family':'serif','size':20}
    plt.contour(x, y, z, colors='k', linewidths=0.2, levels=levels)
    plt.contourf(x, y, z, cmap='rainbow', levels=levels, norm=Normalize(vmin=vmin, vmax=vmax))
    plt.axes()
    circle = plt.Circle((0.5,0.5),0.1, fc='black')
    plt.gca().add_patch(circle)
    plt.axis('scaled')
    plt.title(title, fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    plt.tick_params(axis='both', which='major', labelsize=15)

    cbar = plt.colorbar(pad=0.03, aspect=25, format='%.0e')
    cbar.mappable.set_clim(vmin, vmax)
    cbar.ax.tick_params(labelsize=15)

if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model
    for the cavity flow governed by the steady Navier-Stokes equation.
    """

    # number of training samples
    num_train_samples = 5000
    # number of test samples
    num_test_samples = 200

    # inlet flow velocity
    u0 = 1
    # density
    rho = 1
    # viscosity
    mu = 1e-1
    # Re = rho/mu

    # build a core network model
    network = Network().build()
    network.summary()
    # build a PINN model
    pinn = PINN(network, rho=rho, mu=mu).build()

    # Domain and circle data
    x_f =2
    x_ini=0
    y_f=1
    y_ini=0
    Cx = 0.5
    Cy = 0.5
    a = 0.1
    b = 0.1

    xyt_circle = np.random.rand(num_train_samples, 2)
    xyt_circle[...,0] = 2*(a)*xyt_circle[...,0] +(Cx-a)
    xyt_circle[0:num_train_samples//2,1] = b*(1 - (xyt_circle[0:num_train_samples//2,0]-Cx)**2 / a**2)**0.5 + Cy
    xyt_circle[num_train_samples//2:,1] = -b*(1 - (xyt_circle[num_train_samples//2:,0]-Cx)**2 / a**2)**0.5 + Cy

    # create training input
    xyt_eqn = np.random.rand(num_train_samples, 2)
    xyt_eqn[...,0] = (x_f - x_ini)*xyt_eqn[...,0] + x_ini
    xyt_eqn[...,1] = (y_f - y_ini)*xyt_eqn[...,1] + y_ini

    for i in range(num_train_samples):
      while (xyt_eqn[i, 0] - Cx)**2/a**2 + (xyt_eqn[i, 1] - Cy)**2/b**2 < 1:
        xyt_eqn[i, 0] = (x_f - x_ini) * np.random.rand(1, 1) + x_ini
        xyt_eqn[i, 1] = (y_f - y_ini) * np.random.rand(1, 1) + y_ini

    xyt_w1 = np.random.rand(num_train_samples, 2)  # top-bottom boundaries
    xyt_w1[..., 0] = (x_f - x_ini)*xyt_w1[...,0] + x_ini
    xyt_w1[..., 1] =  y_ini          # y-position is 0 or 1

    xyt_w2 = np.random.rand(num_train_samples, 2)  # top-bottom boundaries
    xyt_w2[..., 0] = (x_f - x_ini)*xyt_w2[...,0] + x_ini
    xyt_w2[..., 1] =  y_f

    xyt_out = np.random.rand(num_train_samples, 2)  # left-right boundaries
    xyt_out[..., 0] = x_f

    xyt_in = np.random.rand(num_train_samples, 2)
    xyt_in[...,0] = x_ini

    x_train = [xyt_eqn, xyt_w1, xyt_w2, xyt_out, xyt_in, xyt_circle]

    # create training output
    zeros = np.zeros((num_train_samples, 3))
    #uv_bnd[..., 0] = -u0 * np.floor(xy_bnd[..., 0]) +1
    #ones = np.ones((num_train_samples, 3))
    #onze = np.random.rand(num_train_samples, 3)
    #onze[...,0] = u0
    #onze[...,1] = 0
    #onze[...,2] = u0
    a = u_0(tf.constant(xyt_in)).numpy()
    b = np.zeros((num_train_samples, 1))
    onze = np.random.permutation(np.concatenate([a,b,a],axis = -1))

    y_train = [zeros, onze, zeros, zeros, zeros, zeros]

    # train the model using L-BFGS-B algorithm
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()

    # create meshgrid coordinates (x, y) for test plots    

    x = np.linspace(x_ini, x_f, num_test_samples)
    y = np.linspace(y_ini, y_f, num_test_samples)
    x, y = np.meshgrid(x, y)
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)
    # predict (psi, p)
    u_v_p = network.predict(xy, batch_size=len(xy))
    u, v, p = [ u_v_p[..., i].reshape(x.shape) for i in range(u_v_p.shape[-1]) ]
    # compute (u, v)
    u = u.reshape(x.shape)
    v = v.reshape(x.shape)
    p = p.reshape(x.shape)
    # plot test results
    fig = plt.figure(figsize=(16, 8))
    contour(x, y, p, 'p')
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(16, 8))
    contour(x, y, u, 'u')
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(16, 8))
    contour(x, y, v, 'v')
    plt.tight_layout()
    plt.show()
    

    ###########################
    from matplotlib.patches import Circle
    font1 = {'family':'serif','size':20}

    fig0, ax0 = plt.subplots(1, 1,figsize=(20,8))
    cf0 = ax0.contourf(x, y, p, np.arange(-0.2, 1, .02),
                   extend='both',cmap='rainbow')
    cbar0 = plt.colorbar(cf0, pad=0.03, aspect=25, format='%.0e')
    plt.title("p", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    ax0.add_patch(Circle((0.5, 0.5), 0.1,color="black"))
    plt.tick_params(axis='both', which='major', labelsize=15)
    cbar0.ax.tick_params(labelsize=15)
    plt.show()

    ###########################

    fig0, ax0 = plt.subplots(1, 1, figsize=(20,8))
    cf0 = ax0.contourf(x, y, u, np.arange(-0.5, 1.1, .02),
                   extend='both',cmap='rainbow')
    cbar0 = plt.colorbar(cf0, )
    plt.title("u", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    ax0.add_patch(Circle((0.5, 0.5), 0.1,color="black"))
    plt.tick_params(axis='both', which='major', labelsize=15)
    cbar0.ax.tick_params(labelsize=15)
    plt.show()

    ###########################

    fig0, ax0 = plt.subplots(1, 1,figsize=(20,8))
    cf0 = ax0.contourf(x, y, v, np.arange(-0.4, 0.4, .02),
                   extend='both',cmap='rainbow')
    cbar0 = plt.colorbar(cf0, pad=0.03, aspect=25, format='%.0e')
    plt.title("v", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    ax0.add_patch(Circle((0.5, 0.5), 0.1,color="black"))
    plt.tick_params(axis='both', which='major', labelsize=15)
    cbar0.ax.tick_params(labelsize=15)
    plt.show()

    ############################ 

    x = np.linspace(0.3, 1, num_test_samples)
    y = np.linspace(0.3, 0.7, num_test_samples)
    x, y = np.meshgrid(x, y)
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)
    # predict (psi, p)
    u_v_p = network.predict(xy, batch_size=len(xy))
    u, v, p = [ u_v_p[..., i].reshape(x.shape) for i in range(u_v_p.shape[-1]) ]
    # compute (u, v)
    u = u.reshape(x.shape)
    v = v.reshape(x.shape)
    p = p.reshape(x.shape)
    # plot test results
    
    fig = plt.figure(figsize=(15, 8))
    #contour(gs[0, 0], x, y, psi, 'psi')
    contour(x, y, p, 'p')
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(15, 8))
    contour(x, y, u, 'u')
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(15, 8))
    contour(x, y, v, 'v')
    plt.tight_layout()
    plt.show()

    ###########################
    from matplotlib.patches import Circle
    font1 = {'family':'serif','size':20}

    fig0, ax0 = plt.subplots(1, 1,figsize=(18,8))
    cf0 = ax0.contourf(x, y, p, np.arange(-0.2, 0.6, .02),
                   extend='both',cmap='rainbow')
    cbar0 = plt.colorbar(cf0, pad=0.03, aspect=25, format='%.0e')
    plt.title("p", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    ax0.add_patch(Circle((0.5, 0.5), 0.1,color="black"))
    plt.tick_params(axis='both', which='major', labelsize=15)
    cbar0.ax.tick_params(labelsize=15)
    plt.show()

    ###########################

    fig0, ax0 = plt.subplots(1, 1, figsize=(18,8))
    cf0 = ax0.contourf(x, y, u, np.arange(-0.5, 1.1, .02),
                   extend='both',cmap='rainbow')
    cbar0 = plt.colorbar(cf0, )
    plt.title("u", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    ax0.add_patch(Circle((0.5, 0.5), 0.1,color="black"))
    plt.tick_params(axis='both', which='major', labelsize=15)
    cbar0.ax.tick_params(labelsize=15)
    plt.show()

    ###########################

    fig0, ax0 = plt.subplots(1, 1,figsize=(18,8))
    cf0 = ax0.contourf(x, y, v, np.arange(-0.4, 0.4, .02),
                   extend='both',cmap='rainbow')
    cbar0 = plt.colorbar(cf0, pad=0.03, aspect=25, format='%.0e')
    plt.title("v", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    ax0.add_patch(Circle((0.5, 0.5), 0.1,color="black"))
    plt.tick_params(axis='both', which='major', labelsize=15)
    cbar0.ax.tick_params(labelsize=15)
    plt.show()
