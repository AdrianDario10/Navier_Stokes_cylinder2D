import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute derivatives for the steady Navier-Stokes equation.
    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    def call(self, xyt):
        """
        Computing derivatives for the steady Navier-Stokes equation.
        Args:
            xy: input variable.
        Returns:
            psi: stream function.
            p_grads: pressure and its gradients.
            u_grads: u and its gradients.
            v_grads: v and its gradients.
        """

        x, y = [ xyt[..., i, tf.newaxis] for i in range(xyt.shape[-1]) ]
        with tf.GradientTape(persistent=True) as ggg:
            ggg.watch(x)
            ggg.watch(y)

            with tf.GradientTape(persistent=True) as gg:
                gg.watch(x)
                gg.watch(y)

                with tf.GradientTape(persistent=True) as g:
                    g.watch(x)
                    g.watch(y)

                    u_v_p = self.model(tf.concat([x, y], axis=-1))
                    u = u_v_p[..., 0, tf.newaxis]
                    v = u_v_p[..., 1, tf.newaxis]
                    p = u_v_p[..., 2, tf.newaxis]
                u_x = g.batch_jacobian(u, x)[..., 0]
                v_x = g.batch_jacobian(v, x)[..., 0]
                u_y = g.batch_jacobian(u, y)[..., 0]
                v_y = g.batch_jacobian(v, y)[..., 0]
                p_x = g.batch_jacobian(p, x)[..., 0]
                p_y = g.batch_jacobian(p, y)[..., 0]

                del g
            u_xx = gg.batch_jacobian(u_x, x)[..., 0]
            u_yy = gg.batch_jacobian(u_y, y)[..., 0]

            v_xx = gg.batch_jacobian(v_x, x)[..., 0]
            v_yy = gg.batch_jacobian(v_y, y)[..., 0]

            del gg
        # if more derivatives are required...
        del ggg

        p_grads = p, p_x, p_y
        u_grads = u, u_x, u_y, u_xx, u_yy
        v_grads = v, v_x, v_y, v_xx, v_yy

        return p_grads, u_grads, v_grads
