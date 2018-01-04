import tensorflow as tf
class G1:
    def __init__(self):
        pass

    def __repr__(self):
        return ('G1')

    def __call__(self, **var):
        return tf.reduce_sum(var['fc'],axis=-1)


class G2:
    def __init__(self, eta, Rs):
        self.eta = eta
        self.Rs = Rs

    def __repr__(self):
        return ('G2(%.2f, %.2f)'%(self.eta, self.Rs))

    def __call__(self, **var):
        G2 = tf.exp(-self.eta*tf.square(var['Rij']-self.Rs)) * var['fc']
        return tf.reduce_sum(G2,axis=-1)


class G4:
    def __init__(self, eta=1., zeta=1., lambd=1.):
        self.eta = eta
        self.zeta = zeta
        self.lambd = lambd

    def __repr__(self):
        return ('G3(%.2f, %.2f, %.2f)'%(self.eta, self.eta, self.lambd))

    def __call__(self, **var):
        expo = tf.exp(-self.eta*(tf.expand_dims(var['Rij'],-1)+
                                 tf.expand_dims(var['Rij'],-2)))
        cosin = tf.pow(var['cos'] * self.lambd + 1,self.zeta)
        cutoff = (tf.expand_dims(var['fc'],-1) *
                  tf.expand_dims(var['fc'],-2))
        G4 = tf.reduce_sum(tf.reduce_sum(2.**(1-self.zeta)*expo*cutoff*cosin,axis=-1),axis=-1)
        return G4
