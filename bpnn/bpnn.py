from ase.calculators.calculator import Calculator
from ase.neighborlist import NeighborList
import tensorflow as tf
import bpnn.symmetryfunctions as sf
import bpnn.neuralnetworks as nn
import numpy as np


class BPNN(Calculator):
    def __init__(self,
                 sfConfig=None, nnConfig=None,
                 symbols=None, Rc=None,
                 atoms=None, dataset=None):
        self.sfConfig = sfConfig
        self.nnConfig = nnConfig
        self.symbols = symbols
        self.Rc = Rc

    def update_config(self):
        if self.Rc is None:
            self.Rc = 6.

        if self.symbols is None:
            if self.atoms is not None:
                self.symbols = list(set([a.symbol for a in atoms]))
            else:
                raise Expection('No label specified')
        if self.sfConfig is None:
            self._make_defaultsf()
        if self.nnConfig is None:
            self._make_defaultnn()

    def _make_defaultsf(self):
        sfConfig = {}
        for si in self.symbols:
            sfConfig[si]={}
            for sj in self.symbols:
                sfConfig[si][sj]=[sf.G1(),
                                  sf.G2(2., 1.),
                                  sf.G2(4., 1.),
                                  sf.G4(1., 1., -1.),
                                  sf.G4(2., 2., 1.)]
        self.sfConfig = sfConfig

    def _make_defaultnn(self):
        nnConfig = {}
        for s in self.symbols:
            input_size = sum([len(c) for c in self.sfConfig[s].values()])
            nnConfig[s]=[nn.relu('%s1'%s,input_size,10),
                         nn.relu('%s2'%s,10,10),
                         nn.relu('%s3'%s,10,10),
                         nn.linear('%s4'%s,10,1)]
        self.nnConfig = nnConfig

    def train(self, max_steps=1000, learning_rate=0.001, dataset=None):
        tf.reset_default_graph()
        if dataset is None:
            dataset = self.datase
        self.update_config()

        if self.symbols is None:
            self.symbols = list(set(sum([[a.symbol for a in traj[0]] for traj in dataset],[])))
        if self.sfConfig is None:
            self.construct_default_sf(self.symbols)
        if self.nnConfig is None:
            self.construct_default_nn(self.symbols)
        print('Constructing Neural Network')
        cost = tf.constant(0.0)
        cost_log = []
        for traj in dataset:
            coord, e_nn = self.construct_model(traj[0])
            e_data = [[atoms.get_potential_energy()] for atoms in traj]
            cost = tf.reduce_mean(tf.square(e_nn-e_data))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        print('Starting Optimization')
        print('='*30)
        print('%-10s%-20s'%('Steps','Energy RMS'))
        print('-'*30)
        with tf.Session() as sess:
            sess.run(init)
            for i in range(max_steps):
                _,cost_now,e = sess.run([optimizer,cost,e_nn],
                                  {coord:[a.get_scaled_positions() for a in traj]})
                cost_log.append(cost_now)
                if i%100 == 0:
                    print('%-10i%-20.5f'%(i,cost_now))
        print('='*30)
        return cost_log

    def construct_model(self,atoms):
        coord = tf.placeholder(tf.float32, [None, len(atoms), 3])

        dif_mat = difference_matrix(coord, atoms)
        dis_mat = tf.sqrt(tf.reduce_sum(tf.square(dif_mat),axis=-1))
        fc_mat = tf.cos(np.pi*dis_mat/self.Rc)*0.5+0.5

        diag1 = 1-tf.expand_dims(tf.eye(len(atoms)),1)
        diag2 = 1-tf.expand_dims(tf.eye(len(atoms)),0)
        diag3 = 1-tf.expand_dims(tf.eye(len(atoms)),2)
        diag_filter = diag1*diag2*diag3
        diag_filter = tf.expand_dims(diag_filter,0)

        dif_trans = tf.transpose(dif_mat,perm=[0,1,3,2])
        cos_mat = tf.matmul(dif_mat,dif_trans)
        cos_mat = tf.where(diag_filter>=tf.ones_like(cos_mat),
                           cos_mat / tf.expand_dims(dis_mat,2) /tf.expand_dims(dis_mat,3),
                           tf.zeros_like(cos_mat))

        e = 0
        for i,a in enumerate(atoms):
            sym_funcs = []
            for s,Gs in self.sfConfig[a.symbol].items():
                slist = [ai.index for ai in atoms if ai.symbol is s]
                dis_mat_filtered = tf.gather(tf.gather(dis_mat,[i],axis=1),
                                             slist, axis=2)
                fc_mat_filtered = tf.gather(tf.gather(fc_mat,[i],axis=1),
                                             slist, axis=2)
                cos_mat_filtered = tf.gather(tf.gather(tf.gather(cos_mat,
                                                                 [i],axis=1),
                                                       slist,axis=2),
                                             slist,axis=3)
                for G in Gs:
                    sym_funcs.append(
                        G(fc=fc_mat_filtered, Rij=fc_mat_filtered, cos=cos_mat_filtered))
            sym_funcs = tf.concat(sym_funcs, axis=1)
            nn_input = sym_funcs
            for nn in self.nnConfig[a.symbol]:
                nn_input = nn(nn_input)
            e += nn_input
        return coord,  e


def difference_matrix(coord,atoms):
    dif_mat = tf.expand_dims(coord,1) - tf.expand_dims(coord,2) # (m,n,n,3)
    dif_mat_pbc=[]
    for i,pbc in enumerate(atoms.pbc):
        dif_i = tf.gather(dif_mat,[i], axis=3)
        if pbc:
            dif_i = tf.where(dif_i<0.5,
                             dif_i,
                             dif_i - tf.sign(dif_i))

        dif_mat_pbc.append(dif_i)
    dif_mat_pbc = tf.concat(dif_mat_pbc,3)

    cell = tf.constant(atoms.get_cell(), shape=[3,3],dtype=tf.float32)
    dif_mat_pbc = tf.tensordot(dif_mat_pbc,cell,axes=1)
    return dif_mat_pbc
