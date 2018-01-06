"""BPNN Calculator
"""
from ase.calculators.calculator import Calculator
import tensorflow as tf
import numpy as np
import bpnn.symmetryfunctions as sf
import bpnn.neuralnetworks as nn


class BPNN(Calculator):
    """BPNN Calculator
    """

    def __init__(self,
                 sfConfig=None, nnConfig=None,
                 symbols=None, r_cutoff=None,
                 atoms=None, dataset=None):
        Calculator.__init__(self)
        self.sf_config = sfConfig
        self.nn_config = nnConfig
        self.symbols = symbols
        self.atoms = atoms
        self.dataset = dataset
        self.r_cutoff = r_cutoff

    def update_config(self):
        """
        Update the configurations according to keyworks
        If some variable is not defined, build a default one
        """
        if self.r_cutoff is None:
            self.r_cutoff = 6.

        if self.symbols is None:
            if self.atoms is not None:
                self.symbols = list(set([a.symbol for a in self.atoms]))
            else:
                print('No label specified')
        if self.sf_config is None:
            self._make_defaultsf()
        if self.nn_config is None:
            self._make_defaultnn()

    def _make_defaultsf(self):
        sf_config = {}
        for s_i in self.symbols:
            sf_config[s_i] = {}
            for s_j in self.symbols:
                sf_config[s_i][s_j] = [sf.G1(),
                                       sf.G2(2., 1.),
                                       sf.G2(4., 1.),
                                       sf.G4(1., 1., -1.),
                                       sf.G4(2., 2., 1.)]
        self.sf_config = sf_config

    def _make_defaultnn(self):
        nn_config = {}
        for s_i in self.symbols:
            input_size = sum([len(c) for c in self.sf_config[s_i].values()])
            nn_config[s_i] = [nn.relu('%s1' % s_i, input_size, 10),
                              nn.relu('%s2' % s_i, 10, 10),
                              nn.relu('%s3' % s_i, 10, 10),
                              nn.linear('%s4' % s_i, 10, 1)]
        self.nn_config = nn_config

    def train(self, max_steps=1000, learning_rate=0.001, dataset=None):
        """
        Train the model
        """
        tf.reset_default_graph()
        if dataset is None:
            dataset = self.dataset
        self.update_config()

        if self.symbols is None:
            self.symbols = list(
                set(sum([[a.symbol for a in traj[0]] for traj in dataset],
                        [])))

        print('Constructing Neural Network')
        cost = tf.constant(0.0)
        cost_log = []
        for traj in dataset:
            coord, e_nn = self.construct_model(traj[0])
            e_data = [[atoms.get_potential_energy()] for atoms in traj]
            cost = tf.reduce_mean(tf.square(e_nn - e_data))
            coord_in = [a.get_scaled_positions() for a in traj]
        e_data = np.array(e_data) - np.mean(e_data)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(cost)
        init = tf.global_variables_initializer()

        print('Starting Optimization')
        print('=' * 30)
        print('%-10s%-20s' % ('Steps', 'Energy RMS'))
        print('-' * 30)

        with tf.Session() as sess:
            sess.run(init)
            for i in range(max_steps):
                _, cost_now, e_now = sess.run([optimizer, cost, e_nn],
                                              {coord: coord_in})
                cost_log.append(cost_now)
                if i % 100 == 0:
                    print('%-10i%-20.5f' % (i, cost_now))
        print('=' * 30)
        return cost_log, e_now

    def construct_model(self, atoms):
        """Build the coordinate -> energy model
        """
        coord = tf.placeholder(tf.float32, [None, len(atoms), 3])

        dif_mat = difference_matrix(coord, atoms)
        dis_mat = tf.sqrt(tf.reduce_sum(tf.square(dif_mat), axis=-1))
        fc_mat = tf.cos(np.pi * dis_mat / self.r_cutoff) * 0.5 + 0.5

        diag1 = 1 - tf.expand_dims(tf.eye(len(atoms)), 1)
        diag2 = 1 - tf.expand_dims(tf.eye(len(atoms)), 0)
        diag3 = 1 - tf.expand_dims(tf.eye(len(atoms)), 2)
        diag_filter = diag1 * diag2 * diag3
        diag_filter = tf.expand_dims(diag_filter, 0)

        dif_trans = tf.transpose(dif_mat, perm=[0, 1, 3, 2])
        cos_mat = tf.matmul(dif_mat, dif_trans)
        cos_mat = tf.where(diag_filter >= tf.ones_like(cos_mat),
                           cos_mat /
                           tf.expand_dims(dis_mat, 2) /
                           tf.expand_dims(dis_mat, 3),
                           tf.zeros_like(cos_mat))

        energy = 0
        for i, atom_i in enumerate(atoms):
            sym_funcs = []
            for s_j, sym_func_j in self.sf_config[atom_i.symbol].items():
                slist = [atom.index for atom in atoms if atom.symbol is s_j]
                dis_mat_filtered = tf.gather(tf.gather(dis_mat, [i], axis=1),
                                             slist, axis=2)
                fc_mat_filtered = tf.gather(tf.gather(fc_mat, [i], axis=1),
                                            slist, axis=2)
                cos_mat_filtered = tf.gather(tf.gather(tf.gather(cos_mat,
                                                                 [i], axis=1),
                                                       slist, axis=2),
                                             slist, axis=3)
                for sym_func in sym_func_j:
                    sym_funcs.append(
                        sym_func(fc=fc_mat_filtered, Rij=dis_mat_filtered,
                                 cos=cos_mat_filtered))
            sym_funcs = tf.concat(sym_funcs, axis=1)
            nn_input = sym_funcs
            for nn_i in self.nn_config[atom_i.symbol]:
                nn_input = nn_i(nn_input)
            energy += nn_input
        return coord, energy


def difference_matrix(coord, atoms):
    """Helper function, calculates pairwise distances according to
    the periodic boundary condition
    """
    dif_mat = tf.expand_dims(coord, 1) - tf.expand_dims(coord, 2)  # (m,n,n,3)
    dif_mat_pbc = []
    for i, pbc in enumerate(atoms.pbc):
        dif_i = tf.gather(dif_mat, [i], axis=3)
        if pbc:
            dif_i = tf.where(dif_i < 0.5,
                             dif_i,
                             dif_i - tf.sign(dif_i))

        dif_mat_pbc.append(dif_i)
    dif_mat_pbc = tf.concat(dif_mat_pbc, 3)

    cell = tf.constant(atoms.get_cell(), shape=[3, 3], dtype=tf.float32)
    dif_mat_pbc = tf.tensordot(dif_mat_pbc, cell, axes=1)
    return dif_mat_pbc
