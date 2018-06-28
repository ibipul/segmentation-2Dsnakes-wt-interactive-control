import numpy as np
import math

class sussman:

    def __init__(self,dt = 0.5):
        self.dt = dt

    def set_phi(self,phi):
        self.NROW, self.NCOL = phi.shape[0], phi.shape[1]
        self.phi = phi

    def change_dt(self,dt):
        self.dt = dt

    # Sussman Shift operations
    def shiftR(self, phi):
        x0 = phi[:, 0]
        x1 = phi[:, 0:self.NCOL - 1]
        return np.c_[x0, x1]

    def shiftL(self, phi):
        x0 = phi[:, 1:self.NCOL]
        x1 = phi[:, -1]
        return np.c_[x0, x1]

    def shiftD(self, phi):
        x0 = phi[0, :]
        x1 = phi[0:(self.NROW - 1), :]
        return np.row_stack([x0, x1])

    def shiftU(self, phi):
        x0 = phi[1:self.NROW, :]
        x1 = phi[-1, :]
        return np.row_stack([x0, x1])

    def sussman_redistancing(self,new_phi):

        # Set the phi before redistancing is done
        self.set_phi(phi=new_phi)

        # Apply shift operations
        a = new_phi - self.shiftR(new_phi)  # right
        b = self.shiftL(new_phi) - new_phi  # left
        c = new_phi - self.shiftD(new_phi)  # backward
        d = self.shiftU(new_phi) - new_phi  # forward

        # Setting up default matrices
        a_pos = np.copy(a)
        a_neg = np.copy(a)
        b_pos = np.copy(b)
        b_neg = np.copy(b)
        c_pos = np.copy(c)
        c_neg = np.copy(c)
        d_pos = np.copy(d)
        d_neg = np.copy(d)

        # Getting Indices of various types
        agz = a < 0
        alz = a > 0
        bgz = b < 0
        blz = b > 0
        cgz = c < 0
        clz = c > 0
        dgz = d < 0
        dlz = d > 0

        a_pos[agz] = 0
        a_neg[alz] = 0
        b_pos[bgz] = 0
        b_neg[blz] = 0
        c_pos[cgz] = 0
        c_neg[clz] = 0
        d_pos[dgz] = 0
        d_neg[dlz] = 0

        apsq = np.square(a_pos)
        ansq = np.square(a_neg)
        bpsq = np.square(b_pos)
        bnsq = np.square(b_neg)
        cpsq = np.square(c_pos)
        cnsq = np.square(c_neg)
        dpsq = np.square(d_pos)
        dnsq = np.square(d_neg)


        d_phi = np.zeros(np.shape(new_phi))
        phi_lez = new_phi < 0
        neg_index = [x for x in zip(np.where(phi_lez)[0], np.where(phi_lez)[1])]
        phi_gez = new_phi > 0
        pos_index = [x for x in zip(np.where(phi_gez)[0], np.where(phi_gez)[1])]

        dposindex = []
        p1 = []
        p2 = []
        for p in pos_index:
            p1.append(max(apsq[p], bnsq[p]))
            p2.append(max(cpsq[p], dnsq[p]))
            d_phi[p] = np.sqrt(max(apsq[p], bnsq[p]) + max(cpsq[p], dnsq[p])) - 1
            dposindex.append(d_phi[p])

        for p in neg_index:
            d_phi[p] = math.sqrt(max(ansq[p], bpsq[p]) + max(cnsq[p], dpsq[p])) - 1

        susman_sign = np.divide(new_phi, np.sqrt(np.square(new_phi) + 1))

        new_phi -= np.multiply(self.dt * susman_sign, d_phi)

        return new_phi