import numpy as np
import math
from curvature_term import meancurvature
from redistance import sussman as ss
from matplotlib import pyplot as ppl

class chanvesse_functional:
    _PHI_LOWER = -2
    _PHI_UPPER = +2
    _eps = 2.2204e-16
    _iter = 2000
    _alpha = 0.005
    _tol = 1e-18

    def __init__(self, img, phi_init, dt):
        self.I = img
        self.nrow, self.ncol = self.I.shape[0], self.I.shape[1]
        self.phi = phi_init.get_phi()
        self.dt = dt
        self.rdist = ss(dt=self.dt)

    def run_chanvesse(self):
        fig = ppl.gcf()
        fig.clf()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.imshow(self.I, cmap=ppl.cm.gray)
        ax1.contour(self.phi, [0.5], colors='r')
        ppl.pause(0.001)
        Dold = -9999
        curvature_obj = meancurvature(rows=self.nrow, cols=self.ncol)
        for s in range(self._iter):
            # Get narrow band
            C = np.logical_and(self.phi < chanvesse_functional._PHI_UPPER, self.phi > chanvesse_functional._PHI_LOWER)
            narrow_band = np.where(C == True)
            narrow_band_point_loc = [x for x in zip(narrow_band[0], narrow_band[1])]

            ## Get in out points
            in_pt_indices = np.where(self.phi < 0)
            out_pt_indices = np.where(self.phi > 0)

            ## means in and out
            mu_in = np.mean(self.I[in_pt_indices])
            mu_out = np.mean(self.I[out_pt_indices])

            # Computations
            D = math.sqrt((mu_in - mu_out)*(mu_in - mu_out)) #Energy functional
            d_T = self.grad_phi(mu_in=mu_in,mu_out=mu_out,narrow_band=narrow_band) # grad phi
            K =  self._alpha * curvature_obj.get_mean_curvature(
                narrow_band=narrow_band,narrow_band_point_loc=narrow_band_point_loc,phi=self.phi) # curvature

            e = np.add(d_T,  K)
            max_e = max(abs(e))

            energy_change = (self.dt / (max_e + self._eps)) * e
            self.phi[C] = np.add(self.phi[C], energy_change)

            if s % 10 == 0:
                self.phi = self.rdist.sussman_redistancing(self.phi)
                fig.clf()
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.imshow(self.I, cmap=ppl.cm.gray)
                ax1.contour(self.phi, [0.5], colors='r')
                ppl.pause(0.001)
                if s% 100 ==0:
                    print( s, 'mu_in:: ', mu_in, ' mu_out::', mu_out, 'E:: ', D)
                    if abs(D - Dold) < chanvesse_functional._tol:
                        print("Energy Maximization achieved, discontinuing iterations")
                        break
                    else:
                        Dold = D

    def grad_phi(self,mu_in, mu_out, narrow_band):
        d_T = np.subtract(np.square(self.I[narrow_band] - mu_in),np.square(self.I[narrow_band] - mu_out))
        return d_T

class yezzi_functional:
    _PHI_LOWER = -2
    _PHI_UPPER = +2
    _eps = 2.2204e-16
    _iter = 2000
    _alpha = 0.005
    _tol = 1e-18

    def __init__(self, img, phi_init, dt):
        self.I = img
        self.nrow, self.ncol = self.I.shape[0], self.I.shape[1]
        self.phi = phi_init.get_phi()
        self.dt = dt
        self.rdist = ss(dt=self.dt)

    def run_yezzi(self):
        fig = ppl.gcf()
        fig.clf()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.imshow(self.I, cmap=ppl.cm.gray)
        ax1.contour(self.phi, [0.5], colors='r')
        ppl.pause(0.001)
        Dold = -9999
        curvature_obj = meancurvature(rows=self.nrow, cols=self.ncol)
        for s in range(self._iter):
            # Get narrow band
            C = np.logical_and(self.phi < yezzi_functional._PHI_UPPER, self.phi > yezzi_functional._PHI_LOWER)
            narrow_band = np.where(C == True)
            narrow_band_point_loc = [x for x in zip(narrow_band[0], narrow_band[1])]

            ## Get in out points
            in_pt_indices = np.where(self.phi < 0)
            out_pt_indices = np.where(self.phi > 0)

            ## Area in and Area out
            a_in = len(in_pt_indices)
            a_out = len(out_pt_indices)

            ## means in and out
            mu_in = np.mean(self.I[in_pt_indices])
            mu_out = np.mean(self.I[out_pt_indices])

            # Computations
            D = -0.5*math.sqrt((mu_in - mu_out)*(mu_in - mu_out)) #Energy functional
            d_T = self.grad_phi(mu_in=mu_in,mu_out=mu_out,a_in=a_in, a_out=a_out,narrow_band=narrow_band) # grad phi
            K =  self._alpha * curvature_obj.get_mean_curvature(
                narrow_band=narrow_band,narrow_band_point_loc=narrow_band_point_loc,phi=self.phi) # curvature

            # d_Phi/d_t
            e = np.add(d_T,  K)
            max_e = max(abs(e))

            energy_change = (self.dt / (max_e + self._eps)) * e
            self.phi[C] = np.add(self.phi[C], energy_change)

            if s % 10 == 0:
                self.phi = self.rdist.sussman_redistancing(self.phi)
                fig.clf()
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.imshow(self.I, cmap=ppl.cm.gray)
                ax1.contour(self.phi, [0.5], colors='r')
                ppl.pause(0.001)
                if s% 100 ==0:
                    print( s, 'mu_in:: ', mu_in, ' mu_out::', mu_out, 'E:: ', D)
                    if abs(D - Dold) < chanvesse_functional._tol:
                        print("Energy Maximization achieved, discontinuing iterations")
                        break
                    else:
                        Dold = D

    def grad_phi(self,mu_in, mu_out, a_in, a_out, narrow_band):
        d_T = -(mu_in - mu_out)*np.add((self.I[narrow_band] - mu_in)/a_in,(self.I[narrow_band] - mu_out)/a_out)
        return d_T


class bhattacharya_functional:
    _PHI_LOWER = -2
    _PHI_UPPER = +2
    _eps = 2.2204e-16
    _iter = 2000
    _alpha = 0.005
    _tol = 1e-18

    def __init__(self, img, phi_init, dt):
        self.I = img
        self.nrow, self.ncol = self.I.shape[0], self.I.shape[1]
        self.phi = phi_init.get_phi()
        self.dt = dt
        self.rdist = ss(dt=self.dt)

    def setup_pdf(self, intensities):
        pass

    def run_bhattacharya_functional(self):
        pass

