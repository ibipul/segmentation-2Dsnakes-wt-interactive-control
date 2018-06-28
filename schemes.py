import numpy as np
import math
from curvature_term import meancurvature
from redistance import sussman as ss
from matplotlib import pyplot as ppl
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter

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
        ax1.contour(self.phi, [-1,+1], colors='r')
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
                ax1.contour(self.phi, [-1,+1], colors='r')
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
            a_in = len(in_pt_indices[0])
            a_out = len(out_pt_indices[0])

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
    _alpha = 0.00001
    _tol = 1e-18
    _biom_filt = np.array([0.0000, 0.0005,0.0032,0.0139,0.0417,0.0916,0.1527,0.1964,0.1964,0.1527,0.0916,0.0417,0.0139,0.0032,0.0005,0.0000])

    def __init__(self, img, phi_init, dt):
        self.I = img
        self.nrow, self.ncol = self.I.shape[0], self.I.shape[1]
        self.phi = phi_init.get_phi()
        self.dt = dt
        self.rdist = ss(dt=self.dt)

    def setup_pdf(self, intensities):
        # Get histogram
        hist, bin_edges = np.histogram(intensities, bins=np.arange(257))
        # Updating missing intensities with 1
        hist[hist == 0] = 1
        hx = np.sum(hist)
        hist = hist / float(hx)
        #Smoothing the hist with a 2 binomial filter
        smooth_hist = np.convolve(hist,bhattacharya_functional._biom_filt,'same')
        hy = np.sum(smooth_hist) + bhattacharya_functional._eps
        smooth_hist = smooth_hist / float(hy)
        return smooth_hist

    def get_delta(self,intensity):
        delta_arr = np.zeros(256)
        delta_arr[intensity] = 1
        return delta_arr

    def run_bhattacharya(self):
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
            C = np.logical_and(self.phi < bhattacharya_functional._PHI_UPPER, self.phi > bhattacharya_functional._PHI_LOWER)
            narrow_band = np.where(C == True)
            narrow_band_point_loc = [x for x in zip(narrow_band[0], narrow_band[1])]

            ## Get in out points
            in_pt_indices = np.where(self.phi < 0)
            out_pt_indices = np.where(self.phi > 0)

            ## Area in and Area out
            a_in = len(in_pt_indices[0])
            a_out = len(out_pt_indices[0])

            ## means in and out
            mu_in = np.mean(self.I[in_pt_indices])
            mu_out = np.mean(self.I[out_pt_indices])

            ## means in and out
            p_in = self.setup_pdf(self.I[in_pt_indices])
            p_out = self.setup_pdf(self.I[out_pt_indices])

            # Computations
            D =  np.sum(np.sqrt(np.dot(p_in+bhattacharya_functional._eps,p_out+bhattacharya_functional._eps))) # Energy functional

            # Energy update pieces
            d_T = self.grad_phi(D=D, p_in=p_in, p_out=p_out, a_in=a_in, a_out=a_out, narrow_band=narrow_band)  # grad phi
            K = self._alpha * curvature_obj.get_mean_curvature(
                narrow_band=narrow_band, narrow_band_point_loc=narrow_band_point_loc, phi=self.phi)  # curvature

            # d_Phi/d_t
            e = np.add(d_T, K)
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
                if s % 100 == 0:
                    print(s, 'mu_in:: ', mu_in, ' mu_out::', mu_out, 'E:: ', D)
                    if abs(D - Dold) < chanvesse_functional._tol:
                        print("Energy Minimization achieved, discontinuing iterations")
                        break
                    else:
                        Dold = D

    def grad_phi(self, D, p_in, p_out, a_in, a_out, narrow_band):
        t_0 = -0.5*D * (1/float(a_in) - 1/float(a_out))
        X = (1/float(a_out))* np.sqrt(np.divide(p_in + bhattacharya_functional._eps, p_out + bhattacharya_functional._eps))
        Y = (1/float(a_in)) * np.sqrt(np.divide(p_out + bhattacharya_functional._eps,p_in + bhattacharya_functional._eps))
        Z = X - Y
        t_1 = -0.5* Z[self.I[narrow_band]]
        d_T = t_0 + t_1
        return d_T

class seg_w_control:
    _PHI_LOWER = -2
    _PHI_UPPER = +2
    _eps = 2.2204e-16
    _iter = 5000
    _alpha = 0.00001
    _tol = 1e-18

    def __init__(self, img, phi_init, dt):
        self.I = img
        self.nrow, self.ncol = self.I.shape[0], self.I.shape[1]
        self.phi = -phi_init.get_phi()
        self.phi_hat = -phi_init.get_phi()
        self.umask = None
        self.U = np.zeros(self.phi.shape)
        self.F = np.zeros(self.phi.shape)
        self.dt = dt
        self.rdist = ss(dt=self.dt)


    def run(self):
        fig = ppl.gcf()
        fig.clf()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(self.I, cmap=ppl.cm.gray)
        ax1.contour(self.phi, [0.5], colors='r')
        ax2.imshow(self.phi,cmap='hot')
        ppl.pause(0.001)
        curvature_obj = meancurvature(rows=self.nrow, cols=self.ncol)
        for s in range(self._iter):
            # Get narrow band - phi
            C = np.logical_and(self.phi < self._PHI_UPPER,
                               self.phi > self._PHI_LOWER)
            narrow_band = np.where(C == True)
            narrow_band_point_loc = [x for x in zip(narrow_band[0], narrow_band[1])]

            # Get narrow band - phi_hat
            C_hat = np.logical_and(self.phi_hat < self._PHI_UPPER,
                               self.phi_hat > self._PHI_LOWER)
            narrow_band_hat = np.where(C_hat == True)
            narrow_band_point_loc_hat = [x for x in zip(narrow_band_hat[0], narrow_band_hat[1])]

            # Get Narrow band -- U
            C_U = np.logical_and(self.phi < self._PHI_UPPER,
                                   self.phi > self._PHI_LOWER)
            narrow_band_U = np.where(C_U == True)
            narrow_band_point_loc_U = [x for x in zip(narrow_band_U[0], narrow_band_U[1])]

            # Get in out points - phi
            in_pt_indices = np.where(self.phi > 0)
            out_pt_indices = np.where(self.phi < 0)

            ## Get in out points - phi_hat
            in_pt_indices_hat = np.where(self.phi_hat > 0)
            out_pt_indices_hat = np.where(self.phi_hat < 0)

            ## Get in out points - U
            in_pt_indices_U = np.where(self.U > 0)
            out_pt_indices_U = np.where(self.U < 0)

            ## means in and out - phi
            mu_in = np.mean(self.I[in_pt_indices])
            mu_out = np.mean(self.I[out_pt_indices])
            self.phi_hat_in = np.mean(self.phi_hat[in_pt_indices_hat])

            # H phi hat - H phi
            del_phi = np.subtract(np.multiply(self.H(phi=self.phi_hat, in_pt_indices=in_pt_indices_hat),self.phi_hat),
                                  np.multiply(self.H(phi=self.phi, in_pt_indices=in_pt_indices),self.phi))
            # H phi - H phi hat
            eta = -1 * del_phi

            G = self.grad_phi(mu_in=mu_in,mu_out=mu_out,narrow_band=narrow_band) #Energy term

            # supremum { |G|, |G_c|} *
            F = np.multiply(np.absolute(G), del_phi)#np.add(np.ones(del_phi.shape)*10,np.multiply(np.maximum(G, G_c), del_phi))  # Control term
            K = curvature_obj.get_mean_curvature_matrix(narrow_band=narrow_band,narrow_band_point_loc=narrow_band_point_loc,phi=self.phi)

            energy_change = np.add(G,K)
            energy_change = np.add(energy_change, F)
            emax = np.amax(energy_change)
            self.phi[narrow_band] = np.add(self.phi[narrow_band], (self.dt / (emax + self._eps)) * energy_change[narrow_band])


            Eu = np.subtract(np.multiply(self.H(phi=self.phi_hat, in_pt_indices=in_pt_indices_hat),self.phi_hat),
                             np.multiply(self.H(phi=self.U, in_pt_indices=in_pt_indices_U),self.U))
            f_eu = -np.multiply(Eu,np.absolute(self.U))
            #f_eu = f_eu/np.amax(f_eu)

            hat_energy_change = np.add(eta, f_eu)
            hat_emax = np.amax(hat_energy_change)


            # self.phi = np.add(self.phi, self.dt* (np.add(energy_change,F)))
            # self.phi_hat = np.add(self.phi_hat, self.dt* (np.add(eta,f_eu)))

            self.phi_hat[narrow_band_hat] = np.add(self.phi_hat[narrow_band_hat], (self.dt/(hat_emax + self._eps))* hat_energy_change[narrow_band_hat])
            #self.phi_hat = np.add(self.phi_hat, (self.dt * self.phi_hat_in / (hat_emax + self._eps)) * hat_energy_change)

            if s % 10 == 0:
                self.phi = self.rdist.sussman_redistancing(self.phi)
                self.phi_hat = self.rdist.sussman_redistancing(self.phi_hat)
                self.U = self.rdist.sussman_redistancing(self.U)

                fig = ppl.gcf()
                fig.clf()
                ax1 = fig.add_subplot(141)
                ax2 = fig.add_subplot(142)
                ax3 = fig.add_subplot(143)
                ax4 = fig.add_subplot(144)
                ax1.imshow(self.I, cmap=ppl.cm.gray)
                ax1.contour(self.phi, [0], colors='r')

                ax2.imshow(self.phi, cmap='hot')
                ax2.contour(self.phi, [0], colors='r')

                ax3.imshow(self.phi_hat, cmap='hot')
                ax3.contour(self.phi_hat, [0], colors='b')

                ax4.imshow(self.U, cmap='hot')
                ax4.contour(self.phi, [0], colors='g')


                ppl.pause(0.001)
                if s % 200 == 0:
                    print(s, 'mu_in:: ', mu_in, ' mu_out::', mu_out, 'U in: ', len(in_pt_indices_U[0]), 'U_out: ', len(out_pt_indices_U[0]),' U sum: ', self.U.sum())
                    x, y = np.meshgrid(np.arange(self.I.shape[1]), np.arange(self.I.shape[0]))
                    pix = np.vstack((x.flatten(), y.flatten())).T

                    def onselect(verts):
                        # Select elements in original array bounded by selector path:
                        p = Path(verts)
                        ind = p.contains_points(pix, radius=1)
                        selected = np.zeros_like(self.I)
                        selected.flat[ind] = self.I.flat[ind]
                        self.umask =  (selected > 0) *0.8
                        #self.h_d = distance(1 - self.umask)
                        #d_max = np.amax(self.h_d)
                        #d_min = np.amin(self.h_d)
                        #self.h_d = (1/(d_max - d_min))*(d_max*np.ones(self.h_d.shape) - self.h_d)
                        self.U = np.add(self.U, self.umask)
                        self.U = medfilt(self.U,5)
                        #self.U = gaussian_filter(self.U, 5)
                        ppl.close()

                    m = LassoSelector(ax1, onselect)
                    ppl.show()



    def grad_phi(self,mu_in, mu_out,narrow_band):
        d_T = np.zeros(self.phi.shape)
        d_T[narrow_band] = -1* np.subtract(np.square(self.I[narrow_band] - mu_in),np.square(self.I[narrow_band] - mu_out))
        return d_T



    def H(self, phi, in_pt_indices):
        base = np.zeros(phi.shape)
        base[in_pt_indices] = 1
        return base


class optimal_segmentation:
    _PHI_LOWER = -2
    _PHI_UPPER = +2
    _eps = 2.2204e-16
    _iter = 5000
    _alpha = 0.00001
    _tol = 1e-18

    def __init__(self, img, phi_init, dt):
        self.I = img
        self.nrow, self.ncol = self.I.shape[0], self.I.shape[1]
        self.phi = -phi_init.get_phi()
        self.phi_hat = -phi_init.get_phi()
        self.umask = None
        self.U = np.zeros(self.phi.shape)
        self.F = np.zeros(self.phi.shape)
        self.dt = dt
        self.rdist = ss(dt=self.dt)

    def run_optimal_segmentation(self):
        fig = ppl.gcf()
        fig.clf()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(self.I, cmap=ppl.cm.gray)
        ax1.contour(self.phi, [0.5], colors='r')
        ax2.imshow(self.phi, cmap='hot')
        ppl.pause(0.001)
        curvature_obj = meancurvature(rows=self.nrow, cols=self.ncol)
        for s in range(self._iter):
            # Get narrow band - phi
            C = np.logical_and(self.phi < self._PHI_UPPER,
                               self.phi > self._PHI_LOWER)
            narrow_band = np.where(C == True)
            narrow_band_point_loc = [x for x in zip(narrow_band[0], narrow_band[1])]

            # Get narrow band - phi_hat
            C_hat = np.logical_and(self.phi_hat < self._PHI_UPPER,
                                   self.phi_hat > self._PHI_LOWER)
            narrow_band_hat = np.where(C_hat == True)
            narrow_band_point_loc_hat = [x for x in zip(narrow_band_hat[0], narrow_band_hat[1])]

            # Get in out points - phi
            in_pt_indices = np.where(self.phi > 0)
            out_pt_indices = np.where(self.phi < 0)

            ## Get in out points - phi_hat
            in_pt_indices_hat = np.where(self.phi_hat > 0)
            out_pt_indices_hat = np.where(self.phi_hat < 0)

            ## Get in out points - U
            in_pt_indices_U = np.where(self.U >= 0)
            out_pt_indices_U = np.where(self.U < 0)

            ## means in and out - phi
            mu_in = np.mean(self.I[in_pt_indices])
            mu_out = np.mean(self.I[out_pt_indices])

            # H(phi*) - H(phi)  = -Eta
            del_phi = np.subtract(self.H(phi=self.phi_hat, in_pt_indices=in_pt_indices_hat),
                                  self.H(phi=self.phi, in_pt_indices=in_pt_indices))

            # Evolution of Phi
            G = self.grad_phi(mu_in=mu_in, mu_out=mu_out, narrow_band=narrow_band)  # Energy term for narrow band of phi
            K = curvature_obj.get_mean_curvature(narrow_band=narrow_band, narrow_band_point_loc=narrow_band_point_loc, phi=self.phi) #Curvature along level- 0 curve
            F = np.multiply(np.absolute(G), del_phi[narrow_band])

            # COmputing and Normalizing energy change to be between -1 tp 1
            energy_change = np.zeros(narrow_band[0].shape)
            energy_change = np.add(energy_change,G)
            #energy_change = np.add(energy_change, K)
            #energy_change = np.add(energy_change, F)
            emax = np.amax(energy_change)

            # Phi snake
            self.phi[narrow_band] = np.add(self.phi[narrow_band],(self.dt/(emax + self._eps))*energy_change)

            #### Evolution of phi*
            ## H(phi*) - H(U)
            Eu = np.subtract(self.H(phi=self.phi_hat, in_pt_indices=in_pt_indices_hat),
                             self.H(phi=self.U, in_pt_indices=in_pt_indices_U))

            # H(phi) - H(phi*) = Eta
            eta = -1 * del_phi

            # Computing and normalizing the hat evolution
            hat_energy_change = np.add(eta, -1*np.multiply(np.absolute(self.U),Eu))[narrow_band_hat]
            hmax = np.amax(hat_energy_change)
            hmin = np.amin(hat_energy_change)

            # Phi* snake
            self.phi_hat[narrow_band_hat] = np.add(self.phi_hat[narrow_band_hat],(self.dt / (hmax + self._eps)) * hat_energy_change)

            if s % 10 == 0:
                self.phi = self.rdist.sussman_redistancing(self.phi)
                self.phi_hat = self.rdist.sussman_redistancing(self.phi_hat)
                self.U = self.rdist.sussman_redistancing(self.U)

                fig = ppl.gcf()
                fig.clf()
                ax1 = fig.add_subplot(141)
                ax2 = fig.add_subplot(142)
                ax3 = fig.add_subplot(143)
                #ax4 = fig.add_subplot(144)
                ax1.imshow(self.I, cmap=ppl.cm.gray)
                ax1.contour(self.phi, [0], colors='r')

                ax2.imshow(self.phi, cmap='hot')
                ax2.contour(self.phi, [0], colors='r')

                ax3.imshow(self.phi_hat, cmap='hot')
                ax3.contour(self.phi_hat, [0], colors='b')

                # ax4.imshow(self.U, cmap='hot')
                # ax4.contour(self.phi, [-2,0,2], colors='g')


                ppl.pause(0.001)
                if s % 2000 == 0:
                    print(s, 'mu_in:: ', mu_in, ' mu_out::', mu_out)#, 'U in: ', len(in_pt_indices_U[0]), 'U_out: ', len(out_pt_indices_U[0]),' U sum: ', self.U.sum())
                    # x, y = np.meshgrid(np.arange(self.I.shape[1]), np.arange(self.I.shape[0]))
                    # pix = np.vstack((x.flatten(), y.flatten())).T
                    #
                    # def onselect(verts):
                    #     # Select elements in original array bounded by selector path:
                    #     p = Path(verts)
                    #     ind = p.contains_points(pix, radius=1)
                    #     selected = np.zeros_like(self.I)
                    #     selected.flat[ind] = self.I.flat[ind]
                    #     self.umask =  (selected > 0) *0.1
                    #     self.h_d = distance(1 - self.umask)
                    #     d_max = np.amax(self.h_d)
                    #     d_min = np.amin(self.h_d)
                    #     self.h_d = (1/(d_max - d_min))*(d_max*np.ones(self.h_d.shape) - self.h_d)
                    #     self.U = np.add(self.U, self.h_d)
                    #     #self.U = medfilt(self.U,5)
                    #     ppl.close()
                    #
                    # m = LassoSelector(ax1, onselect)
                    # ppl.show()

    def grad_phi(self,mu_in, mu_out,narrow_band):
        #d_T = np.zeros(self.phi.shape)
        d_T = -1*np.subtract(np.square(self.I[narrow_band] - mu_in),np.square(self.I[narrow_band] - mu_out))
        return d_T

    def H(self, phi, in_pt_indices):
        base = np.zeros(phi.shape)
        base[in_pt_indices] = 1
        return base









