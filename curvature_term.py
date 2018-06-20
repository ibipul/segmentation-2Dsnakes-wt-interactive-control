import numpy as np
import math

class meancurvature:

    _eps = 2.2204e-16

    def __init__(self, rows, cols):
        self.nrows = rows
        self.ncols = cols

    def get_mean_curvature(self, narrow_band, narrow_band_point_loc,phi):
        num_pts = len(narrow_band_point_loc)
        K = np.zeros(num_pts)

        for i in range(len(narrow_band_point_loc)):

            pt = narrow_band_point_loc[i]
            nr = pt[0]
            nc = pt[1]
            # Boundary conditions
            if all((nr + 1) >= narrow_band[0]):
                nr = self.nrows - 2
            if (nr - 1) <= 0:
                nr = 2
            if all((nc + 1) >= narrow_band[1]):
                nc = self.ncols - 2
            if (nc - 1) <= 0:
                nc = 2

            # % derivatives
            phi_y = phi[(nr, nc + 1)] - phi[(nr, nc - 1)]
            phi_x = phi[(nr + 1, nc)] - phi[(nr - 1, nc)]
            phi_yy = phi[(nr, nc + 1)] - 2 * phi[(nr, nc)] + phi[(nr, nc - 1)]
            phi_xx = phi[(nr + 1, nc)] - 2 * phi[(nr, nc)] + phi[(nr - 1, nc)]
            phi_xy = 0.25 * (-phi[(nr - 1, nc - 1)] - phi[(nr + 1, nc + 1)] + phi[(nr - 1, nc + 1)] + phi[(nr + 1, nc - 1)])

            # norm
            norm = math.sqrt(phi_x ** 2 + phi_y ** 2)
            # K[i]
            K[i] = ((phi_x ** 2 * phi_yy + phi_y ** 2 * phi_xx - 2 * phi_x * phi_y * phi_xy) /
                    (phi_x ** 2 + phi_y ** 2 + self._eps) ** (3 / 2)) * norm

        return K
