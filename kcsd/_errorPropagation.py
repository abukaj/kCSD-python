import numpy as np

class KCSD_wrapper(object):
    def __init__(self, kcsd):
        self._kcsd = kcsd

    def __getattr__(self, item):
        return getattr(self._kcsd, item)

    @property
    def _pots_to_csd_mtx(self):
        estimation_table = self.k_interp_cross
        k_inv = np.linalg.inv(self.k_pot + self.lambd *
                              np.identity(self.k_pot.shape[0]))
        return np.matmul(estimation_table,
                         k_inv)

    def pots_to_csd(self, pots):
        return self.process_estimate(np.matmul(self._pots_to_csd_mtx,
                                               pots))

    def process_estimate(self, estimation):
        """Function used to rearrange estimation according to dimension, to be
        used by the fuctions values
        Parameters
        ----------
        estimation : np.array
        Returns
        -------
        estimation : np.array
            estimated quantity of shape (ngx, ngy, ngz, *)
        """
        if self.dim == 1:
            return estimation.reshape(self.ngx, -1)

        if self.dim == 2:
            return estimation.reshape(self.ngx, self.ngy, -1)

        if self.dim == 3:
            return estimation.reshape(self.ngx, self.ngy, self.ngz, -1)

        return estimation