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

# k2 = ep.KCSD_wrapper(k)
# CsdElectrodeInputAll = k2.pots_to_csd(np.identity(len(pots)))
# l_max = np.abs(CsdElectrodeInputAll).max()
# levels = np.linspace(-l_max, l_max, 32)
# ele_col, ele_row = np.mgrid[0:10, 0:10]
# CsdElectrodeInput = CsdElectrodeInputAll.reshape(CsdElectrodeInputAll.shape[:-1] + (10, 10))
# fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(8,7), gridspec_kw=dict(wspace=0, hspace=0, left=0, bottom=0, top=1,right=1))
# for i in range(10):
#     for j in range(10):
#         col = ele_col[i, j]
#         row = 9 - ele_row[i, j] # Y increases when row decreases
#         partial = CsdElectrodeInput[:, :, i, j]
#         ax = axes[row, col]
#         ax.set_aspect('equal')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         im = ax.contourf(k2.estm_x, k2.estm_y, partial,
#                          levels=levels, cmap=cm.bwr)
#         ax.scatter(ele_pos[:, 0], ele_pos[:, 1], 1,
#                    c='k', marker='.')
#
# cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=7. / 8)
#
# CsdNoiseSd = np.sqrt((CsdElectrodeInputAll ** 2).sum(axis=-1))
# fig = plt.figure(figsize=(7,7))
# ax = plt.subplot(111)
# ax.set_aspect('equal')
# levels = np.linspace(CsdNoiseSd.min(), CsdNoiseSd.max(), 32)
# im = ax.contourf(k2.estm_x, k2.estm_y, CsdNoiseSd,
#                  levels=levels, cmap=cm.Greys)
# ax.set_xlabel('X (mm)')
# ax.set_ylabel('Y (mm)')
# ax.set_title('CSD noise SD given electrode noise of unitary SD')
# ticks = np.linspace(CsdNoiseSd.min(), CsdNoiseSd.max(), 3, endpoint=True)
# plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)
# ax.scatter(ele_pos[:, 0], ele_pos[:, 1], 10,
#            c='m', marker='.')