import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import scipy.linalg as spl
from tqdm import tqdm
from scipy.optimize import minimize
import scipy.special as spc
import pandas as pd
import seaborn as sns
from scipy import sparse
# import optuna
from functools import partial
from scipy.special import gamma
from scipy.optimize import minimize


class EM:
    def __init__(self, size_filepaths: str, x_filepaths: str, y_filepaths: str = None):
        '''
        Инициализируем класс

        size_filepaths: путь к файлу с реальными размерами частиц
        x_filepaths: путь к файлу с перемещениями по оси x
        y_filepaths: путь к файлу с перемещениями по оси y
        '''
        self.size_filepaths = size_filepaths
        self.x_filepaths = x_filepaths
        self.y_filepaths = y_filepaths
        self.sigmas_all = None
        
        sizes_all = [pd.read_csv(filepath, skiprows=3) for filepath in size_filepaths]
        if y_filepaths is not None:
            sizes_all += sizes_all
    
        x_steps_all = [pd.read_csv(filepath, sep='^(\d+|ID),').drop('Unnamed: 0', axis=1) for filepath in x_filepaths]
        if y_filepaths is not None:
            y_steps_all = [pd.read_csv(filepath,sep='^(\d+|ID),').drop('Unnamed: 0', axis=1)\
                .rename({'"Y steps, mkm"':'"X steps, mkm"'}, axis=1) for filepath in y_filepaths]
        else:
            y_steps_all = []

        x_all = x_steps_all + y_steps_all
        
        t0_all = [pd.read_csv(f'{i}-sizes.csv', nrows=1)['t0, c'].values[0] for i in range(1, 21)]
        if y_filepaths is not None:
            t0_all += t0_all
        
        t_all = [pd.read_csv(f'{i}-sizes.csv', nrows=1)['t, c'].values[0] for i in range(1, 21)]
        if y_filepaths is not None:
            t_all += t_all
        
        self.sizes_all = sizes_all
        self.x_all = x_all 
        self.t0_all = t0_all 
        self.t_all = t_all
        self.coef = 1

    def estimate_sigmas(self):
        sigmas_est_all = []
        self.sigmas_hist_all = []
        self.sigma1_all = []
        self.sigma1_hist_all = []
        self.sigmas_all = []
        self.sigma1_all = []


        for i in tqdm(range(len(self.x_all))):
            n = self.x_all[i].shape[0]
            t = self.t_all[i]
            t0 = self.t0_all[i]

            sizes = self.sizes_all[i]
            x_steps = self.x_all[i]
            x_steps['"X steps, mkm"'] = x_steps['"X steps, mkm"'].apply(lambda x: np.array(list(map(float, x.split(',')))))
            
            # x_vecs = x_steps['"X steps, mkm"'].to_list()
            x_vecs = list(x_steps['"X steps, mkm"'])
            B_matxs = []
            C_matxs = []

            for j in range(n):
                k = x_vecs[j].shape[0]
                C_matxs.append(EM.matC(k, t, t0))
                B_matxs.append(EM.matB(k))

            self.C_matxs = C_matxs
            self.B_matxs = B_matxs
            self.sigmas_est = [1] * n
            self.sigma1_est = 1

            sigmas_hist, sigma1_hist = self.emEstimate_multi(x_vecs)
            sigmas_em = [sigmas_hist[-1][i] for i in range(n)]
            self.sigmas_all.append(sigmas_em)
            self.sigma1_all.append(sigma1_hist[-1])

            self.sigmas_hist_all.append(sigmas_hist)
            self.sigma1_hist_all.append(sigma1_hist)

    def estimate_gamma_params(self):
        if not self.sigmas_all:
            self.estimate_sigmas()
        est_all = [sigma for sigmas in self.sigmas_all for sigma in sigmas]
        clipped_sigmas = sorted(est_all)[int(0.05*len(est_all)):int(0.95*len(est_all))]
        self.a, self.b = EM.est_params(1 / np.array(clipped_sigmas))
        return self.a, self.b

    @staticmethod    
    def matTridig(a, b, c, k1=-1, k2=0, k3=1):
        return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
    
    @staticmethod
    def matC(n, t, t0):
        a = np.array([t + 2 * t0 / 3 for _ in range(n)])
        b = np.array([t0 / 6 for _ in range(n - 1)])
        return EM.matTridig(b, a, b)

    @staticmethod
    def matB(n):
        a = np.array([2 for _ in range(n)])
        b = np.array([-1 for _ in range(n - 1)])
        return EM.matTridig(b, a, b)
    
    @staticmethod
    def loglike(X_vecs, V_vecs, sigmas, sigma0, B_matxs, C_matxs):
        res = 0
        n = len(sigmas)
        for i in range(n):
            res -= np.log(np.linalg.det(sigmas[i] * C_matxs[i])) + np.log(np.linalg.det(sigma0 * B_matxs[i])) +\
            (X_vecs[i] - V_vecs[i]).T @ np.linalg.inv(sigmas[i] * C_matxs[i]) @ (X_vecs[i] - V_vecs[i]) + \
            V_vecs[i].T @ np.linalg.inv(sigma0 * B_matxs[i]) @ V_vecs[i]
        return res/2
    
    @staticmethod
    def generate_vecs(n, sigma, sigma1, t, t0):
        c = EM.matC(n, t, t0)
        b = EM.matB(n)
        sigma_y = sigma * c
        sigma_v = sigma1 * b
        y = sps.multivariate_normal.rvs(np.zeros(n), cov=(sigma_y))
        v = sps.multivariate_normal.rvs(np.zeros(n), cov=(sigma_v))
        x = y + v
        return {
            'x': x,
            'y': y,
            'v': v,
            'C': c,
            'B': b
        }
    
    @staticmethod
    def condMean(x, sigma_v, sigma_y):
        return sigma_v @ np.linalg.inv(sigma_v + sigma_y) @ x
    
    @staticmethod
    def condCov(sigma_v, sigma_y):
        return sigma_v - sigma_v @ np.linalg.inv(sigma_y + sigma_v) @ sigma_v
    
    @staticmethod
    def mean_estimate(x_arr, t, t0, n):
        cov = 0
        for x in x_arr:
            for i in range(len(x) - 1):
                cov += x[i] * x[i + 1]
            cov *= 2/(len(x) - 1)
        cov /= len(n)
        mean = np.sum([np.sum(x**2) for x in x_arr])/np.sum(n)
        sigma = (mean + cov) / (t + t0)
        sigma1 = t0 / 6 * sigma - cov
        return sigma, sigma1
    
    @staticmethod
    def emEstimate(x_arr, b_arr, c_arr, n, sigma_est, sigma1_est, tolerance=1e-3, max_iter=50):
        sigma_hist, sigma1_hist = [sigma_est], [sigma1_est]
        iter = 1
        while True:
            tmp1 = 0
            tmp2 = 0
            for i, x in enumerate(x_arr):
                sigma_y = sigma_est * c_arr[i]
                sigma_v = sigma1_est * b_arr[i]
                mean = EM.condMean(x, sigma_v, sigma_y)
                sigma_hat = spl.cholesky(EM.condCov(sigma_v, sigma_y), lower=True)
                tmp1 += np.trace(sigma_hat.T @ np.linalg.inv(c_arr[i]) @ sigma_hat) +\
                                (mean - x).T @ np.linalg.inv(c_arr[i]) @ (mean - x)
                tmp2 += np.trace(sigma_hat.T @ np.linalg.inv(b_arr[i]) @ sigma_hat) +\
                                mean.T @ np.linalg.inv(b_arr[i]) @ mean
            sigma_est = tmp1 / sum(n)
            sigma1_est = tmp2 / sum(n)
            sigma_hist.append(sigma_est)
            sigma1_hist.append(sigma1_est)
            iter += 1
            
            if max(abs(sigma_est - sigma_hist[-2]),\
                        abs(sigma_est - sigma_hist[-2])) < tolerance \
                            or iter > max_iter:
                break
        return [np.array(sigma_hist), np.array(sigma1_hist)]
    
    @staticmethod
    def gen_vecs_multi(sigmas, sigma1, t=None, t0=None, k=None):
        n = len(sigmas)

        if t is None:
            t = np.random.random(n)
        if t0 is None:
            t0 = np.random.random(n)
        if k is None:
            k = np.random.randint(low=100, high=200, size=n)

        X_vecs = []
        Y_vecs = []
        V_vecs = []
        C_matxs = []
        B_matxs = []

        for i in range(n):
            C_matxs.append(EM.matC(k[i], t[i], t0[i]))
            B_matxs.append(EM.matB(k[i]))
            Y_vecs.append(sps.multivariate_normal.rvs(np.zeros(k[i]), sigmas[i] * C_matxs[-1]))
            V_vecs.append(sps.multivariate_normal.rvs(np.zeros(k[i]), sigma1 * B_matxs[-1]))
            X_vecs.append(Y_vecs[-1] + V_vecs[-1])
        
        generated = {
            'X_vecs': X_vecs,
            'Y_vecs': Y_vecs,
            'V_vecs': V_vecs,
            'C_matxs': C_matxs,
            'B_matxs': B_matxs
        }

        return generated
    
    @staticmethod
    def sparse_cholesky(A):
        sparse_matrix = A.T @ A
        sparse_matrix += 1e-6 * sparse.identity(sparse_matrix.shape[0]) # force the sparse matrix is positive definite
        n = sparse_matrix.shape[0]
        LU = sparse.linalg.splu(sparse_matrix, diag_pivot_thresh = 0.0, permc_spec = "NATURAL") # sparse LU decomposition

        L = LU.L @ sparse.diags(LU.U.diagonal()**0.5)
        
        return L # return L (lower triangular metrix)

    def emEstimate_multi(self, X_vecs, tolerance=1e-3, max_iter=200):
        sigmas_hist, sigma1_hist = [np.array(self.sigmas_est)], [self.sigma1_est]
        sigmas_curr = np.copy(self.sigmas_est)
        sigma1_curr = self.sigma1_est
        n = len(X_vecs)
        k = [x.shape[0] for x in X_vecs]
        sum_k = np.sum(k)
        sigmas_new = np.zeros(n)

        for _ in range(max_iter):
            tmp = 0

            for i, X_vec in enumerate(X_vecs):
                sigma_y = sigmas_curr[i] * self.C_matxs[i]
                sigma_v = sigma1_curr * self.B_matxs[i]
                # print(type(sigma_y), type(sigma_v))
                mean = self.condMean(X_vec, sigma_v, sigma_y)
                try:
                    sigma_hat = spl.cholesky(EM.condCov(sigma_v, sigma_y), lower=True)
                    # sigma_hat = EM.sparse_cholesky(EM.condCov(sigma_v, sigma_y))
                except:
                    return sigmas_hist, sigma1_hist
                inv_C = np.linalg.inv(self.C_matxs[i])
                inv_B = np.linalg.inv(self.B_matxs[i])
                trace1 = (sigma_hat.T @ inv_C @ sigma_hat).trace()
                prod1 = (mean - X_vec).T @ inv_C @ (mean - X_vec)
                trace2 = (sigma_hat.T @ inv_B @ sigma_hat).trace()
                prod2 = mean.T @ inv_B @ mean
                sigmas_new[i] = 1 / k[i] * (trace1 + prod1)
                tmp += trace2 + prod2
            
            sigma1_new = tmp / sum_k
            diff = np.sum([(sigmas_new[i] - sigmas_curr[i])**2 for i in range(n)]) + (sigma1_new - sigma1_curr)**2

            sigmas_curr = np.copy(sigmas_new)
            sigma1_curr = sigma1_new
            sigmas_hist.append(sigmas_curr)
            sigma1_hist.append(sigma1_curr)

            if diff < tolerance:
                break
        
        return sigmas_hist, sigma1_hist
    
    @staticmethod
    def extra_push(x_arr, b_arr, c_arr, n, sigma_hist, sigma1_hist, push=0.1, tolerance=1e-3, max_iter=200):
        eps1 = np.sign(sigma_hist[-1] - sigma_hist[-2]) * push
        eps2 = np.sign(sigma1_hist[-1] - sigma1_hist[-2]) * push

        sigma_old, sigma1_old = sigma_hist[-1], sigma1_hist[-1]
        sigma_new = sigma_old + eps1
        sigma1_new = sigma1_old + eps2 
        iter = 0

        while np.sum(eps1**2 + eps2**2) > 1e-3 and iter < 20:
            sigmas = EM.emEstimate_multi(x_arr, max_iter=1)
            
            sigma_step, sigma1_step = sigmas[0][-1], sigmas[1][-1]

            if (np.sign(sigma_step - sigma_new) != np.sign(eps1)).any() or \
            (np.sign(sigma1_step - sigma1_new) != np.sign(eps2)).any():
                eps1 = -eps1 / 2
                eps2 = -eps2 / 2

            sigma_old, sigma1_old = sigma_step, sigma1_step
            sigma_new, sigma1_new = sigma_old + eps1, sigma1_old + eps2
            iter+= 1

        return sigma_new, sigma1_new   


    def plot_history(self, sigmas_true, sigma0_true):
        n = len(sigmas_true)
        hist_len = len(self.sigmas_hist)

        for i in range(n):
            fig, ax = plt.subplots(figsize=(15, 6))
            sigma = [self.sigmas_hist[j][i] for j in range(len(self.sigmas_hist))]
            ax.scatter(np.arange(0, hist_len), sigma)
            ax.plot([0, hist_len], (sigmas_true[i], sigmas_true[i]), c='r')
            ax.set_title(f'$\sigma_{i + 1}^2$ history')
        
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.scatter(np.arange(0, hist_len), self.sigma1_hist)
        ax.plot([0, hist_len], (sigma0_true, sigma0_true), c='r')
        ax.set_title(f'$\sigma_{0}^2$ history') 

    @staticmethod
    def log_like_gamma(a, x):
        n = len(x)
        return -(-n * np.log(spc.gamma(a)) + (a - 1) * np.sum(np.log(x)) - n * a - n * a * np.log(np.mean(x) / a))
    
    def est_params(sigmas):
        my_gamma = partial(EM.log_like_gamma, x=sigmas)
        a_est = minimize(my_gamma, (np.mean(sigmas))**2/np.var(sigmas))['x'][0]
        b_est = np.mean(sigmas) / a_est
        return a_est, b_est
    
