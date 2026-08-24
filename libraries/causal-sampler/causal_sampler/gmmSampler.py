#%%
import numpy as np
try:
    from IPython import get_ipython
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        # Jupyter notebook / JupyterLab
        from tqdm.notebook import tqdm
    else:
        # Other interactive shells
        from tqdm import tqdm
except (NameError, ImportError):
    # Ordinary Python shell
    from tqdm import tqdm
import pickle
import os

def log_gaussian_pdf(X, mu, Sigma, cov_type='full', cov_reg=1e-6):
    """
    Compute the log of the Gaussian probability density function.
    
    Parameters:
    X: np.ndarray,
        shape=(N, d) data points
    mu: np.ndarray,
        shape=(d,) mean vector
    Sigma: np.ndarray,
        shape=(d, d) covariance matrix or shape=(d,) for diagonal covariance
    cov_type: str,
        'full', 'diag', or 'spherical', default is 'full'
    cov_reg: float,
        regularization term for covariance matrix, default is 1e-6

    Return: 
    logp: shape=(N,) log-probability density function values
    """
    N, d = X.shape
    diff = X - mu

    if cov_type == 'full':
        if Sigma.shape == ():
            Sigma = np.array([[Sigma]])
        Sigma_reg = Sigma + cov_reg * np.eye(Sigma.shape[0])

        sign, logdet = np.linalg.slogdet(Sigma_reg)
        if sign <= 0:
            print("Warning: Sigma not positive definite.")
            return -np.inf * np.ones(N)

        inv_Sigma = np.linalg.inv(Sigma_reg)
        quadform = np.sum((diff @ inv_Sigma) * diff, axis=1)
        logp = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quadform)

    elif cov_type == 'diag':
        Sigma_safe = np.maximum(Sigma, cov_reg)
        inv_Sigma = 1.0 / Sigma_safe
        quadform = np.sum((diff ** 2) * inv_Sigma, axis=1)
        logdet = np.sum(np.log(Sigma_safe))
        logp = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quadform)

    elif cov_type == 'spherical':
        Sigma = np.asarray(Sigma)
        Sigma_safe = max(Sigma.item(), cov_reg)
        inv_Sigma = 1.0 / Sigma_safe
        quadform = np.sum(diff ** 2, axis=1) * inv_Sigma
        logdet = d * np.log(Sigma_safe)
        logp = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quadform)
    
    else:
        raise ValueError("Parameter cov_type must be 'full', 'diag' or 'spherical'!")
          
    return  logp

def kmeans_plus_plus_init(X, weights, K, random_seed=None):
    """
    K-means++ initialization for GMM.
    
    Parameters:
    X: np.ndarray,
        shape=(N, d) training data
    weights: np.ndarray,
        shape=(N,) non-negative weights for each data point
    K: int, 
        number of components
    random_seed: int, 
        random seed for reproducibility, default is None
    
    Return: 
    centers: shape=(K, d) initial centers
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    N, d = X.shape
    w = weights / np.sum(weights) 
    
    idx0 = np.random.choice(N, p=w)
    centers = [X[idx0].copy()]
    
    if K == 1:
        return np.array(centers)
    
    dist_sq = np.full(N, np.inf)
    
    for center_id in range(1, K):
        current_center = centers[-1] 
        diff = X - current_center
        dists = np.sum(diff*diff, axis=1)  # shape=(N,)
        dist_sq = np.minimum(dist_sq, dists)
        
        weighted_probs = dist_sq * w
        weighted_probs_sum = np.sum(weighted_probs)
        if weighted_probs_sum <= 1e-16:
            idx = np.random.choice(N)
        else:
            weighted_probs /= weighted_probs_sum
            idx = np.random.choice(N, p=weighted_probs)
        
        centers.append(X[idx].copy())
    
    return np.array(centers)

class WeightedGMM:
    """
    Gaussian Mixture Model with weighted data points using the EM algorithm.
    
    Parameters: 
    K: int, 
        number of components
    cov_type: str, 
        'full', 'diag', or 'spherical', default is 'full'
    cov_reg: float, 
        regularization term for covariance matrix, default is 1e-6
    min_variance_value: float, 
        minimum variance value to avoid singular covariance, default is 1e-6
    max_iter: int, 
        maximum number of EM iterations, default is 1000
    tol: float, 
        convergence threshold for log-likelihood change, default is 1e-7
    init_method: str, 
        'random', 'kmeans++', or 'user_assigned', default is 'random'
    user_assigned_mus: np.ndarray,
        shape=(K, d) initial means if init_method is 'user_assigned', default is None
    random_seed: int, 
        random seed for reproducibility, default is None
    
    """
    def __init__(self,
                 K,
                 cov_type='full',
                 cov_reg=1e-6,
                 min_variance_value=1e-6,
                 max_iter=1000,
                 tol=1e-7,
                 init_method='random',
                 user_assigned_mus=None,
                 random_seed=None):
        self.K = K
        self.cov_type = cov_type
        self.cov_reg = cov_reg
        self.min_variance_value = min_variance_value
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.user_assigned_mus = user_assigned_mus
        self.random_seed = random_seed
        
        self.N = None
        self.d = None
        self.w = None
        self.resp = None
        self.pi = None
        self.mus = None
        self.Sigmas = None
        self.prev_loglik = -np.inf
        self.loglik = None
        self.N_eff = None
        self.logliks = []
        self.avg_loglik = None
        self.AIC = None
        self.BIC = None
        
        self.params = {'pi': self.pi,
                       'mus': self.mus,
                       'Sigmas': self.Sigmas,
                       'cov_type': self.cov_type}
        self.scores = {'avg_loglik': self.avg_loglik,
                       'AIC': self.AIC,
                       'BIC': self.BIC}
        self.fitted = False
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def initialize_params(self, X, weights):
        """
        Initialize the GMM parameters.
        
        Parameters:
        X: np.ndarray,
            shape=(N, d) training data
        weights: np.ndarray,
            shape=(N,) non-negative weights for each data point
        
        """
        
        self.N, self.d = X.shape
        w_raw = np.asarray(weights, float)
        self.N_eff = float(np.sum(w_raw))
        if self.N_eff <= 0:
            raise ValueError("All weights are zero or negative.")
        self.w = w_raw / self.N_eff

        self.pi = np.ones(self.K) / self.K
        if self.init_method == 'random':
            idx = np.random.choice(self.N, self.K, replace=False)
            self.mus = X[idx].copy()
        elif self.init_method == 'kmeans++':
            self.mus = kmeans_plus_plus_init(X, self.w, self.K, self.random_seed)
        elif self.init_method == 'user_assigned':
            self.mus = self.user_assigned_mus.copy()
        else:
            raise ValueError("init_method must be 'random', 'kmeans++' or 'user_assigned'!")
        
        # Initialize Sigmas
        overall_cov = np.cov(X.T, aweights=self.w)

        if self.cov_type == 'full':
            if self.d == 1:
                overall_cov = np.array([[float(overall_cov)]], dtype=float)  # (1,1)
            else:
                overall_cov = np.asarray(overall_cov, dtype=float)           # (d,d)
            self.Sigmas = np.repeat(overall_cov[None, :, :], self.K, axis=0)           # (K,d,d)

        elif self.cov_type == 'diag':
            if self.d == 1:
                overall_cov = np.array([float(overall_cov)], dtype=float)    # (1,)
            else:
                overall_cov = np.diag(np.asarray(overall_cov, dtype=float))  # (d,)
            self.Sigmas = np.repeat(overall_cov[None, :], self.K, axis=0)              # (K,d)

        elif self.cov_type == 'spherical':
            if self.d == 1:
                overall_cov = float(overall_cov)                             # scalar
            else:
                overall_cov = float(np.mean(np.diag(np.asarray(overall_cov, dtype=float))))
            self.Sigmas = np.full((self.K,), overall_cov, dtype=float)                 # (K,)
        else:
            raise ValueError("Parameter cov_type must be 'full', 'diag' or 'spherical'!")
    
    def step_e(self, X):
        """
        Step E of the EM algorithm: compute responsibilities and log-likelihood.
        
        Parameters:
        X: np.ndarray,
            shape=(N, d) training data
        
        """

        if self.cov_type == 'full':
            Sigma_reg = self.Sigmas + self.cov_reg * np.eye(self.d)[None, :, :]
            sign, logdet = np.linalg.slogdet(Sigma_reg)          # (K,)
            inv_S = np.linalg.inv(Sigma_reg)                     # (K, d, d)
            diffs = X[None, :, :] - self.mus[:, None, :]         # (K, N, d)
            difvinv = np.einsum('knd,kde->kne', diffs, inv_S)    # (K, N, d)
            quad = np.einsum('knd,knd->kn', difvinv, diffs)      # (K, N)
            const = self.d * np.log(2*np.pi)
            log_pdf = -0.5 * (const + logdet[:, None] + quad)    # (K, N)

        elif self.cov_type == 'diag':
            Sigma_safe = np.maximum(self.Sigmas, self.cov_reg)   # (K, d)
            inv_S = 1.0 / Sigma_safe
            logdet = np.sum(np.log(Sigma_safe), axis=1)          # (K,)
            diffs = X[None, :, :] - self.mus[:, None, :]         # (K, N, d)
            quad = np.einsum('knd,kd->kn', diffs**2, inv_S)      # (K, N)
            const = self.d * np.log(2*np.pi)
            log_pdf = -0.5 * (const + logdet[:, None] + quad)    # (K, N)

        else:  # spherical
            Sigma_safe = np.maximum(self.Sigmas, self.cov_reg)   # (K,)
            inv_S = 1.0 / Sigma_safe
            logdet = self.d * np.log(Sigma_safe)                 # (K,)
            diffs = X[None, :, :] - self.mus[:, None, :]         # (K, N, d)
            quad = np.sum(diffs**2, axis=2) * inv_S[:, None]     # (K, N)
            const = self.d * np.log(2*np.pi)
            log_pdf = -0.5 * (const + logdet[:, None] + quad)    # (K, N)

        log_pdf = log_pdf.T                                      # (N, K)

        pi_safe = np.clip(self.pi, 1e-12, None)
        log_prob = log_pdf + np.log(pi_safe)[None, :]            # (N, K)
        mx = np.max(log_prob, axis=1, keepdims=True)
        lse = mx + np.log(np.exp(log_prob - mx).sum(axis=1, keepdims=True))  # (N,1)
        self.loglik = float(np.sum(self.w[:, None] * lse))
        self.logliks.append(self.loglik)

        # responsibilities
        resp = np.exp(log_prob - mx)
        resp /= resp.sum(axis=1, keepdims=True)            # (N, K)
        self.resp = resp
        
    def step_m(self, X):
        """
        Step M of the EM algorithm: update mixture weights, means, and covariances.
        
        Parameters:
        X: np.ndarray,  
            shape=(N, d) training data
        """
        
        Wk = np.sum(self.w[:, None] * self.resp, axis=0)        # (K,)

        Wk = np.maximum(Wk, 1e-12)
        self.pi = Wk / np.sum(Wk)

        self.mus = (self.resp.T * self.w).dot(X) / Wk[:, None]  # (K, d)

        if self.cov_type == 'full':
            diffs = X[None, :, :] - self.mus[:, None, :]                                # (K, N, d)
            alpha = self.resp.T * self.w                                                # (K, N)
            difw = diffs * alpha[:, :, None]                                            # (K, N, d)
            covs = np.einsum('kni,knj->kij', difw, diffs)                               # (K, d, d)
            covs /= Wk[:, None, None]
            covs += self.cov_reg * np.eye(self.d)[None, :, :]

            eigvals, eigvecs = np.linalg.eigh(covs)                                     # (K, d), (K, d, d)
            eigvals = np.maximum(eigvals, self.min_variance_value)
            self.Sigmas = np.einsum('kij,kj,klj->kil', eigvecs, eigvals, eigvecs)       # (K, d, d)

        elif self.cov_type == 'diag':
            diffs = X[None, :, :] - self.mus[:, None, :]                                # (K, N, d)
            alpha = self.resp.T * self.w                                                # (K, N)
            var_kd = np.sum(alpha[:, :, None] * (diffs**2), axis=1) / Wk[:, None]       # (K, d)
            self.Sigmas = np.maximum(var_kd + self.cov_reg, self.min_variance_value)    # (K, d)

        else:  # spherical
            diffs = X[None, :, :] - self.mus[:, None, :]                                # (K, N, d)
            alpha = self.resp.T * self.w                                                # (K, N)
            var_k = np.sum(alpha[:, :, None] * (diffs**2), axis=(1, 2)) / (Wk * self.d) # (K,)
            self.Sigmas = np.maximum(var_k + self.cov_reg, self.min_variance_value)     # (K,)

    def fit(self, 
            X, 
            weights,
            show_progress_bar=True):
        """
        Fit the GMM to the data using the EM algorithm.
        
        Parameters:
        X: np.ndarray,
            shape=(N, d) training data
        weights: np.ndarray,
            shape=(N,) non-negative weights for each data point
        show_progress_bar: bool, 
            whether to display the progress bar, default is True
        """
        
        self.initialize_params(X, weights)
        pbar = tqdm(range(self.max_iter), desc="EM iter", disable=not show_progress_bar)

        for it in pbar:
            self.step_e(X)
            self.step_m(X)

            # —— Check convergence —— #
            pbar.set_postfix({"loglik diff.": f"{abs(self.loglik - self.prev_loglik):.2e}"})
            if abs(self.loglik - self.prev_loglik) < self.tol:
                break
            self.prev_loglik = self.loglik
        
        self.params = {'pi':        self.pi,
                       'mus':       self.mus,
                       'Sigmas':    self.Sigmas,
                       'cov_type':  self.cov_type}
        self.fitted = True

    def eval_score(self):

        """
        Evaluate the model's score using average log-likelihood, AIC and BIC.
        """

        if self.cov_type == 'full':
            cov_params = self.K * self.d * (self.d + 1) // 2
        elif self.cov_type == 'diag':
            cov_params = self.K * self.d
        else:
            cov_params = self.K
        k_params = (self.K - 1) + self.K * self.d + cov_params

        self.AIC = 2 * k_params - 2 * self.loglik
        self.BIC = np.log(max(self.N_eff, 1.0)) * k_params - 2 * self.loglik
        self.avg_loglik = self.loglik / max(self.N_eff, 1.0)
        
        self.scores = {'avg_loglik':    self.avg_loglik,
                       'AIC':           self.AIC,
                       'BIC':           self.BIC}

    def sample(self, num_samples):
        """
        Generate samples from the fitted GMM.
        
        Parameters:
        num_samples: int, 
            number of samples to generate
        
        Return:
        samples: shape=(num_samples, d) generated samples
        """
        
        if not self.fitted:
            raise RuntimeError("The model must be fitted before sampling.")
        
        samples_list = []

        comp_indices = np.random.choice(self.K, size=num_samples, p=self.pi)

        for k in range(self.K):
            count_k = np.sum(comp_indices == k)
            if count_k == 0:
                continue
            if self.cov_type == 'full':
                samples_k = np.random.multivariate_normal(mean=self.mus[k], cov=self.Sigmas[k], size=count_k)
            elif self.cov_type == 'diag':
                samples_k = np.random.multivariate_normal(mean=self.mus[k], cov=np.diag(self.Sigmas[k]), size=count_k)
            elif self.cov_type == 'spherical':
                samples_k = np.random.multivariate_normal(mean=self.mus[k], cov=np.eye(self.d)*self.Sigmas[k], size=count_k)
            else:
                raise ValueError("cov_type must be 'full', 'diag' or 'spherical'!")
            samples_list.append(samples_k)

        samples = np.vstack(samples_list)
        return samples
    
    def evaluate_density(self, x_query_batch):
        """
        Evaluate the probability density of the fitted GMM at the given query points.
        
        Parameters:
        x_query_batch: np.ndarray,
            shape=(M, d) query points, where M is the number of query points
        
        Return:
        densities: shape=(M, 1) probability density values at the query points
        """
        
        
        if not self.fitted:
            raise RuntimeError("The model must be fitted before evaluating density.")

        X = np.atleast_2d(x_query_batch)
        N, d = X.shape
        p_list = []        
        
        if self.cov_type == 'full':
            # (K, d, d)
            sign, logdet = np.linalg.slogdet(self.Sigmas)      # (K,)
            invS = np.linalg.inv(self.Sigmas)                  # (K, d, d)

            # diffs: (K, N, d)
            diffs = X[None, :, :] - self.mus[:, None, :]
            # diffs @ invS -> (K, N, d)
            difvinv = np.einsum('knd,kde->kne', diffs, invS)
            # quadform: (K, N)
            quad = np.einsum('kne,kne->kn', difvinv, diffs)

            const = d * np.log(2*np.pi)
            # log_pdf_k: (N, K)
            log_pdf = (-0.5 * (const + logdet[:, None] + quad)).T

        elif self.cov_type == 'diag':
            # Sigmas: (K, d)
            invS = 1.0 / self.Sigmas                        # (K, d)
            logdet = np.sum(np.log(self.Sigmas), axis=1)    # (K,)

            diffs = X[None, :, :] - self.mus[:, None, :]        # (K, N, d)
            quad = np.einsum('knd,kd->kn', diffs**2, invS)  # (K, N)

            const = d * np.log(2*np.pi)
            log_pdf = (-0.5 * (const + logdet[:, None] + quad)).T

        else:  # spherical
            invS = 1.0 / self.Sigmas                       # (K,)
            logdet = d * np.log(self.Sigmas)               # (K,)

            diffs = X[None, :, :] - self.mus[:, None, :]       # (K, N, d)
            quad = np.sum(diffs**2, axis=2) * invS[:, None]# (K, N)

            const = d * np.log(2*np.pi)
            log_pdf = (-0.5 * (const + logdet[:, None] + quad)).T

        # log-sum-exp
        log_pi = np.log(self.pi + 1e-16)[None, :]             # (1, K)
        log_resp = log_pdf + log_pi                     # (N, K)
        m = np.max(log_resp, axis=1, keepdims=True)     # (N, 1)
        lse = m + np.log(np.sum(np.exp(log_resp - m), axis=1, keepdims=True) + 1e-16)
        p = np.exp(lse).flatten()                       # (N,)

        p_list.append(p)

        return np.stack(p_list, axis=1)       

class CWGMMs:
    """
    Causally Weighted Gaussian Mixture Models for causal resampling.
    Each weighted GMM corresponds to a specific intervention value and is fitted using weighted data points.
    
    Parameters:
    K: int or List[int], 
        number of components for each GMM
    cov_type: str or List[str], 
        'full', 'diag', or 'spherical' for each GMM
    cov_reg: float or List[float], 
        regularization term for covariance matrix for each GMM
    min_variance_value: float or List[float], 
        minimum variance value to avoid singular covariance for each GMM
    max_iter: int or List[int], 
        maximum number of EM iterations for each GMM
    tol: float or List[float], 
        convergence threshold for log-likelihood change for each GMM
    intv_values: List[float], 
        list of intervention values corresponding to each GMM
    init_method: str or List[str], 
        'random', 'kmeans++', or 'user_assigned' for each GMM
    user_assigned_mus: shape=(K, d) or List[shape=(K, d)], 
        initial means if init_method is 'user_assigned' for each GMM
    random_seed: int, random seed for reproducibility, default is None
    
    """
    
    def __init__(self,
                 K,
                 cov_type,
                 cov_reg,
                 min_variance_value,
                 max_iter,
                 tol,
                 intv_values,
                 init_method = "kmeans++",
                 user_assigned_mus = None,
                 random_seed = None):
        
        def _convert_to_list(param, legal_len, name):
            if not isinstance(param, list):
                param = [param for _ in range(legal_len)]
            if len(param) != legal_len:
                raise ValueError(f"Length of parameter {name} must be equal to length of intv_values!")
            return param
        
        self.model_meta = {}
        self.intv_values = intv_values
        
        legal_len = len(self.intv_values)
        K = _convert_to_list(K, legal_len, "K")
        cov_type = _convert_to_list(cov_type, legal_len, "cov_type")
        cov_reg = _convert_to_list(cov_reg, legal_len, "cov_reg")
        min_variance_value = _convert_to_list(min_variance_value, legal_len, "min_variance_value")
        max_iter = _convert_to_list(max_iter, legal_len, "max_iter")
        tol = _convert_to_list(tol, legal_len, "tol")
        init_method = _convert_to_list(init_method, legal_len, "init_method")
        user_assigned_mus = _convert_to_list(user_assigned_mus, legal_len, "user_assigned_mus")

        for init_method_i in init_method:
            if init_method_i not in ["random", "kmeans++", "user_assigned"]:
                raise ValueError("init_method must be 'random', 'kmeans++' or 'user_assigned'!")
        self.K = K
        self.cov_type = cov_type
        self.cov_reg = cov_reg
        self.min_variance_value = min_variance_value
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.user_assigned_mus = user_assigned_mus
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
                                              
    def fit(self, X, weights, verbose=2):
        """
        Fit a series of weighted GMMs to the data, each corresponding to a specific intervention value.
        
        Parameters:
        X: np.ndarray,
            shape=(N, d) training data
        weights: np.ndarray,
            shape=(N, len(intv_values)) non-negative weights for each data point and intervention value
        verbose: int,
            0 (no progress bar), 
            1 (progress bar for GMM fitting), 
            2 (progress bar for both GMM fitting and intervention loop), 
            default is 2
        """
        if verbose == 2:
            show_gmm_progress_bar = True
            show_intv_progress_bar = True
        elif verbose == 1:
            show_gmm_progress_bar = False
            show_intv_progress_bar = True
        elif verbose == 0:
            show_gmm_progress_bar = False
            show_intv_progress_bar = False
        else:
            raise ValueError("verbose should be 0, 1 or 2.")
        
        N = X.shape[0]
        if weights.shape != (N, len(self.intv_values)):
            raise ValueError(f"Shape of weights must be ({N}, {len(self.intv_values)})!")
        pbar = tqdm(enumerate(self.intv_values), total = len(self.intv_values), desc = "CW-GMMs fitting", disable = not show_intv_progress_bar, unit = "model")
        for i, intv_value in pbar:
            weight_i = weights[:,i] * N
            hyperparams = {"K": self.K[i],
                           "cov_type": self.cov_type[i],
                           "cov_reg": self.cov_reg[i],
                           "min_variance_value": self.min_variance_value[i],
                           "max_iter": self.max_iter[i],
                           "tol": self.tol[i],
                           "init_method": self.init_method[i],
                           "user_assigned_mus": self.user_assigned_mus[i] if self.user_assigned_mus is not None else None
                        }
            base_model = WeightedGMM(random_seed=self.random_seed,
                                     **hyperparams)
            base_model.fit(X = X, 
                           weights = weight_i,
                           show_progress_bar=show_gmm_progress_bar)
            base_model.eval_score()
            self.model_meta[f"model_{i}"] = {"base_model": base_model, 
                                             "intv_value": intv_value, 
                                             "hyperparams": hyperparams}
        pbar.close()
        
    def save(self, f_name, path=None):
        """
        Save the CWGMMs model to a file using pickle.
        
        Parameters:
        f_name: str, 
            file name to save the model (with or without .pkl extension)
        path: str, 
            directory path to save the model (default is current working directory), default is None
        """
        if path is None:
            path = os.getcwd()
        else:
            path = os.path.abspath(path)
        os.makedirs(path, exist_ok=True)

        f_path = os.path.join(path, f_name if f_name.endswith(".pkl") else f_name + ".pkl")
        tmp_path = f_path + ".tmp"

        try:
            # if the file exists, overwrite it atomically

            with open(tmp_path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            if os.path.exists(f_path):
                os.remove(f_path)
                os.replace(tmp_path, f_path)
            else:
                os.rename(tmp_path, f_path)
        except Exception as e:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise e

        print(f"Model saved successfully at {f_path}.")

    def sample(self, n_samples):
        """
        Generate samples from the fitted CWGMMs.
        
        Parameters:
        n_samples: int or List[int], 
            number of samples to generate from each GMM
        
        Return:
        new_samples: shape=(sum(n_samples), d) generated samples
        intv_values: shape=(sum(n_samples), 1) corresponding intervention values for each sample
        """
        
        if len(self.model_meta) == 0:
            raise ValueError("No base models to sample from.")
        if isinstance(n_samples, int):
            n_samples = [n_samples for _ in range(len(self.model_meta))]
        elif len(n_samples) != len(self.model_meta):
            raise ValueError("length of n_samples must be equal to the number of models!")
        new_samples = []
        intv_values = []

        for gmm_meta, n in zip(self.model_meta.values(), n_samples):
            intv_value = gmm_meta["intv_value"]
            base_model = gmm_meta["base_model"]

            samples = base_model.sample(n)
            new_samples.append(samples)
            intv_values.append([intv_value]*n)

        new_samples = np.vstack(new_samples)
        intv_values = np.concatenate(intv_values, axis=0).reshape(-1, 1)
        
        return new_samples, intv_values