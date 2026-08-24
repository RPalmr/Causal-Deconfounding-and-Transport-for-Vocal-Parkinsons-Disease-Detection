#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import norm
import causal_sampler.gmmSampler as gmms
import causal_sampler.utils as utils
from causalbootstrapping import backend as be
from causalbootstrapping.utils import weight_func_parse
from causalbootstrapping.expr_extend import weightExpr
from grapl.eqn import Eqn
import grapl.dsl as dsl
import warnings
from typing import Union, Optional, Callable
try:
    from IPython import get_ipython
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        # Jupyter notebook / JupyterLab
        from tqdm.notebook import tqdm
    else:
        # IPython Terminal, etc.
        from tqdm import tqdm
except (NameError, ImportError):
    # Standard Python interpreter
    from tqdm import tqdm
#%%

def compute_causal_weight(causal_graph: str,
                          cause_var_name: str, 
                          effect_var_name: str, 
                          intv_values: Union[list, np.ndarray],
                          data_dict: dict, 
                          id_formular: Optional[Union[weightExpr, Eqn]] = None,
                          est_method: str = "histogram",
                          kernel: Optional[Union[str, Callable]] = "Kronecker delta",
                          **kwargs):
    """
    Compute the causal weight for a given intervention.
    
    Parameters:
    causal_graph: str
        The causal graph in the form of a string.
    cause_var_name: str
        The name of the cause variable.
    effect_var_name: str
        The name of the effect variable.
    intv_values: list or np.ndarray of shape (n_intv,)
        The intervention values.
    data_dict: dict
        A dictionary containing the data. The keys are the variable names and the values are the corresponding data arrays.
    id_formular: Optional[Union[weightExpr, Eqn]], Default: None
        The identification formula for the causal effect. If None, it will be computed from the causal graph.
    est_method: str, Default: "histogram"
        The method for estimating the distribution. Options are "histogram", "kde", "gmm", "multinorm", "kmeans".
    kernel: str or Callable or None, Default: "Kronecker delta"
        The kernel function for the intervention. Options are "Kronecker delta", "Gaussian", or a custom callable function.
        If "Gaussian" is selected, the kernel width should be specified in kwargs, otherwise it will default to 0.1.
        If Callable is provided, it should be a function that takes two arguments: (intv_value, cause_value) and returns the kernel value.
        If None, no kernel is used.
    **kwargs: additional arguments for the distribution estimation and kernel function.
        For est_method:
            = "histogram": n_bins (Dict[str, int], default: None)
            = "kde": bandwidth (float or str, default: scott)
            = "gmm": n_components (int, default: 10)
            = "kmeans": k (int, default: 10)
            = "multinorm": None
        For kernel:
            = "Gaussian": kernel_width (float, default: 0.1)
            = "Kronecker delta": None
    
    Returns:
    causal_weights: np.ndarray of shape (n_samples, n_intv)
        The causal weights for each sample and each intervention value.
    """
    
    if kernel not in ["Kronecker delta", "Gaussian", None]:
        if not callable(kernel):
            raise ValueError("kernel should be None, 'Kronecker delta' or 'Gaussian'.")
    if est_method not in ["histogram", "kde", "gmm", "multinorm", "kmeans"]:
        raise ValueError("est_method should be 'histogram', 'kde', 'gmm', 'multinorm' or 'kmeans'.")

    if id_formular is None:
        grapl_obj = dsl.GraplDSL()
        G = grapl_obj.readgrapl(causal_graph)
        id_formular, identifiable = be.id(Y = set(effect_var_name),
                                          X = set(cause_var_name),
                                          G = G)
    w_nom, w_denom, kernel_flag, valid_id_eqn = weight_func_parse(id_formular)
    if not valid_id_eqn:
        raise ValueError("The given identification equation is not valid to derive valid causal weight expression.")
    weight_func_expr = weightExpr(
    w_nom=w_nom,
    w_denom=w_denom,
    cause_var=cause_var_name,
    kernel_flag=kernel_flag
    )

    dist_map = utils.build_dist_map(weight_func_expr = weight_func_expr,
                                    data = data_dict,
                                    cause_var_name = cause_var_name,
                                    est_method = est_method,
                                    **kwargs)

    kernel_flag = weight_func_expr.kernel_flag
    intv_var_name = "intv_" + cause_var_name
    if kernel_flag:
        if kernel is None:
            raise ValueError("kernel needed for the current causal graph.")
        elif kernel == "Kronecker delta":
            _kernel_intv = eval(f"lambda {intv_var_name},{cause_var_name}: np.equal({intv_var_name},{cause_var_name})",
                    {"np": np})
        elif kernel == "Gaussian":
            kernel_width = kwargs.get("kernel_width", 0.1)
            _kernel_intv = eval(f"lambda {intv_var_name},{cause_var_name}: norm.pdf({intv_var_name}-{cause_var_name}, scale={kernel_width})",
                                {"norm": norm})
        elif callable(kernel):
            _kernel_intv = kernel
    else:
        _kernel_intv = None
    N = list(data_dict.values())[0].shape[0]
    w_func, _ = be.build_weight_function(intv_prob = id_formular,
                                         dist_map = dist_map, 
                                         N = N, 
                                         kernel = _kernel_intv,
                                         cause_intv_name_map = {cause_var_name: intv_var_name})

    causal_weights = np.zeros((N, len(intv_values)))
    for i, intv_value in enumerate(intv_values):
        causal_weights_i = be.weight_compute(w_func = w_func,
                                             data = data_dict,
                                             intv_dict = {intv_var_name: intv_value})
        causal_weights[:,i] = causal_weights_i.reshape(-1)
    return causal_weights

class CausalBootstrapSampler:
    """
    This class performs causal bootstrapping resampling.
    
    Parameters:
    causal_graph: str
        The causal graph in the form of a string.
    cause_var_name: str
        The name of the cause variable.
    effect_var_name: str
        The name of the effect variable.
    intv_values: list or np.ndarray of shape (n_intv,)
        The intervention values.
    data_dict: dict
        A dictionary containing the data. The keys are the variable names and the values are the corresponding data arrays.
    id_formular: expr_extend.DOExpr or eqn.Eqn or None, Default: None
        The identification formula for the causal effect. If None, it will be computed from the causal graph.
    est_method: str, Default: "histogram"
        The method for estimating the distribution. Options are "histogram", "kde", "gmm", "multinorm", "kmeans".
    **est_kwargs: additional arguments for the distribution estimation.
        For est_method:
            = "histogram": n_bins (Dict[str, int], default: None)
            = "kde": bandwidth (float or str, default: scott)
            = "gmm": n_components (int, default: 10)
            = "kmeans": k (int, default: 10)
            = "multinorm": None
    """
    def __init__(self,
                 causal_graph: str,
                 cause_var_name: str,
                 effect_var_name: str,
                 intv_values: Union[list, np.ndarray],
                 data_dict: dict,
                 id_formular: Optional[Union[weightExpr, Eqn]] = None,
                 est_method: str = "histogram",
                 **est_kwargs
                 ):
        self.causal_graph = causal_graph
        self.cause_var_name = cause_var_name
        self.effect_var_name = effect_var_name
        self.intv_var_name = "intv_" + cause_var_name
        
        self.data_dict = data_dict
        self.intv_values = intv_values
        self.id_formular = id_formular
        if id_formular is not None:
            warnings.warn("Using provided identification equation. Ensure it is valid and compatible. Parameters causal_graph, effect_var_name, cause_var_name are ignored.")
        self.est_method = est_method
        self.est_kwargs = est_kwargs

    def fit(self,
            kernel: Optional[Union[str, Callable]] = "Kronecker delta",
            random_seed: Optional[int] = None,
            **kernel_kwargs):
        """
        This function computes the causal weights for the causal bootstrapping.
        
        Parameters:
        kernel: str or Callable or None, Default: "Kronecker delta"
            The kernel function for the intervention. Options are "Kronecker delta", "Gaussian", or a custom callable function.
            If "Gaussian" is selected, the kernel width should be specified in kwargs, otherwise it will default to 0.1.
            If Callable is provided, it should be a function that takes two arguments: (intv_value, cause_value) and returns the kernel value.
            If None, no kernel is used.
        random_seed: int, Default: None
            The random seed for the computation.
        **kernel_kwargs: additional arguments for the kernel function.
            For "Gaussian": kernel_width (float, default: 0.1)
            For "Kronecker delta": None
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        if self.id_formular is None:
            grapl_obj = dsl.GraplDSL()
            G = grapl_obj.readgrapl(self.causal_graph)
            id_formular, identifiable = be.id(Y = set(self.effect_var_name),
                                              X = set(self.cause_var_name),
                                              G = G)
            self.id_formular = id_formular

        w_nom, w_denom, kernel_flag, valid_id_eqn = weight_func_parse(self.id_formular)
        if not valid_id_eqn:
            raise ValueError("The given identification equation is not valid to derive valid causal weight expression.")
        
        kwargs = self.est_kwargs.copy()
        kwargs.update(kernel_kwargs)
        self.causal_weights_mat = compute_causal_weight(causal_graph = self.causal_graph,
                                                        cause_var_name = self.cause_var_name,
                                                        effect_var_name = self.effect_var_name,
                                                        intv_values = self.intv_values,
                                                        data_dict = self.data_dict,
                                                        id_formular = self.id_formular,
                                                        est_method = self.est_method,
                                                        kernel = kernel,
                                                        **kwargs) 
    
    def resample(self,
                 n_samples : Union[int, list],
                 sampling_mode: str = "fast",
                 shuffle: bool = True,
                 return_samples: bool = False, 
                 random_seed: Optional[int] = None, 
                 verbose: int = 1):
        """
        This function performs the causal bootstrapping resampling.
        
        Parameters:
        n_samples: int or list of int
            The number of samples to be generated. If a list, it should have the same length as intv_values.
        sampling_mode: str, Default: "fast"
            The sampling mode for the causal bootstrapping. Options are "fast" and "exact".
        shuffle: bool, Default: True
            Whether to shuffle the generated samples.
        return_samples: bool, Default: False
            Whether to return the generated samples, or keep the samples in the class attributes.
        random_seed: int, Default: None
            The random seed for the resampling.
        verbose: int, Default: 1
            The verbosity level for the resampling. 0: no progress bar, 1: show intv progress bar.
            
        Returns:
        deconf_X: np.ndarray
            The generated samples of X so as the deconfounded X.
        deconf_Y: np.ndarray
            The corresponding interventional values so as the deconfounded Y.
        """
        if isinstance(n_samples, int):
            n_samples = [n_samples for i in range(len(self.intv_values))]
        if verbose == 1:
            show_intv_progress_bar = True
        elif verbose == 0:
            show_intv_progress_bar = False
        else:
            raise ValueError("verbose should be 0 or 1.")
        if random_seed is not None:
            np.random.seed(random_seed)
        
        cb_data = {}
        pbar = tqdm(enumerate(self.intv_values), total = len(self.intv_values), desc = "CB Resampling", disable = not show_intv_progress_bar)
        for i, intv_value in pbar:
            intv_i_dict = {self.intv_var_name: intv_value}
            cb_data_i = be.cw_bootstrapper(data = self.data_dict, 
                                           weights = self.causal_weights_mat[:,i].reshape(-1), 
                                           intv_dict = intv_i_dict, 
                                           n_sample = n_samples[i], 
                                           sampling_mode = sampling_mode, 
                                           random_state = random_seed)
            for key in cb_data_i:
                if i == 0:
                    cb_data[key] = cb_data_i[key]
                else:
                    cb_data[key] = np.vstack((cb_data[key], cb_data_i[key]))
        self.deconf_X = cb_data[self.effect_var_name]
        self.deconf_Y = cb_data[self.intv_var_name].reshape(-1,1)
        
        sample_idx = np.arange(self.deconf_X.shape[0])
        if shuffle:
            np.random.shuffle(sample_idx)
        self.deconf_X = self.deconf_X[sample_idx]
        self.deconf_Y = self.deconf_Y[sample_idx]
        if return_samples:
            return self.deconf_X, self.deconf_Y
        
    def fit_deconf_model(self, ml_model: object):
        """
        This function fits the deconfounded model.
        
        Parameters:
        ml_model: object
            A classifier or regressor object. It should have the fit method.
 
        Returns:
        ml_model: object
            The fitted machine learning model.
        """
        
        self.deconf_model = ml_model.fit(self.deconf_X, self.deconf_Y.ravel())
        return self.deconf_model

class CausalGMMSampler:
    """
    This class performs causal GMM resampling.
    
    Parameters:
    causal_graph: str
        The causal graph in the form of a string.
    cause_var_name: str
        The name of the cause variable.
    effect_var_name: str
        The name of the effect variable.
    intv_values: list or np.ndarray of shape (n_intv,)
        The intervention values.
    data_dict: dict
        A dictionary containing the data. The keys are the variable names and the values are the corresponding data arrays.
    id_formular: expr_extend.DOExpr or eqn.Eqn or None, Default: None
        The identification formula for the causal effect. If None, it will be computed from the causal graph.
    est_method: str, Default: "histogram"
        The method for estimating the distribution. Options are "histogram", "kde", "gmm", "multinorm", "kmeans".
    **est_kwargs: additional arguments for the distribution estimation.
        For est_method:
            = "histogram": n_bins (Dict[str, int], default: None)
            = "kde": bandwidth (float or str, default: scott)
            = "gmm": n_components (int, default: 10)
            = "kmeans": k (int, default: 10)
            = "multinorm": None
    """
    def __init__(self, 
                 causal_graph: str,
                 cause_var_name: str, 
                 effect_var_name: str, 
                 intv_values: Union[list, np.ndarray],
                 data_dict: dict, 
                 id_formular: Optional[Union[weightExpr, Eqn]] = None,
                 est_method: str = "histogram",
                 **est_kwargs):
        self.data_dict = data_dict
        self.cause_var_name = cause_var_name
        self.effect_var_name = effect_var_name
        self.intv_var_name = "intv_" + cause_var_name
        
        self.intv_values = intv_values
        self.causal_graph = causal_graph
        if id_formular is not None:
            warnings.warn("Using provided identification equation. Ensure it is valid and compatible. Parameters causal_graph, effect_var_name, cause_var_name are ignored.")
        self.id_formular = id_formular
        self.est_method = est_method
        self.est_kwargs = est_kwargs
        
        N = list(self.data_dict.values())[0].shape[0]
        self.deconf_X = np.zeros((data_dict[effect_var_name].shape[1],N))
        self.deconf_Y = np.zeros((1,N))
        
        self.causal_weights_mat = None
        self.cwgmm_model = None
        self.deconf_model = None
        
    def fit(self, 
            comp_k: Union[int, list],
            max_iter: Union[int, list] = 1000, 
            tol: Union[float, list] = 1e-4, 
            init_method: Union[str, list] = "kmeans++", 
            cov_type: Union[str, list] = "full", 
            cov_reg: Union[float, list] = 1e-6, 
            min_variance_value: Union[float, list] = 1e-6, 
            random_seed: Optional[int] = None, 
            kernel: Optional[Union[str, Callable]] = "Kronecker delta",
            verbose: int = 2, 
            return_model: bool = False,
            **kernel_kwargs):
        """
        This function performs the CW-GMM resampling.
        
        Parameters:
        comp_k: int or list of int
            The number of components for the GMM. If a list, it should have the same length as intv_values.
        max_iter: int or list of int, Default: 1000
            The maximum number of iterations for the GMM fitting. If a list, it should have the same length as intv_values.
        tol: float or list of float, Default: 1e-4
                The tolerance for the GMM fitting. If a list, it should have the same length as intv_values.
        init_method: str or list of str, Default: "kmeans++"
            The initialization method for the GMM fitting. If a list, it should have the same length as intv_values.
        cov_type: str or list of str, Default: "full"
            The covariance type for the GMM fitting. If a list, it should have the same length as intv_values.
        cov_reg: float or list of float, Default: 1e-6
            The covariance regularization for the GMM fitting. If a list, it should have the same length as intv_values.
        min_variance_value: float or list of float, Default: 1e-6
                The minimum variance value for the GMM fitting. If a list, it should have the same length as intv_values.
        random_seed: int, Default: None
            The random seed for the GMM fitting.
        kernel: str or Callable or None, Default: "Kronecker delta"
                The kernel function for the intervention. Options are "Kronecker delta", "Gaussian", or a custom callable function.
                If "Gaussian" is selected, the kernel width should be specified in kwargs, otherwise it will default to 0.1.
                If Callable is provided, it should be a function that takes two arguments: (intv_value, cause_value) and returns the kernel value.
                If None, no kernel is used.
        verbose: int, Default: 2
            The verbosity level for the GMM fitting. 0: no progress bar, 1: show GMM progress bar, 2: show both GMM and intv progress bar.
        return_model: bool, Default: False
            Whether to return the fitted GMM model.
        **kernel_kwargs: additional arguments for the kernel function.
            For "Gaussian": kernel_width (float, default: 0.1)
            For "Kronecker delta": None
        
        Returns:
        cwgmms: object
            The fitted GMM model.
        """

        self.cwgmm_model = gmms.CWGMMs(K = comp_k,
                                       cov_type = cov_type,
                                       cov_reg = cov_reg,
                                       min_variance_value = min_variance_value,
                                       max_iter = max_iter,
                                       tol = tol,
                                       intv_values = self.intv_values,
                                       init_method = init_method,
                                       random_seed = random_seed)
        
        if self.id_formular is None:
            grapl_obj = dsl.GraplDSL()
            G = grapl_obj.readgrapl(self.causal_graph)
            id_formular, identifiable = be.id(Y = set(self.effect_var_name),
                                              X = set(self.cause_var_name),
                                              G = G)
            self.id_formular = id_formular
        w_nom, w_denom, kernel_flag, valid_id_eqn = weight_func_parse(self.id_formular)
        if not valid_id_eqn:
            raise ValueError("The given identification equation is not valid to derive valid causal weight expression.")
        
        kwargs = self.est_kwargs.copy()
        kwargs.update(kernel_kwargs)        
        self.causal_weights_mat = compute_causal_weight(causal_graph = self.causal_graph,
                                                        cause_var_name = self.cause_var_name,
                                                        effect_var_name = self.effect_var_name,
                                                        intv_values = self.intv_values,
                                                        data_dict = self.data_dict,
                                                        id_formular = self.id_formular,
                                                        est_method = self.est_method,
                                                        kernel = kernel,
                                                        **kwargs)
        
        self.cwgmm_model.fit(X = self.data_dict[self.effect_var_name],
                             weights = self.causal_weights_mat,
                             verbose = verbose)
        if return_model:
            return self.cwgmm_model
        
    def resample(self, 
                 n_samples: Union[int, list],
                 shuffle: bool = True, 
                 return_samples: bool = False, 
                 random_seed: Optional[int] = None):
        """
        This function performs the CW-GMM resampling.
        Parameters:
        n_samples: int or list of int
            The number of samples to be generated. If a list, it should have the same length as intv_values.
        shuffle: bool, Default: True
            Whether to shuffle the generated samples.
        return_samples: bool, Default: False
            Whether to return the generated samples, or keep the samples in the class attributes.
        random_seed: Optional[int], Default: None
            The random seed for the resampling.
            
        Returns:
        deconf_X: np.ndarray
            The generated samples of X so as the deconfounded X.
        deconf_Y: np.ndarray
            The corresponding interventional values so as the deconfounded Y.
        """
        if self.cwgmm_model is None:
            raise ValueError("The CW-GMM model is not fitted yet. Please call the fit method first.")
        if random_seed is not None:
            np.random.seed(random_seed)
        
        deconf_X, deconf_Y = self.cwgmm_model.sample(n_samples)
        sample_idx = np.arange(deconf_X.shape[0])
        if shuffle:
            np.random.shuffle(sample_idx)
        self.deconf_X = deconf_X[sample_idx]
        self.deconf_Y = deconf_Y[sample_idx]
        if return_samples:
            return self.deconf_X, self.deconf_Y
    
    def fit_deconf_model(self, ml_model: object):
        """
        This function fits the deconfounded model.
        
        Parameters:
        ml_model: object
            A classifier or regressor object. It should have the fit method.
 
        Returns:
        ml_model: object
            The fitted machine learning model.
        """
        
        self.deconf_model = ml_model.fit(self.deconf_X, self.deconf_Y.ravel())
        return self.deconf_model