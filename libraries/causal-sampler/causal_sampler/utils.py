from typing import Dict, List, Callable, Any
import numpy as np
from causalbootstrapping.distEst_lib import MultivarContiDistributionEstimator

_PREFIX = "intv_"
_POSTFIX = "'"

def make_base_vars(name: str) -> str:
    """
    Make base variable name by removing intervention prefix and prime postfix.
    
    Parameters:
    name: str, 
        variable name with possible intervention prefix and prime postfixs
    
    Returns:
    str, base variable name without intervention prefix and prime postfix
    """
    s = name.strip()
    if s.lower().startswith(_PREFIX):
        s = s[len(_PREFIX):]
    while s.endswith(_POSTFIX):
        s = s[:-len(_POSTFIX)]
    return s

def concat_data(data: Dict[str, np.ndarray], 
                base_var_names: List[str]) -> np.ndarray:
    """
    Concatenate data arrays for the specified base variable names.
    
    Parameters:
    data: Dict[str, np.ndarray], 
        dictionary mapping variable names to their data arrays
    base_var_names: List[str], 
        list of base variable names to concatenate data for
    
    Returns:
    np.ndarray, concatenated data array for the specified base variable names
    """
    data_base_var = {}
    for var in data.keys():
        data_base_var[make_base_vars(var)] = data[var]
    concat_data = None
    for var in base_var_names:
        if concat_data is None:
            concat_data = data_base_var[var]
        elif var in data_base_var.keys():
            concat_data = np.concatenate((concat_data, data_base_var[var]), axis = 1)
        elif var not in data_base_var.keys():
            raise KeyError(f"Variable {var} not found in data.")

    return concat_data

def make_surface_vars(term: List[str], 
                      cause_var_name: str) -> List[str]:
    """
    Make surface variable names by adding intervention prefix to the cause variable.
    
    Parameters:
    term: List[str], 
        list of variable names in the term
    cause_var_name: str, 
        name of the cause variable
    
    Returns:
    List[str], list of surface variable names with intervention prefix for the cause variable
    """
    
    surface_vars = []
    for var in term:
        if var == cause_var_name:
            surface_vars.append("intv_"+var)
        else:
            surface_vars.append(var)
    return surface_vars

def build_base_pdfs(terms: List[set],
                    data: Dict[str, Any],
                    est_method: str = "histogram",
                    **est_kwargs) -> Dict[str, Callable]:
    """
    Build base probability density functions (PDFs) for the specified terms using the given estimation method.
    
    Parameters:
    terms: List[set],
        list of sets of variable names for which to build PDFs
    data: Dict[str, Any],
        dictionary mapping variable names to their data arrays
    est_method: str
        estimation method to use ("histogram", "kde", "multinorm", "kmeans", "gmm")
    est_kwargs: Dict[str, Any],
        additional keyword arguments for the estimation method. 
        For est_method = 
            "histogram", can include "n_bins" (Dict[str, int]); 
            "kde", can include "bandwidth" (str or float);
            "kmeans", can include "k" (int);
            "gmm", can include "n_components" (int).
            "multinorm" takes no additional arguments.

    Returns:
    Dict[str, Callable], dictionary mapping base variable name strings to their corresponding PDF functions
    """
 
    base_pdfs = {}
    for term_i in terms:
        sorted_term_i = sorted(term_i)
        base_var_names = [make_base_vars(var_j) for var_j in sorted_term_i]
        base_var_str = ",".join(base_var_names)
        if base_var_str not in base_pdfs:
            joint_data_i = concat_data(data, base_var_names)
            pdf_estimator = MultivarContiDistributionEstimator(data_fit=joint_data_i)
            if est_method == "histogram":
                if "n_bins" not in est_kwargs:
                    n_bins = [0]*joint_data_i.shape[1]
                else:
                    n_bins = []
                    n_bins_base = {make_base_vars(k):v for k,v in est_kwargs["n_bins"].items()}
                    for term_var_i in sorted_term_i:
                        n_bins += n_bins_base[make_base_vars(term_var_i)]
                pdf_func = pdf_estimator.fit_histogram(n_bins=n_bins)
            elif est_method == "kde":
                bandwidth = est_kwargs.get("bandwidth", "scott")
                pdf_func = pdf_estimator.fit_kde(bandwidth = bandwidth)
            elif est_method == "multinorm":
                pdf_func = pdf_estimator.fit_multinorm()
            elif est_method == "kmeans":
                k = est_kwargs.get("k", 10)
                pdf_func = pdf_estimator.fit_kmeans(k = k)
            elif est_method == "gmm":
                n_components = est_kwargs.get("n_components", 10)
                pdf_func = pdf_estimator.fit_gmm(n_components = n_components)
            else:
                raise NotImplementedError(f"Estimation method {est_method} is not implemented.")
            base_pdfs[base_var_str] = pdf_func

    return base_pdfs

def build_wrapper(pdf_handle, 
                  term_vars: List[str]) -> Callable:
    """
    Build a wrapper function for the given PDF handle to match the specified term variable names.
    
    Parameters:
    pdf_handle: Callable,
        the original PDF function that takes a list of variable values
    term_vars: List[str],
        list of variable names for the term, which may include intervention prefixes and prime postfixs
    
    Returns:
    Callable, a wrapper function that takes the same number of arguments as term_vars and calls pdf_handle
    """
    params = [var.replace("'", "_prime") for var in term_vars]
    params_list = ",".join(params)
    ns = {"_pdf_handle": pdf_handle}
    code = f"_f = lambda {params_list}: _pdf_handle([{params_list}])"
    exec(code, ns)
    return ns["_f"]

def build_dist_map(weight_func_expr: Any, 
                   data: Dict[str, np.ndarray],
                   cause_var_name: str, 
                   est_method: str = "histogram",
                   **est_kwargs) -> Dict[str, Callable]:
    """
    Build a distribution map for the terms in the weight function expression using the given estimation method.
    
    Parameters:
    weight_func_expr: Any,
        an object with attributes w_nom and w_denom, each being a list of sets of variable names
    data: Dict[str, np.ndarray],
        dictionary mapping variable names to their data arrays
    cause_var_name: str,
        name of the cause variable
    est_method: str,
        estimation method to use ("histogram", "kde", "multinorm", "kmeans", "gmm")
    est_kwargs: Dict[str, Any],
        additional keyword arguments for the estimation method. 
        For est_method = 
            "histogram", can include "n_bins" (Dict[str, int]); 
            "kde", can include "bandwidth" (str or float);
            "kmeans", can include "k" (int);
            "gmm", can include "n_components" (int).
            "multinorm" takes no additional arguments.
    
    Returns:
    Dict[str, Callable], dictionary mapping surface variable name strings to their corresponding PDF functions
    """
    
    dist_map = {}
    base_pdfs = build_base_pdfs(weight_func_expr.w_nom + weight_func_expr.w_denom,
                                data, 
                                est_method = est_method,
                                **est_kwargs)
    all_terms = weight_func_expr.w_nom + weight_func_expr.w_denom
    for term_i in all_terms:
        sorted_term_i = sorted(term_i)
        surface_vars = make_surface_vars(sorted_term_i, cause_var_name)
        base_vars = [make_base_vars(var_j) for var_j in sorted_term_i]
        dist_map_key_i = ",".join(surface_vars)
        pdf_key_i = ",".join(base_vars)
        if dist_map_key_i not in dist_map:
            dist_map[dist_map_key_i] = build_wrapper(base_pdfs[pdf_key_i], surface_vars)

    return dist_map