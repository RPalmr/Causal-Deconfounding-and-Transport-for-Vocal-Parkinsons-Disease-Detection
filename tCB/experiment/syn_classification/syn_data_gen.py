import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["axes.titlesize"] = 16

def plot_boundary(ax, X, Y, clf, tb, xlim, ylim, plot_title):
    handles = []
    scatter = ax.scatter(x= X[:,0], y = X[:,1], c = Y, s = 7, alpha = 0.4, cmap='coolwarm')
    handles_scatter, labels_scatter = scatter.legend_elements(prop="colors")
    handles += handles_scatter
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    x_ = np.linspace(xlim[0], xlim[1], 100)
    
    true_b = ax.plot(tb, ylim, 'k', linewidth=4, label="True boundary")
    confounder = ax.plot(xlim, [0,0], 'orange', linewidth = 4, label = "Confounder boundary")
    
    def linear_decision_boundary(clf, xlim):
        weight = clf.coef_[0]
        bias = clf.intercept_[0]
        k = -weight[0] / weight[1]
        b = -bias / weight[1]
        x = np.linspace(xlim[0], xlim[1], 100)
        decison_boundary = k * x + b
        return decison_boundary
    
    db = linear_decision_boundary(clf, xlim)
    clf_b = ax.plot(x_, db, 'r', linewidth = 4, label= 'Classifier Decision')
    handles += [true_b[0], confounder[0], clf_b[0]]
    ax.set_title(plot_title)
    ax.set_xlabel(r"$X_1$")
    ax.set_ylabel(r"$X_2$")
    return ax, handles

def visual_check(data, true_boundary_lst):
    x1_range = [data["X1"].min() - 0.02* np.ptp(data["X1"].values),
                data["X1"].max() + 0.02* np.ptp(data["X1"].values)]
    x2_range = [data["X2"].min() - 0.02* np.ptp(data["X2"].values),
                data["X2"].max() + 0.02* np.ptp(data["X2"].values)]    

    fig, axes = plt.subplots(2,data['env'].nunique(),figsize = (6*data['env'].nunique(), 12))
    axes_flatten = axes.flatten()
    ax_idx = 0
    for confounded in data['confounded'].unique():
        for env in data['env'].unique():
            X = data[(data['env'] == env) & (data['confounded'] == confounded)][['X1', 'X2']].values
            Y = data[(data['env'] == env) & (data['confounded'] == confounded)]['Y'].values
            clf = svm.SVC(kernel = 'linear', C=5)
            clf = clf.fit(X, Y.reshape(-1))
            title = f"Decision boundary of the {confounded} SVM \n and the corresponding data in the environment {env}"
            axes_flatten[ax_idx], handles = plot_boundary(axes_flatten[ax_idx], 
                                                          X, Y, 
                                                          clf, true_boundary_lst[ax_idx], 
                                                          x1_range, x2_range, 
                                                          title)
            ax_idx += 1
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            if i != axes.shape[0] - 1:
                axes[i, j].tick_params(labelbottom=False)
                axes[i, j].set_xlabel('')
            if j > 0:
                axes[i, j].tick_params(labelleft=False)
                axes[i, j].set_ylabel('')
    labels = [r'$Y=0$ (class 1)', r'$Y=1$ (class 2)',
            'True class boundary', 'Confounder boundary', 'SVM decision boundary']
    legend =fig.legend( handles=handles,
                        labels=labels,
                        loc='lower center',
                        bbox_to_anchor=(0.5, 0.045),
                        ncol=5,
                        markerscale=2)
    for handle in legend.legendHandles:
        handle.set_alpha(1.0)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()        

       
def fd_discUY_contZX_syn_generator(env, confounded, N = 5000, random_seed = None):
    if random_seed is None:
        random_seed = np.random.randint(0, 1000)
    np.random.seed(random_seed)
    
    if confounded:
        qyu = 0.85
    else:
        qyu = 0.5
    
    if env == 0:
        qx1s = 0
    elif env == 1:
        qx1s = 2

    qu = 0.6
    qzy = 1.8
    qx1z = 2
    qx2u = 2
    noise_x1s = 0.5
    
    pu = np.array([1 - qu, qu])
    py_u = np.array([[qyu, 1 - qyu], [1 - qyu, qyu]])
    
    u_seed = np.random.rand(N)
    U = np.where(u_seed <= pu[1], 1, 0)
    
    y_seed = np.random.rand(N) 
    py = py_u[1, U]
    Y = np.where(y_seed <= py, 1, 0)
    
    muz = np.zeros((N, 1))
    i = (Y == 1)
    muz[i, 0] += qzy
    muz[~i, 0] -= qzy
    
    Z = muz + np.random.randn(N, 1)
    
    mux = np.zeros((N, 2))

    i = (Z[:, 0] >= 0)
    mux[i, 0] += qx1z
    mux[~i, 0] -= qx1z
    
    if env in (2,3):
        mux[:, 0] += qx1s + noise_x1s * np.random.randn(N)

    i = (U == 1)
    mux[i, 1] += qx2u
    mux[~i, 1] -= qx2u
    X = mux + np.random.randn(N, 2)
    
    return X.reshape(-1,2), Y.reshape(-1,1), Z.reshape(-1,1), U.reshape(-1,1)

def bd_discUY_contX_syn_generator(env, confounded, N = 5000, random_seed = None):
    np.random.seed(random_seed)
    
    if confounded:
        qyu = 0.85
    else:
        qyu = 0.5
    
    if env == 0:
        qx1s = 0
    elif env == 1:
        qx1s = 2
    
    qu = 0.6
    qx1y = 1.5
    qx2u = 2
    noise_x1s = 0.5
    
    pu = np.array([1 - qu, qu])
    py_u = np.array([[qyu, 1 - qyu], [1 - qyu, qyu]])
    
    u_seed = np.random.rand(N)
    U = np.where(u_seed <= pu[1], 1, 0)
    
    y_seed = np.random.rand(N) 
    py = py_u[1, U]
    Y = np.where(y_seed <= py, 1, 0)
    
    mux = np.zeros((N, 2)) 
    i = (Y == 0)
    mux[i, 0] += qx1y[env]
    mux[~i, 0] -= qx1y[env] 
    if env in (2,3):
        mux[:, 0] += qx1s[env] + np.random.randn(N) * noise_x1s
    
    i = (U == 0)
    mux[i, 1] += qx2u[env]
    mux[~i, 1] -= qx2u[env]

    X = mux + np.random.randn(N, 2)
    return X.reshape(-1,2), Y.reshape(-1,1), U.reshape(-1,1)

def non_ident_discUYZ_contWX_syn_generator(env, confounded, N, random_seed=None):
    random_seed = np.random.randint(0,1000) if random_seed is None else random_seed
    np.random.seed(random_seed)

    p_u1 = 0.65
    p_u2 = 0.65
    alpha_y = 0
    alpha_z = 0
    beta_zy = 2
    beta_zu1 = 0.75
    qw_z = 4
    gamma_x1w = 2
    gamma_x2u2 = 2

    if confounded:
        beta_yu1 = 0.5
        beta_yu2 = 1.75
    else:
        beta_yu1 = 0
        beta_yu2 = 0
    
    if env == 0:
        gamma_x1s1 = 0
        qw_s2 = 0
        noise_x1s1 = 0
        noise_ws2 = 0
    elif env == 1:
        gamma_x1s1 = 0
        qw_s2 = -1.5
        noise_x1s1 = 0
        noise_ws2 = 0.2
    elif env == 2:
        gamma_x1s1 = 5
        qw_s2 = 1.5
        noise_x1s1 = 0.2
        noise_ws2 = 0.2

    U1 = (np.random.rand(N) < p_u1).astype(int)
    U2 = (np.random.rand(N) < p_u2).astype(int)

    def sigmoid(x): return 1/(1+np.exp(-x))
    logit_y = alpha_y + beta_yu1*(2*U1-1) + beta_yu2*(2*U2-1)
    p_y     = sigmoid(logit_y)
    Y       = (np.random.rand(N) < p_y).astype(int)

    logit_z = alpha_z + beta_zy*(2*Y-1) + beta_zu1*(2*U1-1)
    p_z     = sigmoid(logit_z)
    Z       = (np.random.rand(N) < p_z).astype(int)

    mu_W = np.zeros((N, 1))
    i = (Z == 1)
    mu_W[i, 0] += qw_z
    mu_W[~i, 0] -= qw_z

    mu_W[:, 0] += qw_s2 + noise_ws2 * np.random.randn(N)
        
    W = mu_W + np.random.randn(N, 1)
    
    mux = np.zeros((N,2))
    i = (W[:, 0] >= 0)
    mux[i, 0] += gamma_x1w
    mux[~i, 0] -= gamma_x1w
    
    mux[:, 0] += gamma_x1s1 + noise_x1s1 * np.random.randn(N)

    i = (U2 == 1)
    mux[i, 1] += gamma_x2u2
    mux[~i, 1] -= gamma_x2u2
    
    X = mux +np.random.randn(N, 2)
    return X.reshape(-1,2), Y.reshape(-1,1), Z.reshape(-1,1), W.reshape(-1,1), U1.reshape(-1,1), U2.reshape(-1,1)

def syn_data_gen(causal_setting, N = 5000, random_seed = None, visual = True):
    
    if causal_setting == "frontdoor":

        data_dict = {"X1": np.empty((0)), 
                    "X2": np.empty((0)), 
                    "Y": np.empty((0)), 
                    "Z": np.empty((0)), 
                    "U": np.empty((0)), 
                    "env": np.empty((0), dtype=int),
                    "confounded": np.empty((0), dtype=object)}
        data_df = pd.DataFrame(data_dict)
        env_lst = [0,1]
        confounded_lst = [False, True]

        for env in env_lst:
            for confounded in confounded_lst:
                    X, Y, Z, U = fd_discUY_contZX_syn_generator(env = env, confounded = confounded, N = N, random_seed = random_seed)
                    temp_df = pd.DataFrame({
                                    "X1": X[:, 0],
                                    "X2": X[:, 1],
                                    "Y": Y[:, 0],
                                    "Z": Z[:, 0],
                                    "U": U[:, 0],
                                    "env": env,
                                    "confounded": "confounded" if confounded else "non-confounded"})
                    data_df = pd.concat([data_df, temp_df], ignore_index=True)

        if visual:
            visual_check(data_df, true_boundary_lst = [[0,0], [2,2], [0,0], [2,2]]) 
            
    elif causal_setting == "backdoor":

        data_dict = {"X1": np.empty((0)),
                    "X2": np.empty((0)),
                    "Y": np.empty((0)),
                    "U": np.empty((0)),
                    "env": np.empty((0), dtype=int),
                    "confounded": np.empty((0), dtype=object)}
        data_df = pd.DataFrame(data_dict)
        env_lst = [0,1]
        confounded_lst = [False, True]

        for env in env_lst:
            for confounded in confounded_lst:
                X, Y, U = bd_discUY_contX_syn_generator(env = env, confounded = confounded, N = N, random_seed = random_seed+env)
                temp_df = pd.DataFrame({
                                "X1": X[:, 0],
                                "X2": X[:, 1],
                                "Y": Y[:, 0],
                                "U": U[:, 0],
                                "env": env,
                                "confounded": "confounded" if confounded else "non-confounded"})
                data_df = pd.concat([data_df, temp_df], ignore_index=True)

        if visual:
            visual_check(data_df, true_boundary_lst = [[0,0], [2,2], [0,0], [2,2]])    
            
    elif causal_setting == "not_ident_1":       

        data_dict = {"X1": np.empty((0)),
                    "X2": np.empty((0)),
                    "Y": np.empty((0)),
                    "Z": np.empty((0)),
                    "W": np.empty((0)),
                    "U1": np.empty((0)),
                    "U2": np.empty((0)),
                    "env": np.empty((0), dtype=int),
                    "confounded": np.empty((0), dtype=object)}
        data_df = pd.DataFrame(data_dict)
        env_lst = [0,1,2]
        confounded_lst = [False, True]

        for env in env_lst: 
            for confounded in confounded_lst:
                X, Y, Z, W, U1, U2 = non_ident_discUYZ_contWX_syn_generator(env = env, confounded = confounded, N = N, random_seed = random_seed+env)
                temp_df = pd.DataFrame({
                                "X1": X[:, 0],
                                "X2": X[:, 1],
                                "Y": Y[:, 0],
                                "Z": Z[:, 0],
                                "W": W[:, 0],
                                "U1": U1[:, 0],
                                "U2": U2[:, 0],
                                "env": env,
                                "confounded": "confounded" if confounded else "non-confounded"})
                data_df = pd.concat([data_df, temp_df], ignore_index=True)

        if visual:
            visual_check(data_df, true_boundary_lst = [[0,0], [0,0], [5,5], [0,0], [0,0], [5,5]])
    elif causal_setting == "not_ident_2":
        pass
    else:
        raise ValueError("Unsupported causal setting. Choose from 'frontdoor', 'backdoor', 'not_ident_1', 'not_ident_2'.")
    return data_df
