#%%
import numpy as np
import matplotlib.pyplot as plt
import syn_data_gen as dataGenerator
from sklearn import svm
from scipy.stats import norm
from causalbootstrapping import backend, workflows
from causalbootstrapping.distEst_lib import MultivarContiDistributionEstimator

plt.rcParams['font.family'] = 'Calibri'
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 14
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
#%% frontdoor dataset (discrete Y, discrete U, Continuous X)

fd_data = dataGenerator.syn_data_gen(causal_setting = "frontdoor", N = 5000, random_seed = 17, visual = False)

X_nonconf_env1 = fd_data.loc[fd_data["env"] == "non-confounded_1", ["X1", "X2"]].values
X_nonconf_env2 = fd_data.loc[fd_data["env"] == "non-confounded_2", ["X1", "X2"]].values
X_conf_env1 = fd_data.loc[fd_data["env"] == "confounded_1", ["X1", "X2"]].values
X_conf_env2 = fd_data.loc[fd_data["env"] == "confounded_2", ["X1", "X2"]].values

Y_nonconf_env1 = fd_data.loc[fd_data["env"] == "non-confounded_1", "Y"].values.reshape(-1,1)
Y_nonconf_env2 = fd_data.loc[fd_data["env"] == "non-confounded_2", "Y"].values.reshape(-1,1)
Y_conf_env1 = fd_data.loc[fd_data["env"] == "confounded_1", "Y"].values.reshape(-1,1)
Y_conf_env2 = fd_data.loc[fd_data["env"] == "confounded_2", "Y"].values.reshape(-1,1)

Z_conf_env1 = fd_data.loc[fd_data["env"] == "confounded_1", "Z"].values.reshape(-1,1)
Z_conf_env2 = fd_data.loc[fd_data["env"] == "confounded_2", "Z"].values.reshape(-1,1)

# %% classifiers trained on the original non-confounded and confounded data
nonconf_clf_env1 = svm.SVC(kernel = 'linear', C=5)
nonconf_clf_env1 = nonconf_clf_env1.fit(X_nonconf_env1, Y_nonconf_env1.reshape(-1))

nonconf_clf_env2 = svm.SVC(kernel = 'linear', C=5)
nonconf_clf_env2 = nonconf_clf_env2.fit(X_nonconf_env2, Y_nonconf_env2.reshape(-1))

conf_clf_env1 = svm.SVC(kernel = 'linear', C=5)
conf_clf_env1 = conf_clf_env1.fit(X_conf_env1, Y_conf_env1.reshape(-1))

conf_clf_env2 = svm.SVC(kernel = 'linear', C=5)
conf_clf_env2 = conf_clf_env2.fit(X_conf_env2, Y_conf_env2.reshape(-1))

#%% TCB weights estimation
p_z_doy = lambda z, y: norm(1.8, 1).pdf(z) if y == 1 else norm(-1.8, 1).pdf(z)

def frontdoor_tcb_weights(intv_y, y_n, z_n, N, P_star_Y, P_star_ZY, P_Z_doY):
    weights = np.zeros(N)
    for i in range(N):
        weight_n = (P_star_Y(y_n[i]) * P_Z_doY(z_n[i], intv_y[i])) / (N * P_star_ZY([z_n[i].reshape(-1), y_n[i].reshape(-1)]))
        weights[i] = weight_n[0]
    return weights.reshape(-1, 1)

def frontdoor_tcb_weights_obs(intv_y, y_n, z_n, N, P_star_Y, P_star_ZY, P_Y, P_ZY):
    weights = np.zeros(N)
    for i in range(N):
        weight_n = (P_star_Y(y_n[i]) * P_ZY([z_n[i].reshape(-1), intv_y[i].reshape(-1)])) / (N * P_star_ZY([z_n[i].reshape(-1), y_n[i].reshape(-1)]) * P_Y(intv_y[i]))
        weights[i] = weight_n[0]
    return weights.reshape(-1, 1)

n_bins_y = [0]
n_bins_zy = [10,0]

data_conf_env1_zy = np.concatenate((Z_conf_env1, Y_conf_env1), axis = 1)
data_conf_env2_zy = np.concatenate((Z_conf_env2, Y_conf_env2), axis = 1)

dist_estimator_zy_env1 = MultivarContiDistributionEstimator(data_fit=data_conf_env1_zy)
pdf_zy_env1 = dist_estimator_zy_env1.fit_histogram(n_bins = n_bins_zy)

dist_estimator_zy_env2 = MultivarContiDistributionEstimator(data_fit=data_conf_env2_zy)
pdf_zy_env2 = dist_estimator_zy_env2.fit_histogram(n_bins = n_bins_zy)

dist_estimator_y_env1 = MultivarContiDistributionEstimator(data_fit=Y_conf_env1)
pdf_y_env1 = dist_estimator_y_env1.fit_histogram(n_bins = n_bins_y)

dist_estimator_y_env2 = MultivarContiDistributionEstimator(data_fit=Y_conf_env2)
pdf_y_env2 = dist_estimator_y_env2.fit_histogram(n_bins = n_bins_y)

N = Y_conf_env2.shape[0]
intv_0 = np.array([0 for i in range(N)]).reshape(-1,1)
intv_1 = np.array([1 for i in range(N)]).reshape(-1,1)

#%% env1

dist_map = {"Z,intv_Y": lambda Z, intv_Y: pdf_zy_env1([Z, intv_Y]),
            "Z,Y'": lambda Z, Y_prime: pdf_zy_env1([Z, Y_prime]),
            "intv_Y": lambda intv_Y: pdf_y_env1(intv_Y),
            "Y'": lambda Y_prime: pdf_y_env1(Y_prime)}
cause_data = {"Y'": Y_conf_env1}
mediator_data = {"Z": Z_conf_env1}
effect_data = {"X": X_conf_env1}

cb_data_intv0_env1 = workflows.frontdoor_cf(cause_data = cause_data, 
                                            effect_data = effect_data, 
                                            mediator_data = mediator_data, 
                                            dist_map = dist_map,
                                            intv_dict = {"intv_Y": 0},
                                            n_sample = 5000)
cb_data_intv1_env1 = workflows.frontdoor_cf(cause_data = cause_data, 
                                            effect_data = effect_data, 
                                            mediator_data = mediator_data, 
                                            dist_map = dist_map,
                                            intv_dict = {"intv_Y": 1},
                                            n_sample = 5000)

X_deconf_env1 = np.concatenate((cb_data_intv0_env1["X"], cb_data_intv1_env1["X"]), axis = 0)
Y_deconf_env1 = np.concatenate((cb_data_intv0_env1["intv_Y"], cb_data_intv1_env1["intv_Y"]), axis = 0)

deconf_clf_env1 = svm.SVC(kernel = 'linear', C=5)
deconf_clf_env1 = deconf_clf_env1.fit(X_deconf_env1, Y_deconf_env1.reshape(-1))

shuffled_indices = np.random.permutation(X_deconf_env1.shape[0])
X_deconf_env1_shuffled = X_deconf_env1[shuffled_indices]
Y_deconf_env1_shuffled = Y_deconf_env1[shuffled_indices]
#%% env2

weights_env2_intv0 = frontdoor_tcb_weights(intv_y = intv_0, y_n = Y_conf_env2, z_n = Z_conf_env2, 
                                          N = N, P_star_Y = pdf_y_env2, P_star_ZY = pdf_zy_env2, 
                                          P_Z_doY = p_z_doy)

weights_env2_intv1 = frontdoor_tcb_weights(intv_y = intv_1, y_n = Y_conf_env2, z_n = Z_conf_env2,
                                          N = N, P_star_Y = pdf_y_env2, P_star_ZY = pdf_zy_env2, 
                                          P_Z_doY = p_z_doy)
   
data_conf_env2 = {"X": X_conf_env2, "Y": Y_conf_env2, "Z": Z_conf_env2}

tcb_data_intv0 = backend.cw_bootstrapper(data = data_conf_env2,
                                         weights = weights_env2_intv0,
                                         intv_dict = {"intv_Y": 0},
                                         n_sample = 5000,
                                         sampling_mode = "fast")

tcb_data_intv1 = backend.cw_bootstrapper(data = data_conf_env2,
                                         weights = weights_env2_intv1,
                                         intv_dict = {"intv_Y": 1},
                                         n_sample = 5000,
                                         sampling_mode = "fast")

X_deconf_env2_tcb = np.concatenate((tcb_data_intv0["X"], tcb_data_intv1["X"]), axis = 0)
Y_deconf_env2_tcb = np.concatenate((tcb_data_intv0["intv_Y"], tcb_data_intv1["intv_Y"]), axis = 0)

deconf_clf_env2 = svm.SVC(kernel = 'linear', C=5)
deconf_clf_env2 = deconf_clf_env2.fit(X_deconf_env2_tcb, Y_deconf_env2_tcb.reshape(-1))

shuffled_indices = np.random.permutation(X_deconf_env2_tcb.shape[0])
X_deconf_env2_tcb_shuffled = X_deconf_env2_tcb[shuffled_indices]
Y_deconf_env2_tcb_shuffled = Y_deconf_env2_tcb[shuffled_indices]

#%%

x1_range = [fd_data["X1"].min() - 0.02* np.ptp(fd_data["X1"].values),
            fd_data["X1"].max() + 0.02* np.ptp(fd_data["X1"].values)]
x2_range = [fd_data["X2"].min() - 0.02* np.ptp(fd_data["X2"].values),
            fd_data["X2"].max() + 0.02* np.ptp(fd_data["X2"].values)]

fig, axes = plt.subplots(3,2, figsize = (10,15))

axes[0, 0], _ = plot_boundary(axes[0, 0], X_nonconf_env1, Y_nonconf_env1,
                           nonconf_clf_env1, [0,0], x1_range, x2_range,
                           'Decision boundary of the non-confounded SVM \n in the non-confounded environment 1')
axes[0, 1], _ = plot_boundary(axes[0, 1], X_nonconf_env2, Y_nonconf_env2,
                           nonconf_clf_env2, [2,2], x1_range, x2_range,
                           'Decision boundary of the non-confounded SVM \n in the non-confounded environment 2')
axes[1, 0], _ = plot_boundary(axes[1, 0], X_conf_env1, Y_conf_env1,
                           conf_clf_env1, [0,0], x1_range, x2_range, 
                           'Decision boundary of the confounded SVM \n in the confounded environment 1')
axes[1, 1], _ = plot_boundary(axes[1, 1], X_conf_env2, Y_conf_env2,
                           conf_clf_env2, [2,2], x1_range, x2_range, 
                           'Decision boundary of the confounded SVM \n in the confounded environment 2')
axes[2, 0], _ = plot_boundary(axes[2, 0], X_deconf_env1_shuffled, Y_deconf_env1_shuffled,
                           deconf_clf_env1, [0,0], x1_range, x2_range, 
                           'Decision boundary of the deconfounded SVM \n in the confounded environment 1')
axes[2, 1], handles = plot_boundary(axes[2, 1], X_deconf_env2_tcb_shuffled, Y_deconf_env2_tcb_shuffled,
                           deconf_clf_env2, [2,2], x1_range, x2_range, 
                           'Decision boundary of the deconfounded SVM \n in the confounded environment 2')
for i in range(3):
    for j in range(2):
        if j == 1:
            axes[i,j].tick_params(labelleft=False)
            axes[i,j].set_ylabel("")
        if i < 2:
            axes[i,j].tick_params(labelbottom=False)
            axes[i,j].set_xlabel("")

labels = [r'$Y=0$ (class 1)', r'$Y=1$ (class 2)',
          'True class boundary', 'Confounder boundary', 'SVM decision boundary']
legend =fig.legend(handles=handles,
                   labels=labels,
                   loc='lower center',
                   bbox_to_anchor=(0.5, 0.08),
                   ncol=5,
                   markerscale=2)
for handle in legend.legend_handles:
    handle.set_alpha(1.0)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()
# %%
