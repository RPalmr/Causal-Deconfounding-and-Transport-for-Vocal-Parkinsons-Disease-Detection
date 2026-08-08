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

bd_data = dataGenerator.syn_data_gen(causal_setting = "backdoor", N = 5000, random_seed = 17, visual = False)

X_nonconf_env1 = bd_data.loc[bd_data["env"] == "non-confounded_1", ["X1", "X2"]].values
X_nonconf_env2 = bd_data.loc[bd_data["env"] == "non-confounded_2", ["X1", "X2"]].values
X_conf_env1 = bd_data.loc[bd_data["env"] == "confounded_1", ["X1", "X2"]].values
X_conf_env2 = bd_data.loc[bd_data["env"] == "confounded_2", ["X1", "X2"]].values

Y_nonconf_env1 = bd_data.loc[bd_data["env"] == "non-confounded_1", "Y"].values.reshape(-1,1)
Y_nonconf_env2 = bd_data.loc[bd_data["env"] == "non-confounded_2", "Y"].values.reshape(-1,1)
Y_conf_env1 = bd_data.loc[bd_data["env"] == "confounded_1", "Y"].values.reshape(-1,1)
Y_conf_env2 = bd_data.loc[bd_data["env"] == "confounded_2", "Y"].values.reshape(-1,1)

U_nonconf_env1 = bd_data.loc[bd_data["env"] == "non-confounded_1", "U"].values.reshape(-1,1)
U_nonconf_env2 = bd_data.loc[bd_data["env"] == "non-confounded_2", "U"].values.reshape(-1,1)
U_conf_env1 = bd_data.loc[bd_data["env"] == "confounded_1", "U"].values.reshape(-1,1)
U_conf_env2 = bd_data.loc[bd_data["env"] == "confounded_2", "U"].values.reshape(-1,1)

nonconf_clf_env1 = svm.SVC(kernel = 'linear', C=5)
nonconf_clf_env1 = nonconf_clf_env1.fit(X_nonconf_env1, Y_nonconf_env1.reshape(-1))

nonconf_clf_env2 = svm.SVC(kernel = 'linear', C=5)
nonconf_clf_env2 = nonconf_clf_env2.fit(X_nonconf_env2, Y_nonconf_env2.reshape(-1))

conf_clf_env1 = svm.SVC(kernel = 'linear', C=5)
conf_clf_env1 = conf_clf_env1.fit(X_conf_env1, Y_conf_env1.reshape(-1))

conf_clf_env2 = svm.SVC(kernel = 'linear', C=5)
conf_clf_env2 = conf_clf_env2.fit(X_conf_env2, Y_conf_env2.reshape(-1))

def backdoor_tcb_weights(intv_y, y_n, u_n, N, P_U, P_star_UY, kernel):
    weights = np.zeros(N)
    for i in range(N):
        weight_n = (P_U(u_n[i]) * kernel(intv_y[i], y_n[i])) / (N*P_star_UY([u_n[i].reshape(-1), intv_y[i].reshape(-1)]))
        weights[i] = weight_n[0]
    return weights.reshape(-1, 1)

n_bins_u = [0]
n_bins_uy = [0,0]

data_conf_env1_uy = np.concatenate((U_conf_env1, Y_conf_env1), axis = 1)
data_conf_env2_uy = np.concatenate((U_conf_env2, Y_conf_env2), axis = 1)

dist_estimator_uy_env1 = MultivarContiDistributionEstimator(data_fit=data_conf_env1_uy)
pdf_uy_env1 = dist_estimator_uy_env1.fit_histogram(n_bins = n_bins_uy)

dist_estimator_uy_env2 = MultivarContiDistributionEstimator(data_fit=data_conf_env2_uy)
pdf_uy_env2 = dist_estimator_uy_env2.fit_histogram(n_bins = n_bins_uy)

dist_estimator_u = MultivarContiDistributionEstimator(data_fit=U_nonconf_env1)
pdf_u = dist_estimator_u.fit_histogram(n_bins = n_bins_u)

dist_estimator_u_env1 = MultivarContiDistributionEstimator(data_fit=U_conf_env1)
pdf_u_env1 = dist_estimator_u_env1.fit_histogram(n_bins = n_bins_u)

dist_estimator_u_env2 = MultivarContiDistributionEstimator(data_fit=U_conf_env2)
pdf_u_env2= dist_estimator_u_env2.fit_histogram(n_bins = n_bins_u)

N = Y_conf_env2.shape[0]
kernel = lambda y, y_n: 1 if y == y_n else 0
intv_0 = np.array([0 for i in range(N)]).reshape(-1,1)
intv_1 = np.array([1 for i in range(N)]).reshape(-1,1)

dist_map = {"U,intv_Y": lambda U,intv_Y : pdf_uy_env1([U, intv_Y]),
            "U": lambda U: pdf_u_env1(U)}

cause_data = {"Y": Y_conf_env1.reshape(-1,1)}
effect_data = {"X": X_conf_env1}
confounder_data = {"U": U_conf_env1.reshape(-1,1)}

kernel_y = lambda Y, Y_intv: np.equal(Y, Y_intv).astype(np.int8)
cb_data_intv0_env1 = workflows.backdoor_cf(cause_data = cause_data, 
                                           effect_data = effect_data, 
                                           confounder_data = confounder_data, 
                                           dist_map = dist_map,
                                           intv_dict = {"intv_Y": 0},
                                           n_sample = 5000,
                                           kernel_intv = kernel_y)
cb_data_intv1_env1 = workflows.backdoor_cf(cause_data = cause_data, 
                                            effect_data = effect_data, 
                                            confounder_data = confounder_data, 
                                            dist_map = dist_map,
                                            intv_dict = {"intv_Y": 1},
                                            n_sample = 5000,
                                            kernel_intv = kernel_y)
X_deconf_env1 = np.concatenate((cb_data_intv0_env1["X"], cb_data_intv1_env1["X"]), axis = 0)
Y_deconf_env1 = np.concatenate((cb_data_intv0_env1["intv_Y"], cb_data_intv1_env1["intv_Y"]), axis = 0)

deconf_clf_env1 = svm.SVC(kernel = 'linear', C=5)
deconf_clf_env1 = deconf_clf_env1.fit(X_deconf_env1, Y_deconf_env1.reshape(-1))

shuffled_indices = np.random.permutation(X_deconf_env1.shape[0])
X_deconf_env1_shuffled = X_deconf_env1[shuffled_indices]
Y_deconf_env1_shuffled = Y_deconf_env1[shuffled_indices]

weights_env2_intv0 = backdoor_tcb_weights(intv_y = intv_0, y_n = Y_conf_env2, u_n = U_conf_env2, 
                                          N = N, P_U = pdf_u, P_star_UY = pdf_uy_env2, kernel = kernel)

weights_env2_intv1 = backdoor_tcb_weights(intv_y = intv_1, y_n = Y_conf_env2, u_n = U_conf_env2,
                                          N = N, P_U = pdf_u, P_star_UY = pdf_uy_env2, kernel = kernel)
weights_env2 = np.concatenate((weights_env2_intv0, weights_env2_intv1), axis = 1)

data_conf_env2 = {"X": X_conf_env2, "Y": Y_conf_env2, "U": U_conf_env2}

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

x1_range = [bd_data["X1"].min() - 0.02* np.ptp(bd_data["X1"].values),
            bd_data["X1"].max() + 0.02* np.ptp(bd_data["X1"].values)]
x2_range = [bd_data["X2"].min() - 0.02* np.ptp(bd_data["X2"].values),
            bd_data["X2"].max() + 0.02* np.ptp(bd_data["X2"].values)]

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
