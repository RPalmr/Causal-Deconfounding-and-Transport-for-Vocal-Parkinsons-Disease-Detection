
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import syn_data_gen as dataGenerator
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy.stats import bernoulli
from causalbootstrapping import backend
from causalbootstrapping.distEst_lib import MultivarContiDistributionEstimator
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.title_fontsize"] = 14

def plot_boundary(ax, X, Y, clf, tb, xlim, ylim, plot_title):
    handles = []
    scatter = ax.scatter(x= X[:,0], y = X[:,1], c = Y, s = 7, alpha = 0.3, cmap='coolwarm')
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
    ax.set_xlabel(r"$X_1$", fontsize=16)
    ax.set_ylabel(r"$X_2$", fontsize=16)
    return ax, handles

def tcb_bootstrap_wrapper(intv_Y, y_n, z_n, w_n,
                          P_star_WYZ, P_star_YZ, P_star_Z, P_Z_doY,
                          X_conf, Y_conf):
    conf_data = {"X": X_conf, "Y": Y_conf}
    tcb_data = {"X": np.empty((0, X_conf.shape[1])), "Y": np.empty((0, Y_conf.shape[1]))}
    for intv_y in intv_Y:
        N = y_n.shape[0]
        intv_y_arr = np.array([intv_y]*N).reshape(-1, 1)
        weights_i = tcb_weights(intv_y_arr, y_n, z_n, w_n, N, 
                                P_star_WYZ, P_star_YZ, P_star_Z, P_Z_doY)
        n_sample = y_n[y_n.flatten() == intv_y].shape[0]
        tcb_data_intv_yi = backend.cw_bootstrapper(data = conf_data,
                                                   weights = weights_i,
                                                   intv_dict = {"intv_Y": intv_y},
                                                   n_sample = n_sample,
                                                   sampling_mode = "fast")
        tcb_data["X"] = np.concatenate((tcb_data["X"], tcb_data_intv_yi["X"]), axis=0)
        tcb_data["Y"] = np.concatenate((tcb_data["Y"], tcb_data_intv_yi["intv_Y"]), axis=0)

    shuffled_indices = np.random.permutation(tcb_data["X"].shape[0])
    tcb_data["X"] = tcb_data["X"][shuffled_indices]
    tcb_data["Y"] = tcb_data["Y"][shuffled_indices]
    
    return tcb_data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bernoulli_p(y):
    term_u10 = sigmoid(alpha_z + beta_zy*(2*y-1) - beta_zu1)
    term_u11 = sigmoid(alpha_z + beta_zy*(2*y-1) + beta_zu1)
    return (1 - p_u1) * term_u10 + p_u1 * term_u11

p_z_doy = lambda z, intv_y: bernoulli(bernoulli_p(intv_y)).pmf(z) 

def tcb_weights(intv_y, y_n, z_n, w_n, N, 
                P_star_WYZ, P_star_YZ, P_star_Z, P_Z_doY):
    weights = np.zeros(N)
    for i in range(N):
        p_star_wintvyz = P_star_WYZ([w_n[i].reshape(-1), intv_y[i].reshape(-1), z_n[i].reshape(-1)])
        p_star_wyz = P_star_WYZ([w_n[i].reshape(-1), y_n[i].reshape(-1), z_n[i].reshape(-1)])
        p_star_intvyz = P_star_YZ([intv_y[i].reshape(-1), z_n[i].reshape(-1)])
        p_star_yz = P_star_YZ([y_n[i].reshape(-1), z_n[i].reshape(-1)])
        p_star_z = P_star_Z(z_n[i].reshape(-1))
        pz_doy = P_Z_doY(z_n[i].reshape(-1), intv_y[i].reshape(-1))
        weight_n = (p_star_yz * p_star_wintvyz * pz_doy) / (N * p_star_intvyz * p_star_wyz * p_star_z)
        weights[i] = weight_n[0]
    return weights.reshape(-1, 1)

complex_data_train = dataGenerator.syn_data_gen(causal_setting="not_ident_1", N=5000, random_seed=111, visual = False)

complex_data_train["Confounding Env."] = complex_data_train.apply(
    lambda row: f"Confounded Env. {row['env']}" if row["confounded"] == "confounded" else f"Non-confounded Env. {row['env']}",
    axis=1
)

data_hist_comparison = complex_data_train.loc[(complex_data_train["Confounding Env."] != "Confounded Env. 0") & 
                                              (complex_data_train["Confounding Env."] != "Confounded Env. 1") & 
                                              (complex_data_train["Confounding Env."] != "Confounded Env. 2"), :]
data_hist_comparison.loc[:, "Environment"] = "Environment " + data_hist_comparison.loc[:, "env"].astype(str)
plt.figure(figsize=(8,4))
plt.title("Distribution of variable X1 across environments")
ax = sns.histplot(data=data_hist_comparison, x="X1", hue="Environment",
                  bins=40, alpha=0.35, element="step", stat = "density", kde = True)
ax.set_xlabel("X1", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.tick_params(axis='both', labelsize=14)
ax.grid(True, linestyle='--', alpha=0.5)
sns.move_legend(ax, "upper right", title = "")
plt.show()

plt.figure(figsize=(8,4))
plt.title("Distribution of variable W across environments")
ax = sns.histplot(data=data_hist_comparison, x="W", hue="Environment", 
                  bins=40, alpha=0.35, element="step", stat = "density", kde = True)
ax.set_xlabel("W", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.tick_params(axis='both', labelsize=14)
ax.grid(True, linestyle='--', alpha=0.5)
sns.move_legend(ax, "upper right", title = "")
plt.show()

envs = [0, 1, 2]
subset_confounding = ["confounded", "non-confounded"]

p_u1 = 0.65
alpha_z = 0.0
beta_zy = 2
beta_zu1 = 0.75

n_bins_z = [0]
n_bins_yz = [0,0]
n_bins_wyz = [50,0,0]

deconf_tcb_data = {}
deconf_clf = {}
conf_clf = {}
unconf_clf = {}
intv_Y = np.unique(complex_data_train.loc[(complex_data_train['env'] == 0) & (complex_data_train["confounded"] == "confounded"), 'Y'])

for env in envs:

    X_conf_train = complex_data_train.loc[(complex_data_train['env'] == env) & (complex_data_train["confounded"] == "confounded"), ['X1', 'X2']].values
    Y_conf_train = complex_data_train.loc[(complex_data_train['env'] == env) & (complex_data_train["confounded"] == "confounded"), 'Y'].values.reshape(-1,1)
    Z_conf_train = complex_data_train.loc[(complex_data_train['env'] == env) & (complex_data_train["confounded"] == "confounded"), 'Z'].values.reshape(-1,1)
    W_conf_train = complex_data_train.loc[(complex_data_train['env'] == env) & (complex_data_train["confounded"] == "confounded"), 'W'].values.reshape(-1,1)
    
    X_unconf_train = complex_data_train.loc[(complex_data_train['env'] == env) & (complex_data_train["confounded"] == "non-confounded"), ['X1', 'X2']].values
    Y_unconf_train = complex_data_train.loc[(complex_data_train['env'] == env) & (complex_data_train["confounded"] == "non-confounded"), 'Y'].values.reshape(-1,1)
    
    WYZ_conf = np.concatenate([W_conf_train, Y_conf_train, Z_conf_train], axis=1)
    YZ_conf = np.concatenate([Y_conf_train, Z_conf_train], axis=1)
    
    dist_est_wyz = MultivarContiDistributionEstimator(data_fit=WYZ_conf)
    pdf_wyz = dist_est_wyz.fit_histogram(n_bins = n_bins_wyz)
    
    dist_est_yz = MultivarContiDistributionEstimator(data_fit=YZ_conf)
    pdf_yz = dist_est_yz.fit_histogram(n_bins = n_bins_yz)
    
    dist_est_z = MultivarContiDistributionEstimator(data_fit=Z_conf_train.reshape(-1, 1))
    pdf_z = dist_est_z.fit_histogram(n_bins = n_bins_z)
    
    deconf_tcb_data[env] = tcb_bootstrap_wrapper(
        intv_Y,
        Y_conf_train,
        Z_conf_train,
        W_conf_train,
        pdf_wyz,
        pdf_yz,
        pdf_z,
        p_z_doy,
        X_conf_train,
        Y_conf_train,
    )        

    deconf_clf[env] = svm.SVC(kernel = 'linear', C=5)
    deconf_clf[env].fit(deconf_tcb_data[env]["X"], deconf_tcb_data[env]["Y"].ravel())

    conf_clf[env] = svm.SVC(kernel = 'linear', C=5)
    conf_clf[env].fit(X_conf_train, Y_conf_train.ravel())

    unconf_clf[env] = svm.SVC(kernel = 'linear', C=5)
    unconf_clf[env].fit(X_unconf_train, Y_unconf_train.ravel())

complex_data_test = dataGenerator.syn_data_gen(causal_setting="not_ident_1", N=5000, random_seed=222, visual = False)
evaluation_rows = []
envs = [0, 1, 2]
model_groups = {
    "confounded": conf_clf,
    "non-confounded": unconf_clf,
    "deconfounded": deconf_clf,
}

x1_range = [complex_data_test["X1"].min() - 0.02* np.ptp(complex_data_test["X1"].values),
            complex_data_test["X1"].max() + 0.02* np.ptp(complex_data_test["X1"].values)]
x2_range = [complex_data_test["X2"].min() - 0.02* np.ptp(complex_data_test["X2"].values),
            complex_data_test["X2"].max() + 0.02* np.ptp(complex_data_test["X2"].values)]

for train_env in envs:
    for test_env in envs:
        for model_name, model_dict in model_groups.items():
            model = model_dict[train_env]
            for test_subset in subset_confounding:
                X_test = complex_data_test.loc[(complex_data_test['env'] == test_env) & (complex_data_test["confounded"] == test_subset), ['X1', 'X2']].values
                Y_test = complex_data_test.loc[(complex_data_test['env'] == test_env) & (complex_data_test["confounded"] == test_subset), 'Y'].values.reshape(-1,1)
                y_pred = model.predict(X_test)
                acc = accuracy_score(Y_test.ravel(), y_pred)
                evaluation_rows.append({
                    "train_env": train_env,
                    "test_env": test_env,
                    "model": model_name,
                    "test_subset": test_subset,
                    "accuracy": acc
                })

fig, axes = plt.subplots(3,3, figsize = (15,14))
for i, model_name in enumerate(model_groups.keys()):
    for j in envs:
        if model_name == "deconfounded":
            X = deconf_tcb_data[j]["X"]
            Y = deconf_tcb_data[j]["Y"]
        else:
            X = complex_data_test.loc[(complex_data_test['env'] == j) & (complex_data_test["confounded"] == model_name), ['X1', 'X2']].values
            Y = complex_data_test.loc[(complex_data_test['env'] == j) & (complex_data_test["confounded"] == model_name), 'Y'].values.reshape(-1,1)
        if j == 2:
            true_boundary = [5,5]
        else:
            true_boundary = [0,0]
        model = model_groups[model_name][j]
        axes[i,j], handles = plot_boundary(axes[i,j], X, Y, model, true_boundary, x1_range, x2_range, "")

        if i!=2:
            axes[i,j].tick_params(labelbottom=False)
            axes[i,j].set_xlabel("")
        if j!=0:
            axes[i,j].tick_params(labelleft=False)
            axes[i,j].set_ylabel("")
            
        if i==0:
            axes[i,j].set_title(f"Test Env. {j}", fontsize=16, fontweight='bold')
        
row_names = ["Confounded model", "Unconfounded model", "TCB Deconfounded model"]
labels = [r'$Y=0$ (class 1)', r'$Y=1$ (class 2)',
          'True class boundary', 'Confounder boundary', 'SVM decision boundary']
legend =fig.legend(handles=handles,
                   labels=labels,
                   loc='lower center',
                   bbox_to_anchor=(0.5, 0.07),
                   ncol=5,
                   markerscale=2)
for handle in legend.legend_handles:
    handle.set_alpha(1.0)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

for i, row_name in enumerate(row_names):
    pos = axes[i, 0].get_position()
    y_center = (pos.y0 + pos.y1) / 2
    fig.text(
        pos.x0 - 0.04, y_center,
        row_name,
        va='center', ha='right',
        fontsize=16, fontweight='bold',
        rotation=90
    )
plt.show()

save_dir = r"C:\Users\jxm1417\Documents\experiment_code\transportable causal bootstrapping\experiment\syn_classification\results\figures"
res_table_dir = r"C:\Users\jxm1417\Documents\experiment_code\transportable causal bootstrapping\experiment\syn_classification\results"
try:
    df_plot = res_table[res_table["test_subset"] == "non-confounded"].copy()
except:
    res_table = pd.read_csv(f"{res_table_dir}/synClas_tcb_evaluation_results.csv")
    df_plot = res_table[res_table["test_subset"] == "non-confounded"].copy()

model_order = ["Confounded Model", "TCB Deconfounded Model", "Non-confounded Model"]
model_name_mapping = {
    "confounded": "Confounded Model",
    "deconfounded": "TCB Deconfounded Model",
    "non-confounded": "Non-confounded Model"
}
df_plot["model"] = df_plot["model"].map(model_name_mapping)
envs = [0,1,2]
env_fullname = ["Env. 0", "Env. 1", "Env. 2"]
env_mapping = dict(zip(envs, env_fullname))

df_plot["train_env"] = df_plot["train_env"].map(env_mapping)
df_plot["test_env"] = df_plot["test_env"].map(env_mapping)

sns.set_theme(style="whitegrid", font_scale=1.1)

g = sns.catplot(
    data=df_plot,
    kind="bar",
    x="test_env",
    y="accuracy",
    hue="model",
    col="train_env",
    col_wrap=3,
    order=env_fullname,
    hue_order=model_order,
    col_order=env_fullname,
    height=6,
    aspect=1.0,
    sharey=True,
    errorbar=None
)

for i, ax in enumerate(g.axes.flat):
    if i == 0:
        ax.set_ylabel("Accuracy", fontsize=15)
        ax.tick_params(axis="y", labelsize=13, left=True, labelleft=True)
    else:
        ax.set_ylabel("")
    
    ax.set_xlabel("Test Env.", fontsize=15)
    ax.tick_params(axis="x", labelsize=13, bottom=True, labelbottom=True)

g.set_titles("Training Env.: {col_name}", size=16)
g.set(ylim=(0.4, 1.0))

sns.move_legend(
    g,
    "lower center",
    title="Model",
    frameon=True,
    bbox_to_anchor=(0.5, -0.03),
    ncol=3,
    fontsize=14,
    title_fontsize=13
)

g.figure.subplots_adjust(
    left=0.07,
    right=0.995,
    top=0.87,
    bottom=0.2,
    hspace=0.25,
    wspace=0.05
)

g.figure.suptitle("Accuracy on non-confounded test data", y=0.98, fontsize=20)
plt.savefig(f"{save_dir}/accuracy_bar_non_confounded_test.png", bbox_inches="tight")
plt.show()

sns.set_theme(style="white", font_scale=1.1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
hm = None
for ax, model in zip(axes, model_order):
    ax.grid(False)

    sub = df_plot[df_plot["model"] == model]
    pivot = sub.pivot_table(
        index="train_env",
        columns="test_env",
        values="accuracy",
        aggfunc="mean",
        observed=False
    ).reindex(index=env_fullname, columns=env_fullname)

    hm = sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.5,
        vmax=1.0,
        cbar=False
    )

    ax.set_title(model, fontsize=16)
    ax.set_xlabel("Test environment", fontsize=15)
    ax.set_xticklabels(env_fullname, rotation=45, ha="center", fontsize=13)

    if ax is axes[0]:
        ax.set_ylabel("Train environment", fontsize=15)
        ax.set_yticklabels(env_fullname, rotation=45, fontsize=13)
        ax.tick_params(axis="y", labelleft=True)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

plt.subplots_adjust(left=0.12, right=0.90, top=0.86, bottom=0.12, wspace=0.25)

cax = fig.add_axes([0.92, 0.15, 0.012, 0.70])

cbar = fig.colorbar(hm.collections[0], cax=cax)
cbar.set_label("Accuracy", fontsize=15)
cbar.ax.grid(False)

plt.suptitle("Accuracy heatmaps on non-confounded test data", y=0.96, fontsize=16)
plt.savefig(f"{save_dir}/accuracy_heatmap_non_confounded_test.png", bbox_inches="tight")
plt.show()
