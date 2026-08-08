

import numpy as np
import pandas as pd
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from causalbootstrapping import backend
from causalbootstrapping.distEst_lib import MultivarContiDistributionEstimator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt

def build_cnn(input_shape=(28, 28, 1)):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(8, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(16, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        
        layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.Flatten(),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def preprocess_for_cnn(X, y=None):
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 2 and X.shape[1] == 28 * 28:
        X = X.reshape(-1, 28, 28, 1)
    elif X.ndim == 3:
        X = X[..., np.newaxis]
    elif X.ndim == 4:
        pass
    else:
        raise ValueError(f"Unexpected X shape: {X.shape}")

    X = X / 255.0

    if y is None:
        return X

    y = np.asarray(y).reshape(-1).astype(np.float32)
    return X, y

def train_cnn_classifier(X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=128):
    X_train, y_train = preprocess_for_cnn(X_train, y_train)

    if X_val is not None and y_val is not None:
        X_val, y_val = preprocess_for_cnn(X_val, y_val)
        validation_data = (X_val, y_val)
    else:
        validation_data = None

    model = build_cnn()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        mode="min",
        verbose=0
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop]
    )
    return model, history

def predict_cnn(model, X):
    X = preprocess_for_cnn(X)
    prob = model.predict(X, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    return pred, prob

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bernoulli_p(y):
    term_u10 = sigmoid(alpha_z + beta_zy*(2*y-1) - beta_zu1)
    term_u11 = sigmoid(alpha_z + beta_zy*(2*y-1) + beta_zu1)
    return (1 - p_u1) * term_u10 + p_u1 * term_u11

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

def visual_check(data, n_samples=8, base_pattern=True, title="", save_dir=None):
    digit_2_imgs = data["X"][data["Y"].flatten() == 0]
    digit_6_imgs = data["X"][data["Y"].flatten() == 1]
    idx_2 = np.random.choice(digit_2_imgs.shape[0], n_samples, replace=False)
    idx_6 = np.random.choice(digit_6_imgs.shape[0], n_samples, replace=False)

    if base_pattern:
        fig, axes = plt.subplots(2, n_samples+1, figsize=((n_samples+1)*1.6, 3.8))
        axes[0, 0].imshow(data["base_pattern"].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title("Base Pattern", fontsize=16)
        axes[0, 0].axis("off")
        axes[1, 0].axis("off")

        for i in range(n_samples):
            axes[0, i+1].imshow(digit_2_imgs[idx_2[i]].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
            axes[0, i+1].axis("off")
            axes[1, i+1].imshow(digit_6_imgs[idx_6[i]].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
            axes[1, i+1].axis("off")

        axes[0, 1].set_title("Digit 2", fontsize=16)
        axes[1, 1].set_title("Digit 6", fontsize=16)
    else:
        fig, axes = plt.subplots(2, n_samples, figsize=((n_samples)*1.6, 3.8))
        for i in range(n_samples):
            axes[0, i].imshow(digit_2_imgs[idx_2[i]].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
            axes[0, i].axis("off")
            axes[1, i].imshow(digit_6_imgs[idx_6[i]].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
            axes[1, i].axis("off")

        axes[0, 0].set_title("Digit 2", fontsize=16)
        axes[1, 0].set_title("Digit 6", fontsize=16)

    fig.suptitle(title, fontsize=18)
    plt.subplots_adjust(wspace=0.01)
    if save_dir is not None:
        plt.savefig(f"{save_dir}/{title.replace(' ', '_').replace('.', '').replace(':', '').replace(',', '')}.png", bbox_inches="tight")

    plt.show()

data_dir = r"C:/Users/jxm1417/Documents/experiment_code/transportable causal bootstrapping/experiment/semi_syn_bgMNIST/data/"
prefix = "bgMNIST"
envs = ["env0", "env1", "env2", "env4", "env5", "env6"]
env_fullname = ["Benchmark", "Gradient", "Stripes", "Rotation", "Rescale", "Shift"]
confounding = ["conf", "unconf"]
confounding_fullname = ["Confounded", "Non-confounded"]
subset = ["train", "test"]
vars = ["Y", "X", "Z", "W"]

data_dict = {}
for e in envs:
    for c in confounding:
        for s in subset:
            data_subset = {}
            for v in vars:
                data_subset[v] = pd.read_csv(f"{data_dir}{prefix}_{e}_{c}_{s}_{v}.csv", header=None).values
            data_dict[f"{e}_{c}_{s}"] = data_subset

for env in envs:
    for conf in confounding:
        data_subset = data_dict[f"{env}_{conf}_test"]
        visual_check(data_subset, n_samples=10, base_pattern=True, title=f"Env.: {env_fullname[envs.index(env)]}, {confounding_fullname[confounding.index(conf)]} data")            

p_u1 = 0.65
alpha_z = 0.0
beta_zy = 4.0
beta_zu1 = 1.0

pz_doy = lambda z, intv_y: bernoulli(bernoulli_p(intv_y)).pmf(z) 

intv_Y = np.unique(data_dict["env0_conf_train"]["Y"])

n_bins_z = [0]
n_bins_yz = [0,0]
n_bins_wyz = [50,0,0]

pdf_est_wyz = {}
pdf_est_yz = {}
pdf_est_z = {}

deconf_tcb_data = {}
deconf_clf = {}
conf_clf = {}
unconf_clf = {}
for env in envs:
    Y_conf_train = data_dict[f"{env}_conf_train"]["Y"]
    Z_conf_train = data_dict[f"{env}_conf_train"]["Z"]
    W_conf_train = data_dict[f"{env}_conf_train"]["W"]
    X_conf_train = data_dict[f"{env}_conf_train"]["X"]
    X_unconf_train = data_dict[f"{env}_unconf_train"]["X"]
    Y_unconf_train = data_dict[f"{env}_unconf_train"]["Y"]

    WYZ_conf = np.concatenate([W_conf_train, Y_conf_train, Z_conf_train], axis=1)
    YZ_conf = np.concatenate([Y_conf_train, Z_conf_train], axis=1)

    dist_est_wyz = MultivarContiDistributionEstimator(WYZ_conf)
    pdf_est_wyz[env] = dist_est_wyz.fit_kde()

    dist_est_yz = MultivarContiDistributionEstimator(YZ_conf)
    pdf_est_yz[env] = dist_est_yz.fit_histogram(n_bins=n_bins_yz)

    dist_est_z = MultivarContiDistributionEstimator(Z_conf_train)
    pdf_est_z[env] = dist_est_z.fit_histogram(n_bins=n_bins_z)

    deconf_tcb_data[env] = tcb_bootstrap_wrapper(
        intv_Y,
        Y_conf_train,
        Z_conf_train,
        W_conf_train,
        pdf_est_wyz[env],
        pdf_est_yz[env],
        pdf_est_z[env],
        pz_doy,
        X_conf_train,
        Y_conf_train,
    )
    
    deconf_clf[env] = KNeighborsClassifier(n_neighbors=5)
    deconf_clf[env].fit(deconf_tcb_data[env]["X"], deconf_tcb_data[env]["Y"].ravel())

    conf_clf[env] = KNeighborsClassifier(n_neighbors=5)
    conf_clf[env].fit(X_conf_train, Y_conf_train.ravel())

    unconf_clf[env] = KNeighborsClassifier(n_neighbors=5)
    unconf_clf[env].fit(X_unconf_train, Y_unconf_train.ravel())
    

demo_samples = 8
save_dir  = r"C:/Users/jxm1417/Documents/experiment_code/transportable causal bootstrapping/experiment/semi_syn_bgMNIST/results/demo_figs"

for env in envs:
    data_subset = deconf_tcb_data[env]
    visual_check(data_subset, n_samples=demo_samples, base_pattern=False, 
                 title=f"Env.: {env_fullname[envs.index(env)]}, TCB deconfounded", save_dir=save_dir)

evaluation_rows = []
model_groups = {
    "confounded": conf_clf,
    "non-confounded": unconf_clf,
    "deconfounded": deconf_clf,
}
test_groups = {
    "confounded": "conf",
    "non-confounded": "unconf",
}

for train_env in envs:
    for test_env in envs:
        for model_name, model_dict in model_groups.items():
            model = model_dict[train_env]
            for test_name, test_key in test_groups.items():
                X_test = data_dict[f"{test_env}_{test_key}_test"]["X"]
                Y_test = data_dict[f"{test_env}_{test_key}_test"]["Y"].ravel()
                y_pred = model.predict(X_test)
                acc = accuracy_score(Y_test.ravel(), y_pred)

                evaluation_rows.append(
                    {
                        "train_env": train_env,
                        "test_env": test_env,
                        "model_type": model_name,
                        "test_subset": test_name,
                        "accuracy": acc,
                    }
                )

res_table = pd.DataFrame(evaluation_rows)
res_table.to_csv("bgMNIST_tcb_evaluation_results_knn.csv", index=False)

save_dir  = r"C:/Users/jxm1417/Documents/experiment_code/transportable causal bootstrapping/experiment/semi_syn_bgMNIST/results/demo_figs"

try:
    df_plot = res_table[res_table["test_subset"] == "non-confounded"].copy()
except:
    res_table = pd.read_csv("bgMNIST_tcb_evaluation_results_knn.csv")
    df_plot = res_table[res_table["test_subset"] == "non-confounded"].copy()
model_order = ["Confounded Model", "TCB Deconfounded Model", "Non-confounded Model"]
model_name_mapping = {
    "confounded": "Confounded Model",
    "deconfounded": "TCB Deconfounded Model",
    "non-confounded": "Non-confounded Model"
}
df_plot["model_type"] = df_plot["model_type"].map(model_name_mapping)
envs = ["env0", "env1", "env2", "env4", "env5", "env6"]
env_mapping = dict(zip(envs, env_fullname))

df_plot["train_env"] = df_plot["train_env"].map(env_mapping)
df_plot["test_env"] = df_plot["test_env"].map(env_mapping)

sns.set_theme(style="whitegrid", font_scale=1.1)

g = sns.catplot(
    data=df_plot,
    kind="bar",
    x="test_env",
    y="accuracy",
    hue="model_type",
    col="train_env",
    col_wrap=3,
    order=env_fullname,
    hue_order=model_order,
    col_order=env_fullname,
    height=4.2,
    aspect=1.0,
    sharey=True,
    errorbar=None
)

g.set_titles("Training Env.: {col_name}", size=16)
g.set(ylim=(0.4, 1.0))

ncols = 3
for i, ax in enumerate(g.axes.flat):
    row = i // ncols
    col = i % ncols

    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    if col == 0:
        ax.tick_params(axis="y", labelsize=13, left=True, labelleft=True)
        ax.set_ylabel("Accuracy", fontsize=15)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_ylabel("")

    if row == 1:
        ax.tick_params(axis="x", labelsize=13, bottom=True, labelbottom=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=13)
        ax.set_xlabel("Test Env.", fontsize=15)
    else:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_xlabel("")

sns.move_legend(
    g,
    "lower center",
    title="Model",
    frameon=True,
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    fontsize=14,
    title_fontsize=13
)

g.fig.subplots_adjust(
    left=0.07,
    right=0.995,
    top=0.90,
    bottom=0.18,
    hspace=0.25,
    wspace=0.12
)

g.fig.suptitle("Accuracy on non-confounded test data", y=0.98, fontsize=18)
plt.savefig(f"{save_dir}/accuracy_barplot_non_confounded_test.png", bbox_inches="tight")
plt.show()

sns.set_theme(style="white", font_scale=1.1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
hm = None
for ax, model in zip(axes, model_order):
    ax.grid(False)

    sub = df_plot[df_plot["model_type"] == model]
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
    ax.set_xlabel("Test Env.", fontsize=15)
    ax.set_xticklabels(env_fullname, rotation=45, ha="center", fontsize=13)

    if ax is axes[0]:
        ax.set_ylabel("Train Env.", fontsize=15)
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

plt.suptitle("Accuracy heatmaps on non-confounded test data", y=0.96, fontsize=18)
plt.savefig(f"{save_dir}/accuracy_heatmap_non_confounded_test.png", bbox_inches="tight")
plt.show()
