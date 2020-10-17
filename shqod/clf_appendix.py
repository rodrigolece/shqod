import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


def compare_labels(y1, y2):
    """
    Return proportion of agreeing labels
    Assume that y1, y2 are 1d arrays with same size
    """
    assert y1.size == y2.size
    return (np.sum(y1 == y2)) / y1.size


def comparison_matrix(y_list):
    """
    Return a matrix of agreement between labels
    Assumes the input of list of 1d arrays of the same size
    """
    k = len(y_list)
    cmat = np.zeros((k, k))
    for i in range(k):
        cmat[i, i] = 1 / 2  # gets doubled later
        for j in range(i + 1, k):
            cmat[i, j] = compare_labels(y_list[i], y_list[j])
    cmat += cmat.T
    return cmat


def show_2dhist(arr, range=None, bins=[10, 10]):
    plt.hist2d(x=arr[:, 0], y=arr[:, 1], range=range, bins=bins)
    plt.show()


def get_smooth_features(paths, processor, feat_types):
    sig_flag = False
    if 'sig' in feat_types:
        sig_flag = True
        sig_out = np.zeros((len(paths), 8)
        feat_types.pop(feat_types.index('sig')

    methods = [getattr(processor, f'get_{feat}' ) for feat in feat_types]
    methods = list(filter(None.__ne__, methods))

    out = np.zeros((len(paths),  len(methods))

    for row, path in enumerate(paths):
        out[row] = [method(path) for method in methods]
        if sig:
            sig_out[row] = get_sig(path)

    if sig_flag:
        out = np.vstack((out, sig_out))
    return out


def outlier_clf(X, clf_type, contam=None):
    if clf_type == "svm":
        clf = OneClassSVM(kernel="rbf", gamma='auto', nu=contam).fit(X)
    elif clf_type == "ifor":
        clf = IsolationForest(contamination=contam).fit(X)
    return clf


# Options
clf_type = "ifor"  # use either {"svm", "ifor"}
feat_types = ["len", "curv", "dtb", "sig", "od_frob", "fractal", "mob_field"]
levels = [6, 8, 11]
genders = {"f", "m"}
APOE_genes = ["e3e3", "e3e4", "e4e4"]
APOE_plot_colors = ["green", "blue", "red"]
save_plot = True
show_plot = False

all_clf = []
all_y = []
all_sco = [] # sco = score
all_y_APOE = []
all_sco_APOE = []
for level in levels:
    clf_per_lvl = []
    y_per_lvl = []
    sco_per_lvl = []
    y_APOE_per_lvl = []
    sco_APOE_per_lvl = []

    for gender in genders:
        clf_per_lvlgen = []
        y_per_lvlgen = []
        sco_per_lvlgen = []
        y_APOE_per_lvlgen = []
        sco_APOE_per_lvlgen = []

        # Fetch gameplay path data (each datum = a list of 2d arrays)
        # It's paths_good_pre because we need to filter the pure ones before training it
        paths_good_pre = fetch_paths("normative", level, gender)  # FILL THIS ONE
        paths_bad = fetch_paths("AD", level, gender)  # FILL THIS ONE
        paths_APOE = [fetch_paths(f"APOE_{gene}", level, gender) for gene in APOE_genes]

        print("\n====================================")
        print(f"For level={level}, gender={gender}:")
        for feat_type in feat_types:
            # Calculate features
            X_good_pre = get_smooth_features(paths_good_pre, feat_type)
            X_bad = get_smooth_features(paths_bad, feat_type)
            X_APOE = [get_smooth_features(paths_APOE_per_gene, feat_type) for paths_APOE_per_gene in paths_APOE]

            # Filter outliers from normative data
            clf_pre = outliers_clf(X_good_pre, feat_type, clf_type)
            y_pre = clf_pre.predict(X_good_pre)
            X_good = X_good_pre[np.nonzero(y_pre == 1)[0]]
            X = [*X_good, *X_bad]
            n_good = len(X_good)
            n_bad = len(X_bad)
            # Might want to subsample X_good first due to imbalance
            contam = n_bad / (n_good + n_bad)

            # Train the classifier
            clf = outliers_clf(X, feat_type, clf_type, contam=contam)
            y = clf.predict(X)
            sco = clf.score_samples(X)
            y_good = y[:n_good]
            y_bad = y[n_good:]
            sco_good = sco[:n_good]
            sco_bad = sco[n_good:]

            # Display Training Results
            trained_contam = np.sum(y == -1) / y.shape[0]
            true_pos_rate = (y_good == 1) / X_good.shape[0]
            true_neg_rate = (y_bad == -1) / X_bad.shape[0]
            print("\n------------------------------------")
            print(f"Feature type {feat_type}:")
            print(f"Contamination = {trained_contam}")
            print(f"True positive rate = {true_pos_rate}")
            print(f"True negative rate = {true_neg_rate}")
            if save_plot or show_plot:
                plt.hist(sco_good, bin=50, color="blue", alpha=0.6)
                plt.hist(sco_bad, bin=50, color="red", alpha=0.6)
            if save_plot:
                plt.save(f"({level},{gender},{feat_type}) Training Score Histogram")
                plt.clf()
            if show_plot:
                plt.show()
                plt.clf()

            # APOE Data
            y_APOE = [clf.predict(X_APOE_per_gene) for X_APOE_per_gene in X_APOE]
            sco_APOE = [clf.score_samples(X_APOE_per_gene) for X_APOE_per_gene in X_APOE]

            if save_plot or show_plot:
                for ind_APOE, sco_APOE in enumerate(sco_APOE):
                    plt.hist(sco_good, bin=50, color=APOE_plot_colors[ind_APOE],
                             alpha=0.6, label=APOE_genes[ind_APOE])
            if save_plot:
                plt.save(f"({level},{gender},{feat_type}) APOE Score Histogram")
                plt.clf()
            if show_plot:
                plt.show()
                plt.clf()

            # Append
            clf_per_lvlgen.append(clf)
            y_per_lvlgen.append(y)
            sco_per_lvlgen.append(sco)
            y_APOE_per_lvlgen.append(y_APOE)
            sco_APOE_per_lvlgen.APPEND(sco_APOE)

        cmat = comparison_matrix(y_per_lvlgen)
        print(f"Comparison matrix of the features.")
        print("The features are " + str(feat_types))
        print(cmat)
        # FILL IN VISUALIZATION OF cmat

        # Append
        clf_per_lvl.append(clf_per_lvlgen)
        y_per_lvl.append(y_per_lvlgen)
        sco_per_lvl.append(sco_per_lvlgen)
        y_APOE_per_lvl.append(y_APOE_per_lvlgen)
        sco_APOE_per_lvl.APPEND(sco_APOE_per_lvlgen)

    # Append
    all_clf.append(clf_per_lvl)
    all_y.append(y_per_lvl)
    all_sco.append(sco_per_lvl)
    all_y_APOE.append(y_APOE_per_lvl)
    all_sco_APOE.append(sco_APOE_per_lvl)

# Save classifiers?
