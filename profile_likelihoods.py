"""Code for profile likelihood plots in figures S5 and S6."""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed

import fitting

RISK_PARAMETER_SET = [
    ("beta", (0, 0)),
    ("beta", (0, 1)),
    ("beta", (1, 0)),
    ("beta", (1, 1)),
]

RISK_PARAMETER_NAMES = ["{0}_{1}{2}".format(param_type, param_idx[0], param_idx[1])
                        for param_type, param_idx in RISK_PARAMETER_SET]

RISK_COLUMN_MAP = {param: i for i, param in enumerate(RISK_PARAMETER_NAMES)}

SPACE_PARAMETER_SET = [
    ("sigma", (0, 0)),
    ("sigma", (1, 0)),
    ("sigma", (1, 1)),
    ("sigma", (2, 1)),
    ("sigma", (2, 2)),
    ("rho", (0, 1)),
    ("rho", (1, 0)),
    ("rho", (1, 1))
]

SPACE_PARAMETER_NAMES = ["{0}_{1}{2}".format(param_type, param_idx[0], param_idx[1])
                         for param_type, param_idx in SPACE_PARAMETER_SET]

SPACE_COLUMN_MAP = {param: i for i, param in enumerate(SPACE_PARAMETER_NAMES)}

def profile_likelihood(param, value, bounds, space=True):
    """Calculate profile likelihood value.

    param: (type, index) with type sigma/rho/beta
    """

    fitter = fitting.Fitter.from_file(os.path.join("Data", "Fit", "FitData.pickle"))
    fitter = fitter.get_fit(space=space)

    param_type, param_idx = param

    if space:
        start = {
            'sigma': fitter.data['sigma'],
            'rho': fitter.data['rho']
        }
    else:
        start = {'beta': fitter.data['beta']}

    start[param_type][param_idx] = value
    bounds[param_type][param_idx] = (value, value)

    fitter.fit(bounds=bounds, start=start)
    loglik_val = fitter._calc_likelihood()
    print(("{0}_{1}{2}".format(param_type, param_idx[0], param_idx[1]), value, loglik_val))

    if space:
        return [*fitter.data['sigma'].flatten()[[0, 3, 4, 7, 8]],
                *fitter.data['rho'].flatten()[[1, 2, 3]], loglik_val]

    return [*fitter.data['beta'].flatten(), loglik_val]

def refine_profile_likelihood(param, existing_data, confidence_thresh, bounds, num_points=5,
                              space=True):
    """Add additional points to PL curve within confidence interval threshold.

    param: (type, index) with type sigma/rho
    Note: must have a unique maximum above the threshold.
    """

    param_type, param_idx = param
    param_name = "{0}_{1}{2}".format(param_type, param_idx[0], param_idx[1])

    if space:
        column_map = SPACE_COLUMN_MAP
    else:
        column_map = RISK_COLUMN_MAP

    existing_data = np.array(existing_data)
    existing_data = existing_data[existing_data[:, column_map[param_name]].argsort()]

    start_index = 0
    while existing_data[start_index+1, -1] < confidence_thresh:
        start_index += 1

    end_index = len(existing_data) - 1
    while existing_data[end_index-1, -1] < confidence_thresh:
        end_index -= 1

    extra_values = []
    n_per_group = [len(x) for x in np.array_split(range(num_points), end_index-start_index)]
    for index, n_vals in zip(range(start_index, end_index), n_per_group):
        extra_values += np.geomspace(existing_data[index, column_map[param_name]],
                                     existing_data[1+index, column_map[param_name]],
                                     endpoint=False, num=n_vals+1)[1:].tolist()

    extra_data = Parallel(n_jobs=6)(delayed(profile_likelihood)(
        (param_type, param_idx), value, bounds, space) for value in extra_values)

    extra_data = np.array(extra_data)
    all_data = np.vstack((existing_data, extra_data))
    all_data = all_data[all_data[:, column_map[param_name]].argsort()]

    return all_data

def plot_profile_likelihood(param_list=None, space=True):
    """Make plot of profile likelihoods, showing 95% CI threshold."""

    fitter = fitting.Fitter.from_file(os.path.join("Data", "Fit", "FitData.pickle"))
    fit = fitter.get_fit(space=space)
    mle_loglik = fit._calc_likelihood()

    plt.style.use("seaborn-whitegrid")

    label_map = {
        "sigma_00": r"$\tilde{\sigma}_{AA}$",
        "sigma_10": r"$\tilde{\sigma}_{BA}$",
        "sigma_11": r"$\tilde{\sigma}_{BB}$",
        "sigma_21": r"$\tilde{\sigma}_{CB}$",
        "sigma_22": r"$\tilde{\sigma}_{CC}$",
        "rho_01": r"$\tilde{\rho}^{HL}$",
        "rho_10": r"$\tilde{\rho}^{LH}$",
        "rho_11": r"$\tilde{\rho}^{LL}$",
        "beta_00": r"$\hat{\rho}^{HH}$",
        "beta_01": r"$\hat{\rho}^{HL}$",
        "beta_10": r"$\hat{\rho}^{LH}$",
        "beta_11": r"$\hat{\rho}^{LL}$",
    }

    if space:
        column_map = SPACE_COLUMN_MAP
        parameter_set = SPACE_PARAMETER_SET
        data = np.load(os.path.join("Data", "SpaceProfileLikelihoods.npz"))
        if param_list is None:
            param_list = SPACE_PARAMETER_NAMES
    else:
        column_map = RISK_COLUMN_MAP
        parameter_set = RISK_PARAMETER_SET
        data = np.load(os.path.join("Data", "RiskProfileLikelihoods.npz"))
        if param_list is None:
            param_list = RISK_PARAMETER_NAMES

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, param in enumerate(param_list):
        param_data = data[param]
        ax.plot(param_data[:, column_map[param]], param_data[:, -1], label=label_map[param],
                color="C{}".format(i))

        param_type, param_idx = parameter_set[column_map[param]]
        point, = ax.plot(fit.data[param_type][param_idx], mle_loglik, "kx", markersize=3)

    point.set_label("MLE")
    ax.axhline(mle_loglik-2*stats.chi2.ppf(q=0.95, df=8), ls="dotted", color="k", alpha=0.5,
               label="95% CI")
    lgd = ax.legend(ncol=1, frameon=True, framealpha=1, bbox_to_anchor=(1.01, 0.5), loc=6)
    ax.set_xscale("log")
    fig.tight_layout()

    return fig, ax, lgd

def make_data():
    """Generate data for risk and space based profile likelihood plots."""

    with open(os.path.join("Data", "data.pickle"), "rb") as infile:
        data = pickle.load(infile)

    fitter = fitting.Fitter.from_file(os.path.join("Data", "Fit", "FitData.pickle"))
    data['setup']['bounds']['sigma'] = data['setup']['bounds']['sigma'].astype(float)
    data['setup']['bounds']['rho'] = data['setup']['bounds']['rho'].astype(float)
    data['setup']['bounds']['beta'] = data['setup']['bounds']['beta'].astype(float)
    npoints = 21
    npoints_fine = 51

    # Risk based profile likelihoods
    risk_fit = fitter.get_fit(space=False)
    mle_loglik = risk_fit._calc_likelihood()
    risk_data = {}

    for param_type, param_idx in RISK_PARAMETER_SET:
        num_before_mle = int(
            (npoints-1) * (6 - abs(np.log10(risk_fit.data[param_type][param_idx]))) / 6)

        param_vals = np.geomspace(10**-6, risk_fit.data[param_type][param_idx], endpoint=False,
                                  num=num_before_mle)

        param_vals = np.concatenate((param_vals, np.geomspace(
            risk_fit.data[param_type][param_idx], 1, endpoint=True, num=1+npoints-num_before_mle)))

        pl_data = Parallel(n_jobs=6)(delayed(profile_likelihood)(
            (param_type, param_idx), value, data['setup']['bounds'], False) for value in param_vals)

        risk_data["{0}_{1}{2}".format(param_type, param_idx[0], param_idx[1])] = np.array(pl_data)
        np.savez(os.path.join("Data", "RiskProfileLikelihoods"), **risk_data)

        pl_data_with_extra = refine_profile_likelihood(
            (param_type, param_idx), pl_data, mle_loglik-2*stats.chi2.ppf(q=0.95, df=8),
            data['setup']['bounds'], num_points=npoints_fine, space=False)

        risk_data["{0}_{1}{2}".format(param_type, param_idx[0], param_idx[1])] = np.array(
            pl_data_with_extra)
        np.savez(os.path.join("Data", "RiskProfileLikelihoods"), **risk_data)

    # Space based profile likelihoods
    space_fit = fitter.get_fit(space=True)
    mle_loglik = space_fit._calc_likelihood()
    space_data = {}

    for param_type, param_idx in SPACE_PARAMETER_SET:
        num_before_mle = int(
            (npoints-1) * (6 - abs(np.log10(space_fit.data[param_type][param_idx]))) / 6)

        param_vals = np.geomspace(10**-6, space_fit.data[param_type][param_idx], endpoint=False,
                                  num=num_before_mle)

        param_vals = np.concatenate((param_vals, np.geomspace(
            space_fit.data[param_type][param_idx], 1, endpoint=True, num=1+npoints-num_before_mle)))

        pl_data = Parallel(n_jobs=6)(delayed(profile_likelihood)(
            (param_type, param_idx), value, data['setup']['bounds'], True) for value in param_vals)

        space_data["{0}_{1}{2}".format(param_type, param_idx[0], param_idx[1])] = np.array(pl_data)
        np.savez(os.path.join("Data", "SpaceProfileLikelihoods"), **space_data)

        pl_data_with_extra = refine_profile_likelihood(
            (param_type, param_idx), pl_data, mle_loglik-2*stats.chi2.ppf(q=0.95, df=8),
            data['setup']['bounds'], num_points=npoints_fine, space=True)

        space_data["{0}_{1}{2}".format(param_type, param_idx[0], param_idx[1])] = np.array(
            pl_data_with_extra)
        np.savez(os.path.join("Data", "SpaceProfileLikelihoods"), **space_data)

def make_plots():
    """Generate figures S5 and S6"""

    fitter = fitting.Fitter.from_file(os.path.join("Data", "Fit", "FitData.pickle"))

    # Risk based plot
    risk_fit = fitter.get_fit(space=False)
    mle_loglik = risk_fit._calc_likelihood()

    fig, ax, lgd = plot_profile_likelihood(param_list=RISK_PARAMETER_NAMES, space=False)
    ax.set_ylim([mle_loglik-6*stats.chi2.ppf(q=0.95, df=8), mle_loglik+5])
    ax.set_xlim([10**-3.5, 10**-1.5])
    ax.set_ylabel("Log Likelihood")
    ax.set_xlabel("Parameter Value")
    ax.ticklabel_format(style='sci', axis='y', useMathText=True)
    fig.savefig(os.path.join("Figures", "SuppFig5.pdf"), dpi=600, additional_artists=[lgd],
                bbox_inches="tight")

    # Space based plot
    space_fit = fitter.get_fit(space=True)
    mle_loglik = space_fit._calc_likelihood()

    fig, ax, lgd = plot_profile_likelihood(param_list=SPACE_PARAMETER_NAMES, space=True)
    ax.set_ylim([mle_loglik-6*stats.chi2.ppf(q=0.95, df=8), mle_loglik+5])
    ax.set_xlim([10**-4, 1])
    ax.set_ylabel("Log Likelihood")
    ax.set_xlabel("Parameter Value")
    ax.ticklabel_format(style='sci', axis='y', useMathText=True)
    fig.savefig(os.path.join("Figures", "SuppFig6.pdf"), dpi=600, additional_artists=['lgd'],
                bbox_inches="tight")

if __name__ == "__main__":
    make_data()
    make_plots()
