"""
Hybrid decoder: impose the classical Springer HSMM state-duration model + cyclic-order
constraint on top of the DEEP model's per-sample posteriors.

Mechanism (reuses the port's exact duration-Viterbi, no reimplementation):
the port's viterbi computes observation_probs[t, state] = pihat[t] * Po_correction / pi.
We feed `models` whose predict_proba returns the deep posteriors and a constant Po_correction,
so observation_probs == deep posteriors (up to a global constant that cancels in argmax).
Everything downstream — physiological duration priors, the S1->systole->S2->diastole cycle
transition matrix, and the duration-dependent Viterbi backtrack — is the classical machinery.

This is the "classical priors rescue deep" hybrid: deep supplies strong emissions, the HSMM
duration structure repairs the physiologically-impossible fragmentation deep produces under
noise / murmur.
"""
import sys
import numpy as np
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/hsmm")
from viterbi import viterbi_segment
from heart_rate import get_heart_rate
from duration_distributions import DataDistribution


class _PosteriorModel:
    """Stand-in for a per-state logistic model: predict_proba returns the deep posterior.
    The port does pihat = predict_proba(x)[:, ::-1] and reads pihat[:,1], i.e. column 0 of
    our output -> put the deep posterior in column 0."""
    def __init__(self, post_col):
        self.post = np.clip(post_col, 1e-9, 1.0)

    def predict_proba(self, X):
        return np.stack([self.post, 1.0 - self.post], axis=1)


def hybrid_decode_with_hr(deep_post_feat, heart_rate, systolic_time, feature_frequency=50,
                          distribution=None):
    """Same as hybrid_decode but with heart rate / systolic interval precomputed (so this is
    a pure-CPU function safe to fan out across a multiprocessing Pool)."""
    if distribution is None:
        distribution = DataDistribution(features_frequency=feature_frequency)
    T = deep_post_feat.shape[0]
    obs_seq = np.zeros((T, 1))                        # dummy; only its length is used
    models = [_PosteriorModel(deep_post_feat[:, i]) for i in range(4)]
    total_obs_distribution = [np.zeros(1), np.eye(1)]  # -> constant Po_correction (cancels)
    _, _, states = viterbi_segment(obs_seq, models, total_obs_distribution, distribution,
                                   float(heart_rate), float(systolic_time), feature_frequency)
    return np.asarray(states).astype(int)


def estimate_hr(signal, sampling_frequency, min_hr=60, max_hr=200):
    hrs, stis = get_heart_rate(signal, sampling_frequency, min_heart_rate=min_hr, max_heart_rate=max_hr)
    return float(hrs[0]), float(stis[0])


def hybrid_decode(deep_post_feat, raw_signal, sampling_frequency=4000, feature_frequency=50,
                  distribution=None, min_hr=60, max_hr=200):
    """deep_post_feat : [T,4] posteriors (S1,sys,S2,dia) at feature_frequency.
    raw_signal used only for heart-rate estimation. Returns states (1..4) at feature_frequency."""
    hr, sti = estimate_hr(raw_signal, sampling_frequency, min_hr, max_hr)
    return hybrid_decode_with_hr(deep_post_feat, hr, sti, feature_frequency, distribution)


_ORDER = {1: 2, 2: 3, 3: 4, 4: 1}  # valid cardiac cycle S1->systole->S2->diastole


def cycle_violation_rate(state_seq):
    """Fraction of state->state transitions in a per-sample sequence that break the cardiac
    cycle. A label-free degradation signal: clean deep output is ~plausible (<5%), noisy/
    fragmented deep output is highly implausible. CirCor GT always follows the cycle, so any
    violation in a prediction is an error."""
    s = np.asarray(state_seq); s = s[s > 0]
    if len(s) < 2:
        return 0.0
    chg = np.where(np.diff(s) != 0)[0]
    trans = [(s[i], s[i + 1]) for i in chg]
    trans = [(a, b) for a, b in trans if a in _ORDER and b in (1, 2, 3, 4)]
    if not trans:
        return 0.0
    return sum(1 for a, b in trans if _ORDER[a] != b) / len(trans)


def gated_hybrid(deep_argmax_feat, deep_post_feat, raw_signal, tau,
                 sampling_frequency=4000, feature_frequency=50, distribution=None):
    """Plausibility-gated hybrid (the final method): keep the deep argmax when it is
    physiologically plausible; invoke the classical duration-Viterbi only when the deep
    output's cycle-violation rate exceeds tau (i.e. deep has degraded). tau is calibrated on
    the clean-data violation distribution (e.g. its 95th percentile) — no test-noise peeking.
    Pareto-dominant: ~=deep on clean, ~=always-on hybrid under heavy noise."""
    if cycle_violation_rate(deep_argmax_feat) <= tau:
        return np.asarray(deep_argmax_feat).astype(int)
    return hybrid_decode(deep_post_feat, raw_signal, sampling_frequency, feature_frequency, distribution)
