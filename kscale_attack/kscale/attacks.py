"""PDF A(c,beta) coverage attacks. Candidate uniqueness is NOT reconstruction.

References: k-scale-formal.pdf, Definitions 1--6 and Proposition 1.
Gamma is structural co-membership, not geometric coverage of raw unions.
No equiprobable posterior or general reconstruction theorem is assumed.
"""
from functools import lru_cache
from itertools import product
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, eye
from ._io import read_metadata, require_dataset, new_stage, write_metadata, positive_integer
from .data import load_S_i
from .merge import BOUND_COLUMNS


class CoverageIndex:
    """Exact box containment over ALL published records, including boundaries.

    Stores all boxes in RAM. Each uncached point scans boxes in bounded chunks;
    this is a correctness baseline, not a production spatial index.
    """
    def __init__(self, release, U, chunk_size=100_000, cache_size=10_000):
        self.U = U
        self.chunk_size = positive_integer(chunk_size, 'chunk_size')
        cache_size = positive_integer(cache_size, 'cache_size')
        release = Path(release)
        manifest = pd.read_parquet(release / 'manifest.parquet').sort_values('i')
        if manifest.i.tolist() != list(range(U)):
            raise ValueError('Generalized manifest must contain IDs 0..U-1 exactly once.')
        blocks, owners = [], []
        for row in manifest.itertuples():
            G = pd.read_parquet(release / row.path, columns=BOUND_COLUMNS).to_numpy(float)
            if not len(G) or not np.isfinite(G).all() or np.any(G[:, :3] > G[:, 3:]):
                raise ValueError(f'Invalid boxes in record {row.i}.')
            blocks.append(G)
            owners.append(np.full(len(G), int(row.i), dtype=np.int64))
        self.boxes = np.concatenate(blocks)
        self.owners = np.concatenate(owners)
        self.point_candidates = lru_cache(maxsize=cache_size)(self._point_candidates)

    def _point_candidates(self, point):
        p = np.asarray(point, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('Point must be finite (x,y,timestamp).')
        found = set()
        for start in range(0, len(self.boxes), self.chunk_size):
            stop = start + self.chunk_size
            G = self.boxes[start:stop]
            mask = np.all(G[:, :3] <= p, axis=1) & np.all(p <= G[:, 3:], axis=1)
            found.update(self.owners[start:stop][mask].tolist())
        return frozenset(found)

    def candidates(self, points):
        result = set(range(self.U))
        for point in points:
            result.intersection_update(self.point_candidates(tuple(point)))
            if not result:
                break
        return result


def _wilson(successes, total):
    if total == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = successes / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total)) / den
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / den
    return center-half, center+half


def summarize_attacks(trials):
    """Monte Carlo intervals conditional on this fixed publication/population."""
    keys = ['k', 'K', 'victim_mode', 'point_mode', 'c', 'beta']
    rows = []
    for key, group in trials.groupby(keys, sort=False):
        J, gamma = group.J.to_numpy(), group.gamma.to_numpy()
        k = int(key[0])
        row = dict(zip(keys, key))
        row.update(n_trials=len(group), mean_J=J.mean(), median_J=np.median(J),
                   mean_gamma=gamma.mean(), mean_accidental=(J-gamma).mean())
        events = {
            'zero': (J == 0, np.ones(len(J), bool)),
            'unique': (J == 1, np.ones(len(J), bool)),
            'collapse': ((J > 0) & (J < k), np.ones(len(J), bool)),
            'collapse_given_nonempty': (J < k, J > 0),
            'structural_overlap': (gamma > 0, np.ones(len(J), bool)),
            'structural_collapse': ((gamma > 0) & (gamma < k), np.ones(len(J), bool)),
        }
        for name, (event, eligible) in events.items():
            n = int(eligible.sum())
            s = int((event & eligible).sum())
            lo, hi = _wilson(s, n)
            row.update({f'p_{name}': s/n if n else np.nan,
                        f'{name}_n': n, f'{name}_successes': s,
                        f'{name}_ci_low': lo, f'{name}_ci_high': hi})
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_attacks(Sigma, Sigma_tilde, output_path, *, c_values=(1,2,3,5),
                     beta_values=(1,2,4,8), trials=100, seed=42,
                     victim_modes=('random','nearby'), point_modes=('random','consecutive'),
                     spatial_scale_m=1000., time_scale_s=3600.,
                     chunk_size=100_000, cache_size=10_000, progress=True):
    """Evaluate nested PDF A(c,beta) attacks without rerunning anonymization.

    beta='all' reveals complete trajectories. Numeric beta uses a fixed cohort
    with N_i >= max numeric beta, even for 'all', to avoid changing populations.
    Too-small cohorts raise errors; points are never silently truncated.
    nearby = nearest trajectory centroids to a random anchor in scaled (x,y,t).
    It is a proximity proxy, NOT proof of simultaneous physical co-location.
    Same Sigma, seed and sampling parameters reuse attacks across k and K.
    Only boxes are public; A is used exclusively for evaluator diagnostics.
    N_i_segment, L, C_G_i, record identity links and parameter-based attacks
    are deliberately outside this geometric attacker model.
    """
    trials = positive_integer(trials, 'trials')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    c_values = sorted(set(positive_integer(c, 'c') for c in c_values))
    numeric = sorted(set(positive_integer(b, 'beta') for b in beta_values if b != 'all'))
    betas = numeric + (['all'] if 'all' in beta_values else [])
    if not c_values or not betas:
        raise ValueError('c_values and beta_values cannot be empty.')
    victim_modes, point_modes = tuple(victim_modes), tuple(point_modes)
    if (not victim_modes or len(set(victim_modes)) != len(victim_modes)
            or not set(victim_modes) <= {'random','nearby'}):
        raise ValueError('victim_modes: unique random/nearby values required.')
    if (not point_modes or len(set(point_modes)) != len(point_modes)
            or not set(point_modes) <= {'random','consecutive'}):
        raise ValueError('point_modes: unique random/consecutive values required.')
    if not np.isfinite([spatial_scale_m,time_scale_s]).all() or min(spatial_scale_m,time_scale_s) <= 0:
        raise ValueError('Proximity scales must be finite and positive.')
    meta = read_metadata(Sigma_tilde)
    require_dataset(Sigma, meta)
    U, k = int(meta['U']), int(meta['k'])
    raw_manifest = pd.read_parquet(Path(Sigma)/'manifest.parquet').sort_values('i')
    if raw_manifest.i.tolist() != list(range(U)):
        raise ValueError('Raw manifest IDs do not match release.')
    A = load_npz(Path(Sigma_tilde)/'A.npz').tocsr().astype(bool)
    if A.shape != (U,U) or A.diagonal().any():
        raise ValueError('Expected U x U picking matrix with zero diagonal.')
    members = A.maximum(eye(U, format='csr', dtype=bool)).tocsr()
    hosts = members.tocsc()
    coverage = np.diff(hosts.indptr)
    if np.any(coverage < k):
        raise ValueError('Release violates coverage >= k.')
    eligible = raw_manifest.loc[raw_manifest.N_i >= max(numeric, default=1),'i'].to_numpy(int)
    if len(eligible) < max(c_values):
        raise ValueError(f'Only {len(eligible)} users have enough points for this grid; reduce beta or c.')
    index = CoverageIndex(Sigma_tilde, U, chunk_size, cache_size)

    @lru_cache(maxsize=256)
    def points(i):
        P = load_S_i(Sigma, int(i))[['x','y','timestamp']].to_numpy(float)
        if len(P) != int(raw_manifest.iloc[i].N_i) or not np.isfinite(P).all():
            raise ValueError(f'Invalid raw trajectory {i}.')
        return P[np.argsort(P[:,2], kind='stable')]

    centers = None
    if 'nearby' in victim_modes:
        centers = np.array([points(int(i)).mean(axis=0) for i in eligible])
        centers /= np.array([spatial_scale_m,spatial_scale_m,time_scale_s])
    rows = []
    with new_stage(output_path) as output:
        for vm in victim_modes:
            vm_id = {'random':0, 'nearby':1}[vm]
            for trial in range(trials):
                rng = np.random.default_rng(np.random.SeedSequence([seed, vm_id, trial]))
                if vm == 'random':
                    victims = rng.choice(eligible, max(c_values), replace=False)
                else:
                    anchor = int(rng.integers(len(eligible)))
                    distances = ((centers-centers[anchor])**2).sum(axis=1)
                    others = np.flatnonzero(np.arange(len(eligible)) != anchor)
                    ranked = others[np.lexsort((eligible[others], distances[others]))]
                    victims = eligible[np.r_[anchor,ranked[:max(c_values)-1]]]
                for pm in point_modes:
                    pm_id = {'random':0, 'consecutive':1}[pm]
                    individual = {}
                    selections = {}
                    for i in victims:
                        i = int(i)
                        P = points(i)
                        prng = np.random.default_rng(np.random.SeedSequence([seed, vm_id, trial, pm_id, i]))
                        if pm == 'random':
                            order = prng.permutation(len(P))
                        else:
                            width = max(numeric, default=1)
                            start = int(prng.integers(len(P)-width+1))
                            order = np.arange(start,start+width)
                        H = set(hosts.indices[hosts.indptr[i]:hosts.indptr[i+1]])
                        for beta in betas:
                            selected = np.arange(len(P)) if beta == 'all' else order[:beta]
                            C = index.candidates(P[selected])
                            if not H <= C:
                                raise ValueError(f'Containment violation for S_{i}, beta={beta}.')
                            individual[i,beta] = C
                            selections[i,beta] = selected.tolist()
                    for beta in betas:
                        joint, structural = set(range(U)), set(range(U))
                        for c, i in enumerate(victims, 1):
                            i = int(i)
                            joint.intersection_update(individual[i,beta])
                            structural.intersection_update(hosts.indices[hosts.indptr[i]:hosts.indptr[i+1]])
                            if c in c_values:
                                chosen = [int(x) for x in victims[:c]]
                                rows.append(dict(k=k,K=int(meta['K']),victim_mode=vm,point_mode=pm,
                                    trial=trial,seed=int(seed),c=c,beta=str(beta),gamma=len(structural),J=len(joint),
                                    victims=json.dumps(chosen),
                                    point_indices=json.dumps({i: selections[i,beta] for i in chosen}),
                                    known_points=sum(len(selections[i,beta]) for i in chosen)))
                if progress and (trial+1) % max(1,trials//10) == 0:
                    print(f'{vm}: {trial+1}/{trials}', flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(output/'attacks.csv', index=False)
        summary = summarize_attacks(frame)
        summary.to_csv(output/'attack_summary.csv', index=False)
        pd.DataFrame(dict(i=np.arange(U),coverage=coverage,
                          member_count=np.diff(members.indptr))).to_csv(output/'structure.csv',index=False)
        write_metadata(output, stage='attacks',dataset_id=meta['dataset_id'],U=U,k=k,K=int(meta['K']),
            c_values=c_values,beta_values=betas,trials=trials,seed=int(seed),
            victim_modes=victim_modes,point_modes=point_modes,eligible_users=len(eligible),
            cohort_min_points=max(numeric,default=1),spatial_scale_m=spatial_scale_m,
            time_scale_s=time_scale_s,raw_path=str(Path(Sigma).resolve()),
            release_path=str(Path(Sigma_tilde).resolve()),
            observed_information='union-of-box containment only; no membership or identity labels',
            index_boxes=len(index.boxes),index_bytes=index.boxes.nbytes+index.owners.nbytes,
            confidence_intervals='Wilson 95%; conditional on fixed release, not population uncertainty')
    return Path(output_path).resolve()


def plot_attack_heatmap(summary, *, victim_mode='random', point_mode='random',
                        metric='p_collapse', ax=None):
    """Plot one release's summary; use separate panels for different k/K."""
    import matplotlib.pyplot as plt
    table = pd.read_csv(summary) if not isinstance(summary,pd.DataFrame) else summary.copy()
    table = table[(table.victim_mode == victim_mode) & (table.point_mode == point_mode)].copy()
    if table.empty or table[['k','K']].drop_duplicates().shape[0] != 1:
        raise ValueError('Select a nonempty single-release summary.')
    table['beta'] = table.beta.astype(str)
    order = sorted(table.beta.unique(),key=lambda b: math.inf if b=='all' else int(b))
    grid = table.pivot(index='c',columns='beta',values=metric).reindex(columns=order).sort_index()
    if ax is None:
        _,ax = plt.subplots()
    image = ax.imshow(grid.to_numpy(),vmin=0,vmax=1,aspect='auto',origin='lower')
    ax.set_xticks(range(len(grid.columns)),grid.columns)
    ax.set_yticks(range(len(grid.index)),grid.index)
    ax.set(xlabel='Known points per user (beta)',ylabel='Known users (c)',
           title=f'{metric}: {victim_mode}, {point_mode}')
    ax.figure.colorbar(image,ax=ax,label='Probability')
    return ax


def single_point_compatible_values(known_points, unknown_count, published_box, grid,
                                   *, max_worlds=1_000_000):
    """Exact finite-grid toy: one point per member, only min/max box public.

    C is the FIRST labeled unknown. Enumerates all assignments of other unknowns.
    Returns feasible C points, NOT a posterior probability. Does not implement
    the complete k-scale selection mechanism or multi-sample gen inversion.
    """
    n = positive_integer(unknown_count,'unknown_count')
    grid = np.asarray(grid,dtype=float)
    if grid.ndim == 1:
        grid = grid[:,None]
    if grid.ndim != 2 or not len(grid) or not np.isfinite(grid).all():
        raise ValueError('grid must be a finite nonempty point array.')
    grid = np.unique(grid,axis=0)
    d = grid.shape[1]
    box = np.asarray(published_box,dtype=float)
    known = np.asarray(known_points,dtype=float).reshape(-1,d)
    if box.shape != (2,d) or not np.isfinite(box).all() or (box[0]>box[1]).any() or not np.isfinite(known).all():
        raise ValueError('Invalid known points or published box.')
    grid = grid[np.all((grid>=box[0]) & (grid<=box[1]),axis=1)]
    if len(grid)**n > positive_integer(max_worlds,'max_worlds'):
        raise ValueError('Toy enumeration exceeds max_worlds; reduce grid or unknown count.')
    feasible = set()
    for assignment in product(range(len(grid)), repeat=n):
        points = np.concatenate([known,grid[list(assignment)]])
        if np.array_equal(points.min(axis=0),box[0]) and np.array_equal(points.max(axis=0),box[1]):
            feasible.add(assignment[0])
    return grid[sorted(feasible)]
