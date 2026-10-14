"""HiGHS flow repair within equal-value records of the same parent and bucket."""
from collections import Counter
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def grouped(values):
    order = np.argsort(values, kind='stable')
    return {int(values[ids[0]]):ids for ids in np.split(order, np.flatnonzero(np.diff(values[order]))+1)}


def distribution(edges, rows):
    n = len(rows)
    levels = np.array([r["level"] for r in rows])
    q = np.bincount(edges[:,0], minlength=n)
    ref = np.bincount(edges[:,1], minlength=n)
    report = {}
    for level in ['all', *sorted(set(levels))]:
        keep = np.ones(n, dtype=bool) if level=='all' else levels==level
        report[level] = {}
        for role, data in [('query',q),('reference',ref),('total',q+ref)]:
            d = data[keep]
            values = {f'p{p}':float(np.percentile(d,p)) for p in [1,5,10,25,50,75,90,99]}
            values.update(records=len(d), zero=int(sum(d==0)), minimum=int(min(d)), maximum=int(max(d)),
                          mean=float(d.mean()), std=float(d.std()), sum_squares=int(d@d),
                          histogram={str(k):int(v) for k,v in sorted(Counter(d).items())})
            report[level][role] = values
    imbalance = np.bincount(edges[:,0], weights=2*edges[:,2]-1, minlength=n)
    report['near_far'] = {'near_pairs':int(sum(edges[:,2])), 'far_pairs':int(sum(1-edges[:,2])),
                          'mean_absolute_query_imbalance':float(np.abs(imbalance).mean()),
                          'mean_fractional_query_imbalance':float((np.abs(imbalance[q>0])/q[q>0]).mean())}
    return report


def solve_group(ids, selected, role, edges, degrees, limits, parent, solver_time_limit=5):
    n = len(limits)
    other = degrees[1-role][ids]
    old = degrees[role][ids]
    count, size = len(selected), len(ids)
    partner, multiplicity = np.unique(parent[edges[selected,1]], return_counts=True) if role==0 else ([],[])
    p = len(partner)
    cap = min(int(limits[ids[0]]), p) if role==0 else int(limits[ids[0]])
    xsize = p*size
    units = np.tile(np.arange(1,cap+1), size)
    owners = np.repeat(np.arange(size), cap)
    c = np.r_[np.zeros(xsize), 2*(other[owners]+units)-1]
    low = np.zeros(len(c))
    low[xsize:] = (units==1) & ((old[owners]>0) | (other[owners]==0))
    ycols = np.arange(xsize,len(c))
    if role==0:
        xcols = np.arange(xsize)
        rr = np.r_[np.repeat(np.arange(p),size),p+np.tile(np.arange(size),p),p+owners]
        cc = np.r_[xcols,xcols,ycols]
        matrix = coo_matrix((np.r_[np.ones(2*xsize),-np.ones(len(ycols))],(rr,cc)),shape=(p+size,len(c))).tocsr()
        rhs = np.r_[multiplicity,np.zeros(size)]
    else:
        matrix = coo_matrix((np.ones(len(c)),(np.zeros(len(c),dtype=int),np.arange(len(c)))),shape=(1,len(c))).tocsr()
        rhs = [count]
    result = linprog(c, A_eq=matrix, b_eq=rhs, bounds=np.c_[low,np.ones(len(c))],
                     method='highs', options={'time_limit':solver_time_limit})
    if result.status == 2:
        # Sparse groups cannot always cover every unused record; preserve existing coverage.
        low[xsize:] = (units==1) & (old[owners]>0)
        result = linprog(c, A_eq=matrix, b_eq=rhs, bounds=np.c_[low,np.ones(len(c))],
                         method='highs', options={'time_limit':solver_time_limit})
    if not result.success:
        return None, {'status':int(result.status),'message':result.message}
    assert np.max(np.abs(result.x-np.rint(result.x))) < 1e-6
    assert np.max(np.abs(matrix@result.x-rhs)) < 1e-6
    assert np.all(result.x>=low-1e-6) and np.all(result.x<=1+1e-6)
    counts = np.rint(result.x[xsize:].reshape(size,cap).sum(axis=1)).astype(int)
    if tuple(sorted(counts+other)) <= tuple(sorted(old+other)):
        return None, {'status':0,'accepted':False}
    replacement = np.empty(count,dtype=np.int32)
    if role==0:
        assignment = np.rint(result.x[:xsize].reshape(p,size)).astype(bool)
        for j, value in enumerate(partner):
            positions = np.flatnonzero(parent[edges[selected,1]]==value)
            replacement[positions] = ids[assignment[j]]
    else:
        replacement[:] = np.repeat(ids,counts)
    assert np.array_equal(np.bincount(replacement,minlength=n)[ids],counts)
    return replacement, {'status':0,'accepted':True,'squared_degree_reduction':int(np.sum((old+other)**2)-np.sum((counts+other)**2))}



def balance_group(ids, selected, edges, degree, parent, solver_time_limit=5):
    query = edges[selected,0]
    parent_values, pi = np.unique(parent[edges[selected,1]], return_inverse=True)
    qi = np.searchsorted(ids,query)
    qcount = degree[ids]
    near_count = np.bincount(pi,weights=edges[selected,2],minlength=len(parent_values))
    owners = np.repeat(np.arange(len(ids)),qcount)
    units = np.concatenate([np.arange(1,count+1) for count in qcount])
    m, p = len(selected), len(parent_values)
    xcols, ycols = np.arange(m), np.arange(m,m+len(units))
    rr, cc = np.r_[pi,p+qi,p+owners], np.r_[xcols,xcols,ycols]
    matrix = coo_matrix((np.r_[np.ones(2*m),-np.ones(len(units))],(rr,cc)),
                        shape=(p+len(ids),m+len(units))).tocsr()
    rhs = np.r_[near_count,np.zeros(len(ids))]
    costs = np.r_[np.zeros(m),8*units-4*qcount[owners]-4]
    result = linprog(costs,A_eq=matrix,b_eq=rhs,bounds=(0,1),method='highs',options={'time_limit':solver_time_limit})
    if not result.success:
        return {'status':int(result.status),'changed':False}
    assert np.max(np.abs(result.x-np.rint(result.x)))<1e-6
    assert np.max(np.abs(matrix@result.x-rhs))<1e-6
    wanted = result.x[:m]>.5
    old_near = np.bincount(qi,weights=edges[selected,2],minlength=len(ids))
    new_near = np.bincount(qi,weights=wanted,minlength=len(ids))
    reduction = int(np.sum((2*old_near-qcount)**2)-np.sum((2*new_near-qcount)**2))
    if reduction<=0:
        return {'status':0,'changed':False}
    replacement = query.copy()
    for j in range(p):
        for label in [0,1]:
            old_slots = np.flatnonzero((pi==j)&(edges[selected,2]==label))
            new_slots = np.flatnonzero((pi==j)&(wanted==label))
            assert len(old_slots)==len(new_slots)
            replacement[old_slots] = query[new_slots]
    assert np.array_equal(np.sort(replacement),np.sort(query))
    edges[selected,0] = replacement
    return {'status':0,'changed':True,'squared_imbalance_reduction':reduction}


def audit(edges, original, limits, parent, bucket, group, parent_cap):
    n, parent_count = len(limits), int(max(parent)) + 1
    assert edges.shape == original.shape
    assert np.array_equal(group[edges[:,:2]],group[original[:,:2]])
    assert np.array_equal(edges[:,2], original[:,2])
    assert np.all(bucket[edges[:,0]]==bucket[edges[:,1]])
    assert np.all(parent[edges[:,0]]!=parent[edges[:,1]])
    unique = edges[:,0].astype(np.int64)*parent_count+parent[edges[:,1]]
    assert len(np.unique(unique))==len(edges)
    for role in [0,1]:
        d = np.bincount(edges[:,role],minlength=n)
        before = np.bincount(original[:,role],minlength=n)
        assert np.all(d<=limits)
        assert np.all(d[before>0]>=1)
        usage = np.bincount(parent[edges[:,role]],minlength=parent_count)
        assert max(usage)<=parent_cap
        assert np.array_equal(usage,np.bincount(parent[original[:,role]],minlength=parent_count))
    total = np.bincount(edges[:,:2].ravel(),minlength=n)
    initial = np.bincount(original[:,:2].ravel(),minlength=n)
    assert tuple(sorted(total))>=tuple(sorted(initial))
    return {'pairs':len(edges),'all_constraints_passed':True,'minimum_total_degree':int(min(total))}


def repair_degrees(edges, limits, parent, group, solver_time_limit=5):
    n = len(limits)
    degrees = [np.bincount(edges[:, role], minlength=n) for role in (0, 1)]
    records_by_group = grouped(group)
    edge_groups = [grouped(group[edges[:, role]]) for role in (0, 1)]
    passes = []
    for sweep in range(1, 6):
        log = {"sweep": sweep, "solved": 0, "accepted": 0, "failed": 0, "squared_degree_reduction": 0}
        for role in (0, 1):
            schedule = sorted(edge_groups[role], key=lambda g: (
                int(np.min((degrees[0] + degrees[1])[records_by_group[g]])), g))
            for g in schedule:
                ids, selected = records_by_group[g], edge_groups[role][g]
                if len(ids) < 2 or np.ptp((degrees[0] + degrees[1])[ids]) <= 1:
                    continue
                replacement, info = solve_group(ids, selected, role, edges, degrees, limits, parent, solver_time_limit)
                log["solved"] += 1
                log["failed"] += int(info["status"] != 0)
                if replacement is None:
                    continue
                edges[selected, role] = replacement
                degrees[role][ids] = np.bincount(replacement, minlength=n)[ids]
                log["accepted"] += 1
                log["squared_degree_reduction"] += info["squared_degree_reduction"]
        passes.append(log)
        print("V24 solver", log, flush=True)
        if log["failed"]:
            raise ValueError("V24 degree repair solver failed; refusing a partial repair")
        if log["accepted"] == 0:
            break
    balance = {"solved": 0, "changed": 0, "failed": 0, "squared_imbalance_reduction": 0}
    for g, selected in edge_groups[0].items():
        ids = records_by_group[g]
        if len(ids) < 2 or len(np.unique(edges[selected, 2])) < 2:
            continue
        info = balance_group(ids, selected, edges, degrees[0], parent, solver_time_limit)
        balance["solved"] += 1
        balance["failed"] += int(info["status"] != 0)
        balance["changed"] += int(info["changed"])
        balance["squared_imbalance_reduction"] += info.get("squared_imbalance_reduction", 0)
    if balance["failed"]:
        raise ValueError("V24 near/far repair solver failed; refusing a partial repair")
    for role in (0, 1):
        assert np.array_equal(np.bincount(edges[:, role], minlength=n), degrees[role])
    return passes, balance


def repair(rows, original, caps, parent_cap, solver_time_limit=5, separate_family_caps=False):
    start = perf_counter()
    if not len(original):
        raise ValueError("V24 repair requires nonempty training pairs")
    parent_keys = [(("binary" if r["measurement_kind"] == "binary" else "continuous") + ":" if separate_family_caps else "") + r["parent_id"] for r in rows]
    _, parent = np.unique(parent_keys, return_inverse=True)
    _, bucket = np.unique([r["pair_bucket_key"] for r in rows], return_inverse=True)
    keys = [(r["pair_bucket_key"], r["parent_id"], r["measurement_kind"],
             str(r["category_id"]) if r["measurement_kind"] == "binary" else str(float(r["geometry_value"])))
            for r in rows]
    key_index = {key: i for i, key in enumerate(sorted(set(keys)))}
    group = np.array([key_index[key] for key in keys])
    limits = np.array([caps[r["pair_bucket_key"]]["resolved_degree_cap"] for r in rows])
    audit(original, original, limits, parent, bucket, group, parent_cap)
    edges = original.copy()
    passes, balance = repair_degrees(edges, limits, parent, group, solver_time_limit)
    report = {"parent_cap_per_role": parent_cap, "solver_time_limit_seconds": solver_time_limit, "before": distribution(original, rows),
              "after": distribution(edges, rows), "passes": passes, "near_far_balancing": balance,
              "parent_cap_scope": "binary_or_nonbinary_numeric" if separate_family_caps else "shared",
              "method": "alternating_HiGHS_min_cost_flow_compatible_endpoint_assignments",
              "restriction": "same bucket, parent, and exact geometry value or binary category",
              "objective": "reduce squared combined degrees; accept strict sorted-degree improvement",
              "changed_pair_rows": int(np.sum(np.any(edges[:, :2] != original[:, :2], axis=1))),
              "audit": audit(edges, original, limits, parent, bucket, group, parent_cap),
              "seconds": perf_counter() - start}
    return edges, report
