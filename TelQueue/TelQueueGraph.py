#!/usr/bin/env python3
"""
TelQueueGraph (night-only, 2-week windows, no slew/sep costs)

- Each request has a long visibility window (default 14 days).
- Observations may ONLY run between 21:00 and 05:00 (crossing midnight OK).
- Transitions are instantaneous (no slew / settle modeled here).
- We construct a forward-in-time DAG using earliest feasible starts to avoid cycles.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math



# =========================
# Data model
# =========================

@dataclass
class Request:
    rid: str
    ra_deg: float          ## Kept for future use; not used in this simplified graph
    dec_deg: float         ## Kept for future use; not used in this simplified graph
    window_start: float    ## seconds from t0 (0 = start of horizon)
    window_end: float      ## seconds from t0
    duration: float        ## seconds
    priority: float = 1.0  ## optional weight


# =========================
# Time helpers (night windows)
# =========================

SECONDS_PER_DAY = 24 * 3600

def _day_and_tod(t: float) -> Tuple[int, float]:
    """Return (day_index, time_of_day_seconds) for absolute time t (sec from t0)."""
    day = int(t // SECONDS_PER_DAY)
    tod = t - day * SECONDS_PER_DAY
    return day, tod

def next_night_start_at_or_after(t: float,
                                 duration: float,
                                 night_start_hour: int = 21,
                                 night_end_hour: int = 5) -> float:
    """
    Given a time t, return the earliest start s >= t such that [s, s+duration]
    lies fully within a single night window. A "night" is from 21:00 (day d)
    to 05:00 (day d+1).

    This function ignores request windows; it just respects nightly hours.
    """
    night_start = night_start_hour * 3600
    night_end = night_end_hour * 3600  ## this is "next morning" cutoff

    day, tod = _day_and_tod(t)

    ## Case A: we're in the early morning window [00:00, 05:00)
    if tod < night_end:
        window_end_abs = day * SECONDS_PER_DAY + night_end
        if t + duration <= window_end_abs:
            return t  ## can start immediately
        else:
            return day * SECONDS_PER_DAY + night_start  ## tonight 21:00

    ## Case B: daytime [05:00, 21:00)
    if tod < night_start:
        return day * SECONDS_PER_DAY + night_start

    ## Case C: evening/night [21:00, 24:00)
    window_end_abs = (day + 1) * SECONDS_PER_DAY + night_end
    if t + duration <= window_end_abs:
        return t
    else:
        return (day + 1) * SECONDS_PER_DAY + night_start

def earliest_start_within_request(t: float,
                                  req: Request,
                                  night_start_hour: int = 21,
                                  night_end_hour: int = 5) -> Optional[float]:
    """
    Return the earliest feasible start s >= t that:
      - lies within nightly hours (21:00–05:00), AND
      - fits entirely within req.window_start/end, AND
      - accommodates req.duration.

    If none exists, return None.
    """
    t0 = max(t, req.window_start)
    s = next_night_start_at_or_after(t0, req.duration,
                                     night_start_hour=night_start_hour,
                                     night_end_hour=night_end_hour)

    if s < req.window_start:
        s = next_night_start_at_or_after(req.window_start, req.duration,
                                         night_start_hour=night_start_hour,
                                         night_end_hour=night_end_hour)

    if s + req.duration <= req.window_end:
        return s
    return None


# =========================
# Synthetic request generator
# =========================

def generate_requests(n: int,
                      horizon_days: int = 14,
                      min_duration_sec: int = 300,   ## 5 minutes
                      max_duration_sec: int = 1200,  ## 20 minutes
                      seed: Optional[int] = 42,
                      distribute_across_year: bool = True,
                      year_days: int = 365,
                      randomize_windows: bool = True,
                      shuffle_rids: bool = True) -> List[Request]:
    """
    Create n synthetic requests with 2-week visibility windows.
    
    - If `distribute_across_year` and `randomize_windows` are True (default),
      each window start is drawn **uniformly at random** across the year
      (subject to `year_days - horizon_days`), so months are mixed.
    - If `distribute_across_year` is True and `randomize_windows` is False,
      windows are *evenly spaced* with small jitter (previous behavior).
    - If `shuffle_rids` is True (default), we randomly permute the mapping
      of RID labels (R1..Rn) to the generated windows so RID order carries
      no information about time ordering.
    """
    rng = random.Random(seed)
    window_len = horizon_days * SECONDS_PER_DAY
    
    # First build raw items (without RIDs yet)
    items = []
    for i in range(n):
        dur = rng.randint(min_duration_sec, max_duration_sec)
        
        if distribute_across_year:
            if randomize_windows:
                # Purely random start anywhere it fits
                max_start = max(0, (year_days - horizon_days) * SECONDS_PER_DAY)
                wstart = rng.uniform(0.0, float(max_start))
            else:
                # Even segments + jitter
                seg = year_days / max(n, 1)
                base_day = (i + 0.5) * seg
                jitter = rng.uniform(-0.35 * seg, 0.35 * seg)
                start_day = int(max(0, min(year_days - horizon_days, round(base_day + jitter))))
                wstart = float(start_day) * SECONDS_PER_DAY
        else:
            wstart = 0.0
        
        wend = wstart + float(window_len)
        items.append(dict(
            ra_deg=rng.uniform(0.0, 360.0),
            dec_deg=rng.uniform(-35.0, 75.0),
            window_start=wstart,
            window_end=wend,
            duration=float(dur),
            priority=rng.uniform(0.5, 1.5),
        ))
    
    # Assign RIDs, possibly shuffled
    rid_list = [f"R{i+1}" for i in range(n)]
    if shuffle_rids:
        rng.shuffle(rid_list)
    
    reqs: List[Request] = []
    for rid, item in zip(rid_list, items):
        reqs.append(Request(
            rid=rid,
            ra_deg=item["ra_deg"],
            dec_deg=item["dec_deg"],
            window_start=item["window_start"],
            window_end=item["window_end"],
            duration=item["duration"],
            priority=item["priority"],
        ))
    return reqs
# =========================
# Build feasibility graph
# =========================

def build_graph(requests: List[Request],
                start_ready_time: float = 0.0,
                night_start_hour: int = 21,
                night_end_hour: int = 5,
                idle_penalty_per_sec: float = 0.0,
                strict_chronology: bool = False ## Should make this False
                ) -> Tuple[nx.DiGraph, List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
    """
    Build a forward-in-time feasibility graph with night-only scheduling.

    ## Nodes
    - "SRC" (start), each request.rid, and "SNK" (sink).
    - Request nodes store visibility + earliest feasible start/finish from SRC.

    ## Edges
    - SRC -> j if j can start at or after start_ready_time within a night.
    - i -> j if j can start after i's earliest finish (same night or later), still within j's window.
    - i -> SNK if i is feasible at all (lets you terminate).

    ## Costs
    - No slew/angle costs.
    - Edge 'cost' = idle_penalty_per_sec * wait_time (defaults to 0).

    ## DAG note
    - To keep a DAG, we enforce a forward ordering by each node's earliest feasible
      start from SRC; ties broken by request id.
    """
    G = nx.DiGraph()
    G.add_node("SRC", is_source=True, earliest_finish=start_ready_time, ra_deg=0, dec_deg=-35)

    ## Compute earliest feasible start/finish for each request from SRC
    est_src: Dict[str, Optional[float]] = {}
    efin_src: Dict[str, Optional[float]] = {}

    for r in requests:
        s = earliest_start_within_request(start_ready_time, r,
                                          night_start_hour=night_start_hour,
                                          night_end_hour=night_end_hour)
        est_src[r.rid] = s
        efin_src[r.rid] = (s + r.duration) if s is not None else None

        G.add_node(
            r.rid,
            ra_deg=r.ra_deg,
            dec_deg=r.dec_deg,
            window_start=r.window_start,
            window_end=r.window_end,
            duration=r.duration,
            priority=r.priority,
            earliest_start_from_src=s,
            earliest_finish_from_src=efin_src[r.rid],
        )

    ## SRC -> j edges
    for r in requests:
        s = est_src[r.rid]
        if s is None:
            continue  ## infeasible request
        wait_time = max(0.0, s - start_ready_time)
        cost = idle_penalty_per_sec * wait_time
        G.add_edge("SRC", r.rid,
                   trans_time=0.0,
                   wait_time=wait_time,
                   cost=cost,
                   earliest_start_j=s,
                   earliest_finish_j=s + r.duration)

    ## i -> j edges (zero transition time)
    for i in requests:
        fi = efin_src[i.rid]
        if fi is None:
            continue  ## i cannot be scheduled at all

        for j in requests:
            if i.rid == j.rid:
                continue

            ## Optional DAG guard: only allow edges that go "forward" in earliest-start ordering
            if strict_chronology:
                si0 = est_src[i.rid]
                sj0 = est_src[j.rid]
                if sj0 is None or si0 is None:
                    continue
                if si0 > sj0:
                    continue
                if si0 == sj0 and i.rid >= j.rid:
                    continue

            ## Can j start at/after i's earliest finish, respecting nights and its window?
            sj = earliest_start_within_request(fi, j,
                                               night_start_hour=night_start_hour,
                                               night_end_hour=night_end_hour)
            if sj is None:
                continue

            fj = sj + j.duration
            wait_time = max(0.0, sj - fi)
            cost = idle_penalty_per_sec * wait_time

            G.add_edge(i.rid, j.rid,
                       trans_time=0.0,
                       wait_time=wait_time,
                       cost=cost,
                       earliest_start_j=sj,
                       earliest_finish_j=fj)

    ## SNK node and exits
    G.add_node("SNK", is_sink=True,ra_deg=360, dec_deg=75)
    for r in requests:
        if est_src[r.rid] is not None:
            G.add_edge(r.rid, "SNK", cost=0.0, trans_time=0.0, wait_time=0.0)

    ## MILP-friendly outputs
    arcs: List[Tuple[str, str]] = list(G.edges())
    costs: Dict[Tuple[str, str], float] = {(u, v): float(G.edges[u, v]["cost"]) for (u, v) in arcs}
    return G, arcs, costs


# =========================
# Example usage + final check printing
# =========================

def _fmt_hms(seconds: Optional[float]) -> str:
    """Quick formatter for printing times as Dd HH:MM:SS (handles None)."""
    if seconds is None:
        return "—"
    day = int(seconds // SECONDS_PER_DAY)
    rem = int(seconds - day * SECONDS_PER_DAY)
    hh = rem // 3600
    mm = (rem % 3600) // 60
    ss = rem % 60
    return f"D{day} {hh:02d}:{mm:02d}:{ss:02d}"

def print_edge_visibility_report(G: nx.DiGraph, max_edges: Optional[int] = None) -> None:
    """
    For each edge (u -> v), print:
      - u and v
      - visibility windows for u and v
      - durations
      - edge earliest_start_j / earliest_finish_j / wait / cost
    """
    def node_window(n: str):
        d = G.nodes[n]
        return d.get("window_start"), d.get("window_end"), d.get("duration")

    ## Sort edges by (earliest start of v, then u, then ids) for readability
    def sort_key(e):
        u, v = e
        su = G.nodes[u].get("earliest_start_from_src", float("inf")) if u not in ("SRC", "SNK") else (-1 if u == "SRC" else float("inf"))
        sv = G.nodes[v].get("earliest_start_from_src", float("inf")) if v not in ("SRC", "SNK") else (-1 if v == "SRC" else float("inf"))
        return (sv, su, u, v)

    edges = list(G.edges())
    edges.sort(key=sort_key)

    print("\n=== Edge Visibility Check ===")
    printed = 0
    for (u, v) in edges:
        if max_edges is not None and printed >= max_edges:
            print(f"... ({len(edges) - printed} more edges not shown)")
            break

        uws, uwe, udur = node_window(u) if u in G.nodes else (None, None, None)
        vws, vwe, vdur = node_window(v) if v in G.nodes else (None, None, None)

        data = G.edges[u, v]
        est_j = data.get("earliest_start_j")
        efin_j = data.get("earliest_finish_j")
        wait = data.get("wait_time", 0.0)
        cost = data.get("cost", 0.0)

        print(f"{u:>4} -> {v:<4} | "
              f"U win: [{_fmt_hms(uws)}, {_fmt_hms(uwe)}], dur={udur if udur is not None else '—'}s | "
              f"V win: [{_fmt_hms(vws)}, {_fmt_hms(vwe)}], dur={vdur if vdur is not None else '—'}s | "
              f"edge start={_fmt_hms(est_j)}, finish={_fmt_hms(efin_j)}, wait={int(wait)}s, cost={cost:.3f}")
        printed += 1

def draw_ra_dec_graph_nx(
    G,
    outpath="graph_ra_dec.png",
    ra_attr="ra_deg",          # node attribute for Right Ascension
    dec_attr="dec_deg",        # node attribute for Declination
    label_attr="rid",      # node attribute used for label text
    ra_units="deg",        # "deg" (0–360) or "hour" (0–24)
    center_ra=None,        # None -> auto center (circular mean in deg)
    invert_x=True,         # True = sky-chart style (RA increases to the left)
    figsize=(9, 7),
    dpi=200,
    node_size=160,
    node_color="tab:blue",
    node_edgecolor="black",
    node_alpha=0.9,
    edge_color="0.5",
    edge_width=1.0,
    edge_alpha=0.6,
    font_size=9,
):
    """
    Draw a NetworkX graph on an (RA, Dec) plane with node labels from `label_attr`.
    - Requires each node to have RA/Dec in attributes `ra_attr`/`dec_attr`.
    - No grid is shown.
    - Saves to `outpath` if provided; otherwise displays the figure.
    """

    # ---- helpers ----
    def to_deg(ra):
        return float(ra) * 15.0 if str(ra_units).lower().startswith("h") else float(ra)

    def circ_mean_deg(vals_deg):
        r = np.deg2rad(vals_deg)
        s, c = np.sin(r).mean(), np.cos(r).mean()
        ang = math.degrees(math.atan2(s, c))
        return (ang + 360.0) % 360.0

    def recenter(vals_deg, center):
        # Map RA into [center-180, center+180)
        return (vals_deg - center + 540.0) % 360.0 - 180.0 + center

    # ---- collect node coords ----
    nodes_data = [(n, d) for n, d in G.nodes(data=True) if ra_attr in d and dec_attr in d]
    if not nodes_data:
        raise ValueError(f"No nodes have both '{ra_attr}' and '{dec_attr}' attributes.")

    ra_deg = np.array([to_deg(d[ra_attr]) for _, d in nodes_data], dtype=float)
    dec_deg = np.array([float(d[dec_attr]) for _, d in nodes_data], dtype=float)

    if center_ra is None:
        center_ra = circ_mean_deg(ra_deg)
    ra_centered = recenter(ra_deg, center_ra)

    # positions and labels
    pos = {n: (x, y) for (n, d), x, y in zip(nodes_data, ra_centered, dec_deg)}
    labels = {n: str(d.get(label_attr, n)) for n, d in G.nodes(data=True)}

    # ---- draw ----
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    nx.draw(
        G,
        pos=pos,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        edgecolors=node_edgecolor,
        alpha=node_alpha,
        width=edge_width,
        edge_color=edge_color,
        with_labels=False,   # labels drawn separately
    )
    nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=font_size, ax=ax)

    # axes styling (no grid)
    ax.set_xlabel(f"Right Ascension (deg, centered at {center_ra:.1f}°)")
    ax.set_ylabel("Declination (deg)")
    ax.set_aspect("equal")
    if invert_x:
        ax.invert_xaxis()
    ax.grid(False)  # <- no grid

    # tight-ish limits with small padding
    pad_x = max(1.0, 0.03 * (ra_centered.max() - ra_centered.min() + 1e-6))
    pad_y = max(1.0, 0.03 * (dec_deg.max() - dec_deg.min() + 1e-6))
    ax.set_xlim(ra_centered.min() - pad_x, ra_centered.max() + pad_x)
    ax.set_ylim(dec_deg.min() - pad_y, dec_deg.max() + pad_y)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    ## Generate a small set
    reqs = generate_requests(
        n=4,
        horizon_days=14,      ## 2-week windows
        min_duration_sec=300, ## 5 min
        max_duration_sec=1200,## 20 min
        seed=7,
        distribute_across_year=True,
        year_days=365         ## 2025
    ,
        randomize_windows=True,
        shuffle_rids=True)

    G, arcs, costs = build_graph(
        reqs,
        start_ready_time=0.0,  ## telescope is ready at t0
        night_start_hour=21,
        night_end_hour=5,
        idle_penalty_per_sec=0.0,  ## set >0 to discourage waiting
        strict_chronology=False
    )

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print("Earliest feasible starts from SRC:")
    for r in reqs:
        s = G.nodes[r.rid]["earliest_start_from_src"]
        f = G.nodes[r.rid]["earliest_finish_from_src"]
        if s is None:
            print(f"  {r.rid}: infeasible")
        else:
            print(f"  {r.rid}: start={_fmt_hms(s)}  finish={_fmt_hms(f)}  dur={int(r.duration)}s")

    ## Final check: print edges + each endpoint's visibility windows
    print_edge_visibility_report(G)
    draw_ra_dec_graph_nx(G, outpath="sky_graph.png", ra_units="deg", label_attr="rid")

