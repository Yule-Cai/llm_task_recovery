import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import rcParams
import pandas as pd
import os

rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          9,
    'axes.titlesize':     10,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})

FIGS = 'figures'
os.makedirs(FIGS, exist_ok=True)

# Load data
def load(fname):
    return json.load(open(fname))

def load_map(path):
    return pd.read_csv(path, header=None).values.astype(np.float32)

def dedupe_traj(traj):
    """Remove consecutive duplicates from trajectory."""
    out = [traj[0]]
    for pt in traj[1:]:
        if pt != out[-1]:
            out.append(pt)
    return out

def draw_panel(ax, grid, title, traj_normal=None, traj_recovery=None,
               agent=None, item=None, goal=None, waypoints=None,
               radar_center=None, radar_r=3,
               blocked_cells=None, new_item=None,
               stuck_point=None, anomaly_step=None):
    H, W = grid.shape

    # Map background
    img = np.ones((H, W, 3))
    img[grid == 1] = [0.22, 0.22, 0.22]
    ax.imshow(img, origin='upper', interpolation='nearest', zorder=1)

    # Radar window
    if radar_center is not None:
        rx, ry = radar_center
        x0 = max(ry - radar_r - 0.5, -0.5)
        y0 = max(rx - radar_r - 0.5, -0.5)
        ax.add_patch(plt.Rectangle((x0, y0), 2*radar_r+1, 2*radar_r+1,
                     linewidth=1.8, edgecolor='#D6604D',
                     facecolor='#F4A582', alpha=0.18, zorder=2))
        ax.add_patch(plt.Rectangle((x0, y0), 2*radar_r+1, 2*radar_r+1,
                     linewidth=1.8, edgecolor='#D6604D',
                     facecolor='none', zorder=3))

    # Blocked cells
    if blocked_cells:
        for (bx, by) in blocked_cells:
            ax.add_patch(plt.Rectangle((by-0.6, bx-0.6), 1.2, 1.2,
                         color='#B2182B', alpha=1.0, zorder=6))

    # Normal trajectory
    if traj_normal and len(traj_normal) > 1:
        cols = [p[1] for p in traj_normal]
        rows = [p[0] for p in traj_normal]
        ax.plot(cols, rows, '-', color='#4393C3', alpha=0.8,
                linewidth=1.6, zorder=5)
        # Arrow at midpoint
        mid = len(cols)//2
        if mid > 0:
            ax.annotate('', xy=(cols[mid], rows[mid]),
                       xytext=(cols[mid-1], rows[mid-1]),
                       arrowprops=dict(arrowstyle='->', color='#4393C3',
                                      lw=1.2, mutation_scale=10), zorder=6)

    # Recovery trajectory
    if traj_recovery and len(traj_recovery) > 1:
        cols = [p[1] for p in traj_recovery]
        rows = [p[0] for p in traj_recovery]
        ax.plot(cols, rows, '--', color='#4DAC26', alpha=0.9,
                linewidth=1.6, zorder=5)

    # stuck marker drawn at end of function for visibility

    # Items
    if item is not None:
        ax.plot(item[1], item[0], 's', color='#762A83',
                markersize=9, zorder=7,
                markeredgecolor='white', markeredgewidth=0.8)

    if new_item is not None and item is not None:
        ax.plot(new_item[1], new_item[0], 'D', color='#FF6B00',
                markersize=11, zorder=7,
                markeredgecolor='#8B0000', markeredgewidth=1.8)
        ax.annotate('', xy=(new_item[1], new_item[0]),
                    xytext=(item[1], item[0]),
                    arrowprops=dict(arrowstyle='->', color='#D6604D',
                                   lw=1.8, mutation_scale=12), zorder=8)

    # Goal
    if goal is not None:
        ax.plot(goal[1], goal[0], '*', color='#F4A582',
                markersize=14, zorder=7,
                markeredgecolor='#D6604D', markeredgewidth=0.8)

    # Waypoints
    if waypoints:
        wp_colors = ['#1A9850', '#2166AC', '#762A83']
        for i, wp in enumerate(waypoints):
            ax.plot(wp[1], wp[0], '^', color=wp_colors[i],
                    markersize=9, zorder=7,
                    markeredgecolor='white', markeredgewidth=0.8)
            lx = min(wp[1]+1.2, W-3)
            ax.text(lx, wp[0], f'WP{i+1}', fontsize=7,
                    color=wp_colors[i], fontweight='bold',
                    zorder=8, va='center')

    # Agent start
    if agent is not None:
        ax.plot(agent[1], agent[0], 'o', color='#2166AC',
                markersize=10, zorder=9,
                markeredgecolor='white', markeredgewidth=1.2)

    # Draw stuck marker last to ensure visibility
    if stuck_point is not None:
        ax.plot(stuck_point[1], stuck_point[0], 'o',
                color='none', markersize=14, markeredgewidth=2.5,
                markeredgecolor='#FF0000', zorder=15)
        ax.plot(stuck_point[1], stuck_point[0], 'x',
                color='#FF0000', markersize=11, markeredgewidth=2.5,
                zorder=16)

    ax.set_xlim(-0.5, W-0.5)
    ax.set_ylim(H-0.5, -0.5)
    ax.set_title(title, fontsize=9.5, pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6); spine.set_color('0.4')


def main():
    # Load map
    map_path = 'data/maps/L3_map0.csv'
    grid = load_map(map_path)

    # Load episodes
    pn  = load('/mnt/user-data/uploads/episode_pick_normal.json')
    pd_ = load('/mnt/user-data/uploads/episode_pick_displace.json')
    po  = load('/mnt/user-data/uploads/episode_pick_obstruct.json')
    pat = load('/mnt/user-data/uploads/episode_patrol_normal.json')

    # Deduplicate trajectories
    traj_pn  = dedupe_traj([tuple(p) for p in pn['trajectory']])
    traj_pd  = dedupe_traj([tuple(p) for p in pd_['trajectory']])
    traj_po  = dedupe_traj([tuple(p) for p in po['trajectory']])
    traj_pat = dedupe_traj([tuple(p) for p in pat['trajectory']])

    # For normal: show first 27 steps (before stuck)
    traj_pn_early = traj_pn[:27]

    # For displacement: show pre-anomaly (first 27) as blue, post-anomaly as green
    traj_pd_pre  = traj_pd[:27]
    traj_pd_post = traj_pd[27:85]  # recovery attempt

    # For obstruction: show pre-anomaly (first 27) as blue, stuck portion
    traj_po_pre  = traj_po[:27]
    traj_po_stuck = traj_po[27:50]

    # For patrol: show first 26 steps (before stuck at wp1)
    traj_pat_early = traj_pat[:26]

    # Stuck points
    stuck_pn  = (22, 27)      # last unique position before oscillation
    stuck_pd  = traj_pd[83]   # second stuck
    stuck_po  = traj_po[26]
    stuck_pat = traj_pat[25]

    # Radar at stuck position
    radar_pn  = stuck_pn
    radar_pd  = traj_pd_pre[-1]
    radar_po  = stuck_po
    radar_pat = stuck_pat

    fig, axes = plt.subplots(2, 2, figsize=(9, 8.2),
                             gridspec_kw={'hspace': 0.15, 'wspace': 0.08})
    fig.suptitle('Fig. 0  Environment Overview and Real Agent Trajectories (L3 Map)',
                 fontsize=11, y=0.995)

    # (a) Normal pick-and-deliver
    draw_panel(axes[0,0], grid,
               '(a) Pick-and-Deliver (Normal)',
               traj_normal=traj_pn_early,
               agent=tuple(pn['agent_start']),
               item=tuple(pn['item_pos']),
               goal=tuple(pn['goal_pos']),
               radar_center=radar_pn,
)


    # (b) Displacement anomaly
    draw_panel(axes[0,1], grid,
               '(b) Displacement Anomaly',
               traj_normal=traj_pd_pre,
               traj_recovery=traj_pd_post,
               agent=tuple(pd_['agent_start']),
               item=tuple(pd_['item_pos']),
               goal=tuple(pd_['goal_pos']),
               new_item=tuple(pd_['item_pos_after']),
               radar_center=radar_pd,
)

    # (c) Obstruction anomaly
    g_obs = grid.copy()
    H, W = grid.shape
    valid_blocked = [[bx,by] for bx,by in po['blocked_cells']
                     if 0<=bx<H and 0<=by<W]
    for (bx,by) in valid_blocked:
        g_obs[bx,by] = 1
    draw_panel(axes[1,0], g_obs,
               '(c) Obstruction Anomaly',
               traj_normal=traj_po_pre,
               traj_recovery=traj_po_stuck,
               agent=tuple(po['agent_start']),
               item=tuple(po['item_pos']),
               goal=tuple(po['goal_pos']),
               blocked_cells=valid_blocked,
               radar_center=radar_po,
)

    # (d) Patrol task
    draw_panel(axes[1,1], grid,
               '(d) Ordered Patrol Task',
               traj_normal=traj_pat_early,
               agent=tuple(pat['agent_start']),
               waypoints=[tuple(wp) for wp in pat['waypoints']],
               radar_center=radar_pat,
)

    # Legend
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#2166AC',
               markersize=9, markeredgecolor='white', label='Agent start'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#762A83',
               markersize=8, markeredgecolor='white', label='Item (original)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#FF6B00',
               markersize=8, markeredgecolor='#8B0000',
               markeredgewidth=1.5, label='Item (displaced)'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor='#F4A582',
               markersize=12, markeredgecolor='#D6604D', label='Goal'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='#1A9850',
               markersize=8, markeredgecolor='white', label='Waypoint'),
        mpatches.Patch(facecolor='#B2182B', label='Injected obstacles'),
        Line2D([0],[0], color='#4393C3', linewidth=1.5, label='Agent trajectory'),
        Line2D([0],[0], color='#4DAC26', linewidth=1.5,
               linestyle='--', label='Post-anomaly trajectory'),

        mpatches.Patch(facecolor='#F4A582', alpha=0.3,
                       edgecolor='#D6604D', linewidth=1.5,
                       label='Radar window (7×7)'),
        mpatches.Patch(facecolor='#404040', label='Wall'),
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=8, frameon=True,
               bbox_to_anchor=(0.5, 0.005),
               edgecolor='0.75', framealpha=0.95)

    plt.subplots_adjust(bottom=0.14)
    out_png = os.path.join(FIGS, 'fig0_map_viz.png')
    out_pdf = os.path.join(FIGS, 'fig0_map_viz.pdf')
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()
    print('Saved:', out_png)

if __name__ == '__main__':
    main()