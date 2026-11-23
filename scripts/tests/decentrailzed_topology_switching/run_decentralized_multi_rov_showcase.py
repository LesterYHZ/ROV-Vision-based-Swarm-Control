from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from fleet_model import (
    ROV, SphereObstacle, AprilTagLinkEvaluator, create_default_scenario,
    leader_figure8, rot_b_to_n, rot_n_to_b, step_rov,
)
from decentralized_rov import (
    DecentralizedTopologySelector, LocalParentKinematicsEstimator,
    LocalPIDParentTracker, LocalRelativeMotionEstimator,
    DecentralizedVisionCBF, make_local_tag_observation, detect_local_objects,
)


def find_parent_cycles(parent_map):
    """Offline logger helper: return directed cycles in follower->parent assignments."""
    found=set()
    for start in parent_map:
        path=[]
        index={}
        node=start
        while node in parent_map and parent_map[node] is not None:
            if node in index:
                cyc=path[index[node]:]
                if cyc:
                    # Canonical rotation makes the same directed cycle count once.
                    rotations=[tuple(cyc[i:]+cyc[:i]) for i in range(len(cyc))]
                    found.add(min(rotations))
                break
            if node in path:
                break
            index[node]=len(path)
            path.append(node)
            node=parent_map[node]
    return sorted(found)


def run(duration=300.0, dt=0.1, outdir='results', seed=4):
    rng=np.random.default_rng(seed)
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    print('SHOWCASE configuration: 300 s figure-eight, 140 deg x 100 deg AprilTag camera,')
    print('22 m detection range, five perimeter obstacles, and scripted local link outages.')
    leader,followers,obstacles=create_default_scenario()
    # Put F2 closer to the leader than F1, and F3 behind F1. During the scripted
    # F1->Leader tag dropout, F1 can still see F2 while F2 can see the leader.
    # This explicitly exercises the desired chain F3 -> F1 -> F2 -> Leader.
    lp0, ly0, _, _ = leader_figure8(0.0, period=240.0)
    forward0=np.array([math.cos(ly0),math.sin(ly0),0.0])
    side0=np.array([-math.sin(ly0),math.cos(ly0),0.0])
    followers[0].pos=lp0-8.8*forward0+0.15*side0  # F1
    followers[1].pos=lp0-4.3*forward0-0.10*side0  # F2, ahead of F1
    followers[2].pos=lp0-13.2*forward0+0.25*side0 # F3, behind F1
    for f in followers:
        f.yaw=ly0
    # Place reef_A slightly ahead of the late-turn follower's camera direction so it is
    # detectable before the safety boundary (rather than entering from behind a forward camera).
    obstacles[0].pos=np.array([10.0,-12.5,8.0])
    obstacles[1].pos=np.array([33.0,13.5,8.0])
    obstacles[2].pos=np.array([8.0,14.0,8.0])
    obstacles[3].pos=np.array([33.0,-14.0,8.0])
    obstacles[4].pos=np.array([7.0,-14.5,8.0])
    fleet=[leader]+followers
    n=len(fleet)

    # There is deliberately NO priority/rank table. Any visible other vehicle may be selected.
    # The simulation includes a synthetic camera dropout on F1->Leader to model tag occlusion
    # or turn-induced detection failure while F1->F2 and F2->Leader remain available.
    f1_f2_occlusion_end=32.0
    f1_leader_dropout=(32.0, 52.0)

    evaluator=AprilTagLinkEvaluator(
        hfov_deg=140.0,vfov_deg=100.0,max_range=22.0,min_range=1.0,
        max_incidence_deg=105.0,max_rel_heading_deg=155.0,obstacles=obstacles)

    selectors={i:DecentralizedTopologySelector(
        own_id=i,
        q_enter=0.02,q_keep=0.012,desired_range=5.0,
        switch_penalty=0.12,switch_margin=0.035,min_dwell=1.0,
        hold_cost=3.5,quality_weight=1.0,range_weight=0.12,
        heading_weight=0.08,incidence_weight=0.06)
        for i in range(1,n)}

    parent_estimators={i:LocalParentKinematicsEstimator(tau=0.35) for i in range(1,n)}
    tag_motion_estimators={i:LocalRelativeMotionEstimator(tau=0.28,first_detection_closing_speed=0.12)
                           for i in range(1,n)}
    collision_motion_estimators={i:LocalRelativeMotionEstimator(tau=0.25,first_detection_closing_speed=0.22)
                                 for i in range(1,n)}
    # Per-agent desired formation offsets relative to whichever parent it locally selects.
    slot_y={1:0.0,2:2.25,3:-2.25}
    trackers={i:LocalPIDParentTracker(desired_range=4.6,desired_lateral=slot_y[i],desired_down=0.0)
              for i in range(1,n)}
    # Formation offsets depend on the selected parent: F1 is initially a rear-echelon
    # follower of L (8.8 m), but when it switches to F2 it uses the normal 4.6 m local gap.
    # This keeps F2 in F1's forward camera while F2 itself tracks the leader.
    parent_range_overrides={(1,0):8.8}
    cbfs={i:DecentralizedVisionCBF(
        camera_offset=(0.32,0.0,0.0),hfov_deg=140,vfov_deg=100,
        fov_margin_deg=(7,6),min_range=1.0,max_range=22.0,range_margin=0.15,
        max_rel_heading_deg=155,max_incidence_deg=105,
        collision_alpha=1.7,perception_alpha=1.8,follower_radius=0.72,
        command_lower=(-0.65,-0.40,-0.32,-1.00),
        command_upper=(0.90,0.40,0.32,1.00),
        max_delta=(0.40,0.35,0.30,0.50),
        weights=(1.0,2.0,2.0,1.25),smooth_weight=0.12)
        for i in range(1,n)}

    previous_cmd={i:np.zeros(4) for i in range(1,n)}
    parents={i:None for i in range(1,n)}
    lost_parent_time={i:0.0 for i in range(1,n)}
    topology_period=0.5
    topology_steps=max(1,int(round(topology_period/dt)))
    nsteps=int(round(duration/dt))

    times=[]
    pos_hist={i:[] for i in range(n)}; yaw_hist={i:[] for i in range(n)}
    parent_hist={i:[] for i in range(1,n)}; error_hist={i:[] for i in range(1,n)}
    selected_quality={i:[] for i in range(1,n)}; selected_range={i:[] for i in range(1,n)}
    selected_heading={i:[] for i in range(1,n)}; selected_incidence={i:[] for i in range(1,n)}
    hold_hist={i:[] for i in range(1,n)}; feasible_hist={i:[] for i in range(1,n)}
    cbf_margin_hist={i:[] for i in range(1,n)}; cbf_corr_hist={i:[] for i in range(1,n)}
    min_clearance_hist={i:[] for i in range(1,n)}
    local_quality_hist={(i,j):[] for i in range(1,n) for j in range(n) if j!=i}
    local_visible_hist={(i,j):[] for i in range(1,n) for j in range(n) if j!=i}
    switch_events=[]; total_switches={i:0 for i in range(1,n)}
    local_decision_modes={i:[] for i in range(1,n)}
    local_candidate_counts={i:[] for i in range(1,n)}
    cycle_flag_hist=[]
    f1_dropout_steps=0; f1_to_f2_during_dropout=0; f2_to_leader_during_dropout=0

    for k in range(nsteps+1):
        t=k*dt
        lp,ly,lv,lr=leader_figure8(t,period=300.0)
        leader.pos=lp; leader.yaw=ly; leader.nu=rot_n_to_b(ly)@lv; leader.r=lr
        times.append(t)
        for node in fleet:
            pos_hist[node.idx].append(node.pos.copy()); yaw_hist[node.idx].append(node.yaw)

        # Each follower builds an independent camera frame containing every other vehicle
        # whose tag it attempts to detect. No shared observation or link-quality matrix exists.
        local_frames={}
        for i in range(1,n):
            observer=followers[i-1]
            frame={}
            for target in fleet:
                if target.idx==i:
                    continue
                obs=make_local_tag_observation(observer,target,evaluator)
                # Scripted perception sequence to exercise the desired decentralized relay:
                # (1) before t=32 s, F1's view of F2 is occluded, so F1 selects the leader;
                # (2) from 32 to 36 s, a leader turn/partial occlusion hides L from F1,
                #     while F1 can see F2 and F2 can still see L;
                # (3) all masks are local detector outputs, not shared network information.
                if i==1 and target.idx==2 and t < f1_f2_occlusion_end:
                    obs.visible=False
                    obs.quality=0.0
                    obs.occluded=True
                if (i==1 and target.idx==0
                        and f1_leader_dropout[0] <= t <= f1_leader_dropout[1]):
                    obs.visible=False
                    obs.quality=0.0
                    obs.occluded=True
                frame[target.idx]=obs
            local_frames[i]=frame
            for j,obs in frame.items():
                local_quality_hist[(i,j)].append(obs.quality)
                local_visible_hist[(i,j)].append(int(obs.visible))

        in_dropout=f1_leader_dropout[0] <= t <= f1_leader_dropout[1]
        if in_dropout:
            f1_dropout_steps += 1

        # Fully decentralized topology update: each follower chooses freely among all
        # currently visible other ROV tags, using only its own camera frame and local history.
        if k % topology_steps == 0:
            for i in range(1,n):
                old=parents[i]
                new,cost,diag=selectors[i].select(local_frames[i],t)
                if new != old:
                    switch_events.append((t,i,-1 if old is None else old,-1 if new is None else new,
                                          str(diag['mode']),float(cost)))
                    if old is not None: total_switches[i]+=1
                    parent_estimators[i].reset(i)
                    tag_motion_estimators[i].reset(f"parent_tag_{i}")
                    trackers[i].reset(i)
                    trackers[i].desired[0]=parent_range_overrides.get((i,new),4.6)
                parents[i]=new
                local_decision_modes[i].append(diag['mode'])
                local_candidate_counts[i].append(len(diag['candidate_ids']))
            if in_dropout:
                if parents[1] == 2:
                    f1_to_f2_during_dropout += 1
                if parents[2] == 0:
                    f2_to_leader_during_dropout += 1

        # Offline-only cycle monitor. This graph reconstruction is never fed to a follower.
        cycle_flag_hist.append(int(len(find_parent_cycles(parents)) > 0))

        if k==nsteps: break

        commands={}
        for i in range(1,n):
            follower=followers[i-1]
            parent_id=parents[i]
            tag_obs=local_frames[i].get(parent_id) if parent_id is not None else None
            if tag_obs is not None and tag_obs.visible:
                lost_parent_time[i]=0.0
                # Synthetic camera pose noise; controller receives only this local measurement.
                pos_sigma=0.006+0.016*(1.0-tag_obs.quality)
                noisy=tag_obs
                rel_parent_body=tag_obs.target_origin_body+rng.normal(0.0,pos_sigma,3)
                parent_vel_est,parent_yaw_rate_est=parent_estimators[i].update(follower,noisy,dt)
                # The finite-difference estimator uses the noiseless local measurement in this
                # simulator; perturb the pose supplied to PID to model detector noise.
                q_nom=trackers[i].compute(follower,rel_parent_body,parent_vel_est,
                                          tag_obs.relative_yaw,parent_yaw_rate_est,dt)
                tag_p_dot=tag_motion_estimators[i].update_position(
                    f"parent_tag_{i}",tag_obs.tag_position_body,follower,dt,is_static=False)
                local_parent_obs=tag_obs
                local_parent_yaw_rate=parent_yaw_rate_est
                hold_hist[i].append(0)
                selected_quality[i].append(tag_obs.quality)
                selected_range[i].append(tag_obs.range)
                selected_heading[i].append(math.degrees(abs(tag_obs.relative_yaw)))
                selected_incidence[i].append(math.degrees(tag_obs.incidence))
                error_hist[i].append(float(np.linalg.norm(rel_parent_body-trackers[i].desired)))
            else:
                lost_parent_time[i]+=dt
                parent_estimators[i].reset(i)
                tag_motion_estimators[i].reset(f"parent_tag_{i}")
                last=previous_cmd[i]
                decay=math.exp(-lost_parent_time[i]/5.5)
                q_nom=decay*np.array([0.75*last[0],0.58*last[1],0.60*last[2],0.62*last[3]])
                tag_p_dot=None; local_parent_obs=None; local_parent_yaw_rate=None
                hold_hist[i].append(1)
                selected_quality[i].append(0.0); selected_range[i].append(np.nan)
                selected_heading[i].append(np.nan); selected_incidence[i].append(np.nan)
                error_hist[i].append(np.nan)

            # Independent visual detection of static obstacles and other ROV bodies.
            local_detections=detect_local_objects(
                follower,fleet,obstacles,collision_motion_estimators[i],dt,
                camera_offset=(0.32,0,0),hfov_deg=360,vfov_deg=160,max_range=17.0)

            # Local unified CBF: only the selected tag pose and this camera's object detections.
            cmd,info=cbfs[i].filter(
                observer=follower,q_nom=q_nom,parent_obs=local_parent_obs,
                parent_tag_velocity_body=tag_p_dot,parent_yaw_rate=local_parent_yaw_rate,
                collision_detections=local_detections,dt=dt,previous_cmd=previous_cmd[i])
            commands[i]=cmd; previous_cmd[i]=cmd.copy()
            parent_hist[i].append(-1 if parent_id is None else parent_id)
            feasible_hist[i].append(int(info['feasible']))
            cbf_margin_hist[i].append(info['min_margin']); cbf_corr_hist[i].append(info['correction_norm'])

            clearances=[]
            # Ground-truth is used only by the simulator for post-run evaluation, not by follower logic.
            for ob in obstacles: clearances.append(np.linalg.norm(ob.pos-follower.pos)-(0.72+ob.radius))
            for other in fleet:
                if other.idx!=i: clearances.append(np.linalg.norm(other.pos-follower.pos)-(0.72+0.82))
            min_clearance_hist[i].append(float(min(clearances)))

        # Synchronous actuator/vehicle propagation; followers do not receive each other's commands.
        for i in range(1,n):
            follower=followers[i-1]
            step_rov(follower,commands[i],dt,tau=(0.38,0.42,0.45,0.22),accel=(0.95,0.80,0.70,1.65))
            current_n=np.array([0.025*math.sin(0.07*t+i),0.03*math.cos(0.05*t+0.7*i),0.0])
            follower.pos += current_n*dt

    # Post-run evaluation using simulator ground truth (not available to topology selectors).
    ts=np.asarray(times)
    pos={i:np.asarray(pos_hist[i]) for i in range(n)}
    yaw={i:np.unwrap(np.asarray(yaw_hist[i])) for i in range(n)}
    collision_samples=0; min_clearance_all=float('inf')
    for k in range(1,len(ts)):
        for i in range(1,n):
            vals=[np.linalg.norm(ob.pos-pos[i][k])-(0.72+ob.radius) for ob in obstacles]
            vals += [np.linalg.norm(pos[j][k]-pos[i][k])-(0.72+0.82) for j in range(n) if j!=i]
            mc=min(vals); min_clearance_all=min(min_clearance_all,mc)
            if mc<0: collision_samples+=1

    # Evaluate selected-edge visibility with ground truth only for metrics.
    physical_losses=0; active_samples=0
    for i in range(1,n):
        for k,parent_id in enumerate(parent_hist[i]):
            if parent_id<0: continue
            kk=min(k,len(ts)-1)
            observer=ROV(i,f'Follower_{i}',pos[i][kk],yaw[i][kk])
            target=ROV(parent_id,f'Parent_{parent_id}',pos[parent_id][kk],yaw[parent_id][kk])
            link=evaluator.evaluate(observer,target)
            active_samples+=1
            if not link.visible: physical_losses+=1

    metrics={
        'architecture':'decentralized_local_vision_no_inter_agent_communication',
        'duration_s':duration,'dt_s':dt,'steps':nsteps,
        'inter_agent_messages_sent':0,'shared_link_quality_matrices':0,'centralized_optimizer_calls':0,
        'tag_camera_hfov_deg':140.0,'collision_visual_envelope_hfov_deg':360.0,
        'parent_selection':'any_other_locally_visible_vehicle_no_rank_restriction',
        'cycle_prevention':'not_imposed_cycles_are_an_open_limitation',
        'topology_assignment_events_including_initial':len(switch_events),
        'parent_switches_after_initial_assignment':max(0,len(switch_events)-(n-1)),
        'active_parent_link_fraction':active_samples/max((n-1)*nsteps,1),
        'selected_link_physical_loss_samples':physical_losses,
        'selected_link_physical_loss_fraction':physical_losses/max(active_samples,1),
        'collision_samples':collision_samples,'minimum_sampled_clearance_m':min_clearance_all,
        'qp_infeasible_samples':sum(sum(1-x for x in feasible_hist[i]) for i in range(1,n)),
        'mean_selected_link_quality':float(np.nanmean([np.asarray(selected_quality[i]) for i in range(1,n)])),
        'mean_tracking_error_m':float(np.nanmean([np.asarray(error_hist[i]) for i in range(1,n)])),
        'observed_cycle_samples_offline':int(sum(cycle_flag_hist)),
        'observed_cycle_fraction_offline':float(np.mean(cycle_flag_hist)) if cycle_flag_hist else 0.0,
        'cycle_prevention_enabled':False,
        'synthetic_f1_to_leader_dropout_start_s':f1_leader_dropout[0],
        'synthetic_f1_to_f2_occlusion_end_s':f1_f2_occlusion_end,
        'synthetic_f1_to_leader_dropout_end_s':f1_leader_dropout[1],
        'f1_dropout_duration_samples':int(f1_dropout_steps),
        'f1_selected_f2_fraction_during_dropout':float(f1_to_f2_during_dropout/max(1, int(math.floor((f1_leader_dropout[1]-f1_leader_dropout[0])/(topology_period))+1))),
        'f2_selected_leader_fraction_during_dropout':float(f2_to_leader_during_dropout/max(1, int(math.floor((f1_leader_dropout[1]-f1_leader_dropout[0])/(topology_period))+1))),
    }
    metrics['total_holdover_samples'] = int(sum(sum(hold_hist[i]) for i in range(1,n)))
    metrics['all_followers_zero_holdover'] = bool(all(np.mean(hold_hist[i]) == 0.0 for i in range(1,n)))
    metrics['all_selected_links_physically_visible'] = bool(physical_losses == 0)
    metrics['all_cbfs_feasible'] = bool(metrics['qp_infeasible_samples'] == 0)
    metrics['showcase_configuration'] = '300 s figure-eight; 140x100 deg camera; 22 m range; q_enter=0.02; five perimeter obstacles'
    for i in range(1,n):
        metrics[f'follower_{i}_parent_switches']=total_switches[i]
        metrics[f'follower_{i}_hold_fraction']=float(np.mean(hold_hist[i]))
        metrics[f'follower_{i}_mean_tracking_error_m']=float(np.nanmean(error_hist[i]))
        metrics[f'follower_{i}_mean_candidate_count']=float(np.mean(local_candidate_counts[i]))
        metrics[f'follower_{i}_minimum_clearance_m']=float(np.nanmin(min_clearance_hist[i]))

    with open(out/'simulation_metrics.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['metric','value']); w.writerows(metrics.items())
    with open(out/'decentralized_switch_events.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['time_s','follower_id','old_parent_id','new_parent_id','local_mode','local_cost'])
        for row in switch_events: w.writerow(row)
    with open(out/'local_topology_timeline.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['time_s','F1_parent','F2_parent','F3_parent'])
        m=min(len(parent_hist[i]) for i in range(1,n))
        for k in range(m): w.writerow([ts[k]]+[parent_hist[i][k] for i in range(1,n)])
    np.savez_compressed(out/'decentralized_simulation_data.npz',time=ts,
        **{f'position_{i}':pos[i] for i in range(n)},
        **{f'yaw_{i}':yaw[i] for i in range(n)},
        **{f'parent_{i}':np.asarray(parent_hist[i],int) for i in range(1,n)},
        **{f'error_{i}':np.asarray(error_hist[i],float) for i in range(1,n)},
        **{f'quality_{i}_{j}':np.asarray(local_quality_hist[(i,j)],float) for i,j in local_quality_hist},
        cycle_flag=np.asarray(cycle_flag_hist,int))

    # Plot: trajectories and obstacles.
    fig=plt.figure(figsize=(12,9)); ax=fig.add_subplot(111,projection='3d')
    colors=['black','#1f77b4','#ff7f0e','#2ca02c']; names=['Leader','Follower 1','Follower 2','Follower 3']
    for i in range(n):
        ax.plot(pos[i][:,0],pos[i][:,1],pos[i][:,2],color=colors[i],lw=2,label=names[i])
        ids=np.linspace(0,len(ts)-1,20,dtype=int)
        for kk in ids:
            x0,y0,z0=pos[i][kk]; L=1.2
            ax.quiver(x0,y0,z0,L*math.cos(yaw[i][kk]),L*math.sin(yaw[i][kk]),0,
                      color=colors[i],arrow_length_ratio=0.25,linewidth=0.75)
    for ob in obstacles:
        uu=np.linspace(0,2*np.pi,18); vv=np.linspace(0,np.pi,10)
        X=ob.pos[0]+ob.radius*np.outer(np.cos(uu),np.sin(vv))
        Y=ob.pos[1]+ob.radius*np.outer(np.sin(uu),np.sin(vv))
        Z=ob.pos[2]+ob.radius*np.outer(np.ones_like(uu),np.cos(vv))
        ax.plot_wireframe(X,Y,Z,color='gray',alpha=.45,linewidth=.45); ax.text(*ob.pos,ob.name,fontsize=7)
    ax.set_xlabel('North (m)'); ax.set_ylabel('East (m)'); ax.set_zlabel('Down (m)')
    ax.set_title('Decentralized vision-only multi-ROV formation and static obstacles')
    ax.invert_zaxis(); ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(out/'decentralized_ned_trajectories.png',dpi=180); plt.close(fig)

    # Parent selection timeline.
    fig,axs=plt.subplots(3,1,figsize=(12,7),sharex=True)
    for row,i in enumerate(range(1,n)):
        arr=np.asarray(parent_hist[i],int); tt=ts[:len(arr)]
        axs[row].step(tt,arr,where='post',lw=1.6)
        axs[row].set_yticks([-1,0,1,2,3]); axs[row].set_yticklabels(['HOLD','Leader','F1','F2','F3'])
        axs[row].set_ylabel(f'F{i} parent'); axs[row].grid(True,alpha=.3)
    axs[-1].set_xlabel('Time (s)'); fig.suptitle('Independent local parent decisions (no communication)')
    for a in axs:
        a.axvspan(0.0,f1_f2_occlusion_end,color='gray',alpha=.10,label='F1->F2 occlusion')
        a.axvspan(*f1_leader_dropout,color='orange',alpha=.15,label='F1->Leader dropout')
    fig.tight_layout(); fig.savefig(out/'decentralized_topology_timeline.png',dpi=180); plt.close(fig)

    # Topology snapshots; arrows are reconstructed by logger, not communicated to agents.
    snap_times=np.linspace(0,duration,7); fig,axs=plt.subplots(1,len(snap_times),figsize=(17,3.4))
    node_xy={0:(0,.82),1:(-.9,0),2:(0,-.22),3:(.9,0)}
    for ax,st in zip(axs,snap_times):
        kk=int(np.clip(round(st/dt),0,len(ts)-1)); ax.set_title(f't={st:.0f}s',fontsize=9)
        for node,(xx,yy) in node_xy.items():
            ax.scatter(xx,yy,s=410 if node==0 else 320,c=colors[node],edgecolors='black',zorder=3)
            ax.text(xx,yy,'L' if node==0 else f'F{node}',ha='center',va='center',color='white',fontsize=8,fontweight='bold')
        for child in range(1,n):
            if kk<len(parent_hist[child]):
                parent=parent_hist[child][kk]
                if parent>=0:
                    x1,y1=node_xy[parent]; x2,y2=node_xy[child]
                    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='purple',lw=1.6,shrinkA=16,shrinkB=16))
        ax.set_xlim(-1.3,1.3); ax.set_ylim(-.6,1.25); ax.axis('off')
    fig.suptitle('Observed decentralized parent graph (reconstructed offline; cycles not prevented)')
    fig.tight_layout(); fig.savefig(out/'decentralized_topology_snapshots.png',dpi=180); plt.close(fig)

    # Local tag quality for every other vehicle, including higher-ID followers.
    fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for row,i in enumerate(range(1,n)):
        for j in range(n):
            if j==i: continue
            arr=np.asarray(local_quality_hist[(i,j)],float)
            axs[row].plot(ts[:len(arr)],arr,label=('Leader' if j==0 else f'F{j}'),lw=1.1)
        axs[row].axhline(selectors[i].q_enter,color='k',ls='--',lw=.8,label='enter threshold')
        if i==1:
            axs[row].axvspan(0.0,f1_f2_occlusion_end,color='gray',alpha=.10)
            axs[row].axvspan(*f1_leader_dropout,color='orange',alpha=.15)
        axs[row].set_ylim(-.02,1.02); axs[row].set_ylabel(f'F{i} local quality'); axs[row].grid(True,alpha=.3); axs[row].legend(ncol=4,fontsize=8)
    axs[-1].set_xlabel('Time (s)'); fig.suptitle('All candidate AprilTag qualities measured independently by each follower')
    fig.tight_layout(); fig.savefig(out/'decentralized_local_link_quality.png',dpi=180); plt.close(fig)

    # Tracking/hold diagnostics.
    fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for i in range(1,n):
        tt=ts[:len(error_hist[i])]
        axs[0].plot(tt,error_hist[i],label=f'F{i}')
        axs[1].step(tt,hold_hist[i],where='post',label=f'F{i}')
    axs[0].set_ylabel('Selected-parent position error (m)'); axs[1].set_ylabel('Holdover (1=yes)')
    axs[1].set_xlabel('Time (s)')
    for a in axs: a.grid(True,alpha=.3); a.legend()
    fig.suptitle('Local tracking performance and holdover')
    fig.tight_layout(); fig.savefig(out/'decentralized_tracking_and_holdover.png',dpi=180); plt.close(fig)

    # CBF safety diagnostics.
    fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for i in range(1,n):
        tt=ts[:len(cbf_corr_hist[i])]
        axs[0].plot(tt,cbf_corr_hist[i],label=f'F{i}')
        axs[1].plot(tt,min_clearance_hist[i],label=f'F{i}')
    axs[0].set_ylabel('||q_safe - q_PID||'); axs[1].set_ylabel('True clearance (m)')
    axs[1].axhline(0,color='r',ls='--',lw=.9); axs[1].set_xlabel('Time (s)')
    for a in axs: a.grid(True,alpha=.3); a.legend()
    fig.suptitle('Per-follower vision-based CBF corrections and offline clearance evaluation')
    fig.tight_layout(); fig.savefig(out/'decentralized_cbf_diagnostics.png',dpi=180); plt.close(fig)

    # Offline cycle-risk diagnostic (not used by any follower).
    fig,ax=plt.subplots(figsize=(11,2.8))
    ax.step(ts[:len(cycle_flag_hist)],cycle_flag_hist,where='post',color='purple')
    ax.set_yticks([0,1],['No cycle observed','Cycle observed'])
    ax.set_xlabel('Time (s)'); ax.set_title('Offline monitoring of cycles in independently selected parent graph')
    ax.grid(True,alpha=.3); fig.tight_layout()
    fig.savefig(out/'offline_cycle_monitor.png',dpi=180); plt.close(fig)

    print('Decentralized simulation metrics:')
    for key,val in metrics.items(): print(f'  {key}: {val}')
    print('Local switch events:')
    for row in switch_events: print(' ',row)
    return metrics,switch_events


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--duration',type=float,default=300.0)
    ap.add_argument('--dt',type=float,default=0.1)
    ap.add_argument('--outdir',type=str,default='results')
    ap.add_argument('--seed',type=int,default=4)
    args=ap.parse_args()
    run(args.duration,args.dt,args.outdir,args.seed)
