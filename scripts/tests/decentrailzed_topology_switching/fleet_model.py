"""Shared reduced-order ROV and synthetic camera model used by the decentralized example.

Only the simulator/sensor generator has fleet ground truth. Each controller receives its own
local camera observations; no controller receives another ROV's state or parent assignment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math
import numpy as np


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot_b_to_n(yaw: float) -> np.ndarray:
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]])


def rot_n_to_b(yaw: float) -> np.ndarray:
    return rot_b_to_n(yaw).T


def unit(v: np.ndarray, eps: float=1e-9) -> np.ndarray:
    n=float(np.linalg.norm(v)); return np.asarray(v,float)/max(n,eps)


def segment_point_distance(a: np.ndarray,b: np.ndarray,p: np.ndarray) -> float:
    ab=b-a; den=float(ab@ab)
    if den<1e-12: return float(np.linalg.norm(p-a))
    t=float(np.clip(((p-a)@ab)/den,0.0,1.0))
    return float(np.linalg.norm(p-(a+t*ab)))


@dataclass
class ROV:
    idx: int
    name: str
    pos: np.ndarray
    yaw: float
    nu: np.ndarray=field(default_factory=lambda:np.zeros(3))
    r: float=0.0
    yaw_rate_cmd: float=0.0
    q_cmd: np.ndarray=field(default_factory=lambda:np.zeros(4))
    def velocity_ned(self)->np.ndarray:
        return rot_b_to_n(self.yaw)@self.nu


@dataclass
class SphereObstacle:
    name: str
    pos: np.ndarray
    radius: float


@dataclass
class LinkObservation:
    quality: float
    visible: bool
    range: float
    azimuth: float
    elevation: float
    rel_heading: float
    incidence: float
    occluded: bool


class AprilTagLinkEvaluator:
    """Synthetic camera/AprilTag geometry used only to render local observations."""
    def __init__(self,hfov_deg=72.0,vfov_deg=52.0,max_range=13.0,min_range=1.2,
                 max_incidence_deg=68.0,max_rel_heading_deg=78.0,
                 camera_offset=(0.32,0.0,0.0),tag_tail_offset=(-1.0,0.0,0.0),
                 obstacles:Optional[List[SphereObstacle]]=None):
        self.half_h=math.radians(hfov_deg/2.0); self.half_v=math.radians(vfov_deg/2.0)
        self.max_range=max_range; self.min_range=min_range
        self.max_inc=math.radians(max_incidence_deg); self.max_heading=math.radians(max_rel_heading_deg)
        self.camera_offset=np.asarray(camera_offset,float); self.tag_tail_offset=np.asarray(tag_tail_offset,float)
        self.obstacles=obstacles or []

    def evaluate(self,observer:ROV,target:ROV)->LinkObservation:
        R_i=rot_b_to_n(observer.yaw)
        cam_n=observer.pos+R_i@self.camera_offset
        tag_n=target.pos+rot_b_to_n(target.yaw)@self.tag_tail_offset
        p_cam=rot_n_to_b(observer.yaw)@(tag_n-cam_n)
        rng=float(np.linalg.norm(p_cam)); x,y,z=p_cam
        az=math.atan2(y,x); el=math.atan2(z,max(math.hypot(x,y),1e-9))
        rel_h=wrap_pi(target.yaw-observer.yaw)
        tag_normal_n=rot_b_to_n(target.yaw)@np.array([-1.0,0.0,0.0])
        tag_to_camera=unit(cam_n-tag_n)
        inc_cos=float(np.clip(tag_normal_n@tag_to_camera,-1.0,1.0))
        inc=math.acos(inc_cos); occluded=False
        for ob in self.obstacles:
            if segment_point_distance(cam_n,tag_n,ob.pos)<ob.radius+0.12:
                occluded=True; break
        visible=(x>0.0 and self.min_range<=rng<=self.max_range and
                 abs(az)<=self.half_h and abs(el)<=self.half_v and
                 inc<=self.max_inc and abs(rel_h)<=self.max_heading and not occluded)
        if not visible: q=0.0
        else:
            qr=math.exp(-((rng-5.0)/7.0)**2)
            qf=math.exp(-1.6*(abs(az)/self.half_h)**4-1.6*(abs(el)/self.half_v)**4)
            qi=max(0.0,(inc_cos-math.cos(self.max_inc))/(1.0-math.cos(self.max_inc)))
            qh=math.exp(-1.2*(abs(rel_h)/self.max_heading)**2)
            q=float(np.clip(qr*qf*qi*qh,0.0,1.0))
        return LinkObservation(q,visible,rng,az,el,rel_h,inc,occluded)


def leader_figure8(t:float,period=120.0)->Tuple[np.ndarray,float,np.ndarray,float]:
    om=2.0*math.pi/period
    N=20.0+9.0*math.sin(om*t); E=9.0*math.sin(2.0*om*t)
    D=8.0+0.45*math.sin(om*t+0.5)
    vN=9.0*om*math.cos(om*t); vE=18.0*om*math.cos(2.0*om*t)
    vD=0.45*om*math.cos(om*t+0.5); yaw=math.atan2(vE,vN)
    eps=1e-3
    def heading(tt):
        return math.atan2(18.0*om*math.cos(2.0*om*tt),9.0*om*math.cos(om*tt))
    yaw_rate=wrap_pi(heading(t+eps)-heading(t-eps))/(2.0*eps)
    return np.array([N,E,D]),yaw,np.array([vN,vE,vD]),yaw_rate


def step_rov(rov:ROV,q_cmd:np.ndarray,dt:float,tau=(0.65,0.75,0.8,0.55),
             accel=(0.55,0.42,0.35,0.8)):
    q_cmd=np.asarray(q_cmd,float)
    actual=np.array([rov.nu[0],rov.nu[1],rov.nu[2],rov.r])
    raw=np.clip((q_cmd-actual)/np.asarray(tau,float),-np.asarray(accel,float),np.asarray(accel,float))
    actual=actual+raw*dt; rov.nu=actual[:3]; rov.r=float(actual[3])
    rov.pos=rov.pos+rot_b_to_n(rov.yaw)@rov.nu*dt
    rov.yaw=wrap_pi(rov.yaw+rov.r*dt); rov.q_cmd=q_cmd.copy()


def create_default_scenario():
    obstacles=[
        SphereObstacle('reef_A',np.array([15.5,-5.2,8.0]),0.80),
        SphereObstacle('reef_B',np.array([32.0,13.0,8.0]),1.15),
        SphereObstacle('reef_C',np.array([7.0,13.0,8.0]),1.10),
        SphereObstacle('reef_D',np.array([32.0,-14.0,8.0]),1.10),
        SphereObstacle('reef_E',np.array([7.0,-14.0,8.0]),1.05),
    ]
    leader_pos,leader_yaw,_,_=leader_figure8(0.0,period=240.0)
    heading=np.array([math.cos(leader_yaw),math.sin(leader_yaw),0.0])
    followers=[]; lateral=[0.0,-1.6,0.0]
    for k in range(3):
        pos=leader_pos-(4.5*(k+1))*heading+np.array([0.0,0.0,0.15*k])
        side=np.array([-math.sin(leader_yaw),math.cos(leader_yaw),0.0])
        pos=pos+lateral[k]*side
        followers.append(ROV(k+1,f'Follower_{k+1}',pos.copy(),leader_yaw))
    leader=ROV(0,'Leader',leader_pos.copy(),leader_yaw)
    return leader,followers,obstacles
