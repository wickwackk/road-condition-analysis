# utils/geopose_utils.py
import math
from typing import Dict, Tuple

def ypr_deg_to_quat(yaw_deg: float, pitch_deg: float, roll_deg: float) -> Tuple[float, float, float, float]:
    """Yaw/Pitch/Roll (deg, Z-Y-X order) → quaternion (x,y,z,w)."""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    cy, sy = math.cos(y*0.5), math.sin(y*0.5)
    cp, sp = math.cos(p*0.5), math.sin(p*0.5)
    cr, sr = math.cos(r*0.5), math.sin(r*0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return qx, qy, qz, qw

def quat_to_ypr_deg(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """Quaternion (x,y,z,w) → Yaw/Pitch/Roll (deg, Z-Y-X)."""
    # yaw (z)
    siny_cosp = 2*(qw*qz + qx*qy)
    cosy_cosp = 1 - 2*(qy*qy + qz*qz)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    # pitch (y)
    sinp = 2*(qw*qy - qz*qx)
    pitch = math.degrees(math.copysign(math.pi/2, sinp)) if abs(sinp) >= 1 else math.degrees(math.asin(sinp))
    # roll (x)
    sinr_cosp = 2*(qw*qx + qy*qz)
    cosr_cosp = 1 - 2*(qx*qx + qy*qy)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
    return yaw, pitch, roll

def geopose_basic_ypr(lat: float, lon: float, h: float, yaw: float, pitch: float, roll: float) -> Dict:
    return {"position": {"lat": float(lat), "lon": float(lon), "h": float(h)},
            "angles": {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}}

def geopose_basic_quat(lat: float, lon: float, h: float, qx: float, qy: float, qz: float, qw: float) -> Dict:
    return {"position": {"lat": float(lat), "lon": float(lon), "h": float(h)},
            "quaternion": {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)}}
