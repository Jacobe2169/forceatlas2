import numpy as np
import math
from numba import njit,jit



def attraction(
        node_u,
        node_v,
        scaling_ratio: float,
        lin_log=False,
        prevent_overlap=False,
        weight=None,
        distributed=False,
):
    return __attraction(
        node_u.coord,
        node_v.coord,
        node_u.mass,
        node_u.size,
        node_v.size,
        scaling_ratio,
        lin_log,
        prevent_overlap,
        weight,
        distributed,
    )

    
@jit
def __attraction(
        node_u_coord,
        node_v_coord,
        node_u_mass,
        node_u_size,
        node_v_size,
        scaling_ratio,
        lin_log,
        prevent_overlap,
        weight,
        distributed,
):

    dist = np.sqrt(
        (node_u_coord[0] - node_v_coord[0]) ** 2
        + (node_u_coord[1] - node_v_coord[1]) ** 2
    )
    if prevent_overlap:
        dist -= node_u_size - node_v_size

    if not prevent_overlap and not distributed and not lin_log:
        return weight * - scaling_ratio
    
    if distributed and not prevent_overlap and not lin_log:
        return -scaling_ratio * weight /node_u_mass
    
    if lin_log and not distributed and not prevent_overlap and dist >0:
        return -scaling_ratio * weight * np.log(1+dist)/dist
    
    if lin_log and distributed and not prevent_overlap and dist >0:
        return -scaling_ratio * weight * np.log(1+dist)/dist/node_u_mass
    
    if prevent_overlap and not lin_log and not distributed and dist >0:
        return -scaling_ratio * weight

    if prevent_overlap and distributed and not lin_log and dist >0:
        return -scaling_ratio * weight/node_u_mass
    
    if lin_log and prevent_overlap and not distributed and dist >0:
        return -scaling_ratio * weight * np.log(1+dist)/dist

    if lin_log and prevent_overlap and distributed and dist >0:
        return -scaling_ratio * weight * np.log(1+dist)/dist/node_u_mass
    return 0


def repulsion(node_u, node_v, scaling_ratio, prevent_overlap=False):
    return __repulsion(
        node_u.coord,
        node_v.coord,
        node_u.mass,
        node_v.mass,
        node_u.size,
        node_v.size,
        scaling_ratio,
        prevent_overlap,
    )


@jit
def __repulsion(
        node_u_coord,
        node_v_coord,
        node_u_mass,
        node_v_mass,
        node_u_size,
        node_v_size,
        scaling_ratio,
        prevent_overlap,
):
    dist = np.sqrt(
        (node_u_coord[0] - node_v_coord[0]) ** 2
        + (node_u_coord[1] - node_v_coord[1]) ** 2
    )
    if prevent_overlap:
        dist -= node_u_size - node_v_size

    if dist > 0:
        return scaling_ratio * node_u_mass * node_v_mass / dist / dist

    elif prevent_overlap and dist < 0:
        return 100* scaling_ratio * node_u_mass * node_v_mass

    return 0


def repulsion_region(node_u, region, scaling_ratio, prevent_overlap=False):
    return __repulsion_region(
        node_u.coord,
        node_u.mass,
        region.mass,
        region.massCenterX,
        region.massCenterY,
        scaling_ratio,
        prevent_overlap,
    )


@jit
def __repulsion_region(
        coord_node_u,
        mass_node_u,
        mass_region,
        mass_center_x,
        mass_center_y,
        scaling_ratio,
        prevent_overlap):
    dist = np.sqrt((coord_node_u[0] - mass_center_x) ** 2 +\
         (coord_node_u[1] - mass_center_y) ** 2)
    
    if not prevent_overlap and dist > 0:
            return scaling_ratio * (mass_node_u) * (mass_region) / dist/ dist
    if prevent_overlap:
        if dist > 0:
            return scaling_ratio * (mass_node_u) * (mass_region) /  dist / dist
        elif dist < 0:
            return -scaling_ratio * (mass_node_u) * (mass_region) /  dist

    return 0


def gravity(node_u, gravity, scaling_ratio, strong_gravity=False):
    return __gravity(node_u.coord, node_u.mass, scaling_ratio, gravity, strong_gravity)


@jit
def __gravity(node_u_coord, node_u_mass, scaling_ratio, gravity, strong_gravity: bool):
    dist = np.sqrt(node_u_coord[0] ** 2 + node_u_coord[1] ** 2)
    if dist > 0:
        if strong_gravity:
            return scaling_ratio * gravity * (node_u_mass)
        return (gravity * (node_u_mass) * scaling_ratio)/dist
    return 0



def adjust_speed(speed,nodes_attributes,jitter_tolerance=1.0,speed_efficiency=1.0):
    total_swinging = 0.0 
    total_effective_traction = 0.0
    min_speed_efficiency = 0.05
    graph_size = len(nodes_attributes)

    for _,n in nodes_attributes:
        swinging = np.sqrt((n.old_dx - n.dx) **2 + (n.old_dy - n.dy) **2)
        total_swinging += n.mass * swinging
        total_effective_traction += .5 * n.mass * np.sqrt(
            (n.old_dx + n.dx) **2 + (n.old_dy + n.dy) **2)


    estimated_optimal_jitter_tolerance = .05 * np.sqrt(len(nodes_attributes))
    min_JT,max_JT = np.sqrt(estimated_optimal_jitter_tolerance),10
    jt = jitter_tolerance * max(min_JT,
                               min(max_JT, estimated_optimal_jitter_tolerance * total_effective_traction / (
                                    graph_size**2)))

    
    if total_swinging / total_effective_traction > 2.0:
        if speed_efficiency > min_speed_efficiency:
            speed_efficiency *= .5
        jt = max(jt, jitter_tolerance)

    

    targetSpeed = jt * speed_efficiency * total_effective_traction / total_swinging

    if total_swinging > jt * total_effective_traction:
        if speed_efficiency > min_speed_efficiency:
            speed_efficiency *= .7
    elif speed < 1000:
        speed_efficiency *= 1.3

    speed += min(targetSpeed - speed, .5 * speed)
    return speed


