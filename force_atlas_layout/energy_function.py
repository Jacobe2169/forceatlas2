import numpy as np

from numba import njit


def euclidean_dist(x, y):
    return np.linalg.norm(x - y)


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


@njit
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

    weight = 1 if not weight else weight
    weight = -scaling_ratio * weight

    dist = np.sqrt(
        (node_u_coord[0] - node_v_coord[0]) ** 2
        + (node_u_coord[1] - node_v_coord[1]) ** 2
    )
    if prevent_overlap:
        dist -= node_u_size - node_v_size

    if dist > 0:
        f = weight
        if lin_log:
            f = (weight * np.log(1 + dist)) / dist
        if distributed:
            f /= node_u_mass
        return f
    return 1


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


@njit
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
    if prevent_overlap and dist <0:
        return 100 * scaling_ratio * node_u_mass * node_v_mass

    return 1


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


@njit
def __repulsion_region(
    coord_node_u,
    mass_node_u,
    mass_region,
    mass_center_x,
    mass_center_y,
    scaling_ratio,
    prevent_overlap):

    dist = np.sqrt((coord_node_u[0] - mass_center_x) ** 2 + (coord_node_u[1] - mass_center_y) ** 2)

    if not prevent_overlap:
        if dist > 0:
            return scaling_ratio * mass_node_u * mass_region / dist / dist
        return 1

    factor = scaling_ratio * mass_node_u * mass_region
    if dist > 0:
        return factor / dist / dist
    elif dist <0:
        return -factor / dist
        
    return 1

def gravity(node_u, gravity,scaling_ratio, strong_gravity=False):
    return __gravity(node_u.coord,node_u.mass,scaling_ratio, gravity, strong_gravity)


@njit
def __gravity(node_u_coord,node_u_mass,scaling_ratio,gravity, strong_gravity:bool):
    dist = np.sqrt(node_u_coord[0]**2 + node_u_coord[1]**2)
    if dist >0:
        if strong_gravity:
            return scaling_ratio * gravity * node_u_mass
        return gravity * node_u_mass *scaling_ratio /dist
    return 1