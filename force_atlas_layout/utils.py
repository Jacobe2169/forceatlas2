import numpy as np
from numba import jit, vectorize
from numba.experimental import jitclass
from numba.typed import Dict, listobject
from numba import types


def euclidean_distance(x, y):
    x_dist =  x[0] - y[0]
    y_dist =  x[1] - y[1]
    return x_dist, y_dist, np.sqrt(x_dist ** 2 + y_dist ** 2)


class AttractionForce:
    def apply(self, u, v, weight, node_attributes):
        raise NotImplemented


class RepulsionForce:

    def apply(self, u, v, node_attributes):
        raise NotImplemented

    def apply_r(self, u, region, node_attributes):
        raise NotImplemented

    def apply_g(self, u, gravity, node_attributes):
        raise NotImplemented


class RootRegion():
    def __init__(self, nodes: list, nodes_attributes):
        self.nodes_attributes = nodes_attributes
        self.nodes = nodes

        self.massCenterX = 0
        self.massCenterY = 0
        self.size = 0
        self.mass = 0
        self.subregions = []

        self.update_mass_geometry()


    def update_mass_geometry(self):
        if len(self.nodes) > 1:
            self.mass = 0
            mass_sum_x = 0
            mass_sum_y = 0
            for n in self.nodes:
                self.mass += self.nodes_attributes[n].mass
                mass_sum_x += self.nodes_attributes[n].x * self.nodes_attributes[n].mass
                mass_sum_y += self.nodes_attributes[n].y * self.nodes_attributes[n].mass

            self.massCenterY = mass_sum_y / self.mass
            self.massCenterX = mass_sum_x / self.mass

            self.size = float('-inf')
            for n in self.nodes:
                dist = np.sqrt((self.nodes_attributes[n].x - self.massCenterX) ** 2 + (
                            self.nodes_attributes[n].y - self.massCenterY) ** 2)
                self.size = np.max([self.size, 2 * dist])





    def build_sub_region(self):
        if len(self.nodes) > 1:
            top_left_nodes,bottom_left_nodes,top_right_nodes,bottom_right_nodes = [],[],[],[]

            for n in self.nodes:
                if self.nodes_attributes[n].x < self.massCenterX:
                    if self.nodes_attributes[n].y > self.massCenterY:
                        top_left_nodes.append(n)
                    else:
                        bottom_left_nodes.append(n)
                else:
                    if self.nodes_attributes[n].y > self.massCenterY:
                        top_right_nodes.append(n)
                    else:
                        bottom_right_nodes.append(n)


            for node_grp in [top_right_nodes, top_left_nodes, bottom_right_nodes, bottom_left_nodes]:
                if len(node_grp) > 0:
                    if len(node_grp) < len(self.nodes):
                        self.subregions.append(RootRegion(node_grp, self.nodes_attributes))
                    else:
                        for n in node_grp:
                            self.subregions.append(RootRegion(n, self.nodes_attributes))

            for region in self.subregions:
                region.build_sub_region()

    def apply_force(self, n, repulsion_funtion: RepulsionForce, barnes_hut_theta):
        if len(self.nodes) < 2 and len(self.subregions) > 0:
            repulsion_funtion.apply_r(self.nodes[0], self.subregions[0], self.nodes_attributes)
        else:
            dist = np.sqrt((self.nodes_attributes[n].x - self.massCenterX) ** 2 + (
                        self.nodes_attributes[n].y - self.massCenterY) ** 2)
            if dist * barnes_hut_theta > self.size:
                repulsion_funtion.apply_r(n, self, self.nodes_attributes)
            else:
                for region in self.subregions:
                    region.apply_force(n, repulsion_funtion, barnes_hut_theta)


class LinRepulsion(RepulsionForce):
    def __init__(self, coefficient):
        RepulsionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, node_attributes):
        x_dist, y_dist, dist = euclidean_distance([node_attributes[u].x, node_attributes[u].y],
                                                  [node_attributes[v].x, node_attributes[v].y])
        if dist > 0:
            f = self.c * node_attributes[u].mass * node_attributes[v].mass / dist/ dist
            node_attributes[u].dx += x_dist * f
            node_attributes[u].dy += y_dist * f

            node_attributes[v].dx -= x_dist * f
            node_attributes[v].dy -= y_dist * f

    def apply_r(self, u, region, node_attributes):
        mass_center_x, mass_center_y = region.massCenterX, region.massCenterY
        x_dist = node_attributes[u].x - mass_center_x
        y_dist = node_attributes[u].y - mass_center_y
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
        if dist > 0:
            f = self.c * node_attributes[u].mass * region.mass / dist / dist
            node_attributes[u].dx += x_dist * f
            node_attributes[u].dy += y_dist * f

    def apply_g(self, u, gravity, node_attributes):
        x_dist = node_attributes[u].x
        y_dist = node_attributes[u].y
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
        if dist > 0:
            f = self.c * node_attributes[u].mass * gravity / dist
            node_attributes[u].dx -= x_dist * f
            node_attributes[u].dy -= y_dist * f


class LinRepulsionwithAntiCollision(LinRepulsion):

    def __init__(self, coefficient):
        LinRepulsion.__init__(self, coefficient)

    def apply(self, u, v, node_attributes):
        x_dist, y_dist, dist = euclidean_distance([node_attributes[u].x, node_attributes[u].y],
                                                  [node_attributes[v].x, node_attributes[v].y])
        dist = dist- node_attributes[u].size - node_attributes[v].size
        if dist > 0:
            f = self.c * node_attributes[u].mass * node_attributes[v].mass / dist / dist
            node_attributes[u].dx += x_dist * f
            node_attributes[u].dy += y_dist * f

            node_attributes[v].dx -= x_dist * f
            node_attributes[v].dy -= y_dist * f
        elif dist < 0:
            f = 100 * self.c * node_attributes[u].mass * node_attributes[v].mass
            node_attributes[u].dx += x_dist * f
            node_attributes[u].dy += y_dist * f

            node_attributes[v].dx -= x_dist * f
            node_attributes[v].dy -= y_dist * f

    def apply_r(self, u, region, node_attributes):
        mass_center_x, mass_center_y = region.massCenterX, region.massCenterY
        x_dist = node_attributes[u].x - mass_center_x
        y_dist = node_attributes[u].y - mass_center_y
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
        if dist > 0:
            f = self.c * node_attributes[u].mass * region.mass / dist / dist
            node_attributes[u].dx += x_dist * f
            node_attributes[u].dy += y_dist * f

        elif dist < 0:
            f = -self.c * node_attributes[u].mass * region.mass / dist
            node_attributes[u].dx += x_dist * f
            node_attributes[u].dy += y_dist * f


class StrongGravity(RepulsionForce):

    def __init__(self, coefficient):
        RepulsionForce.__init__(self)
        self.c = coefficient

    def apply_g(self, u, gravity, node_attributes):
        x_dist = node_attributes[u].x
        y_dist = node_attributes[u].y
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
        if dist > 0:
            f = self.c * node_attributes[u].mass * gravity
            node_attributes[u].dx -= x_dist * f
            node_attributes[u].dy -= y_dist * f


class LinAttraction(AttractionForce):
    def __init__(self, coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist, y_dist, _ = euclidean_distance([node_attributes[u].x, node_attributes[u].y],
                                               [node_attributes[v].x, node_attributes[v].y])
        f = -self.c * weight
        node_attributes[u].dx += x_dist * f
        node_attributes[u].dy += y_dist * f

        node_attributes[v].dx -= x_dist * f
        node_attributes[v].dy -= y_dist * f


class LinAttractionMassDistributed(AttractionForce):
    def __init__(self, coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist, y_dist, _ = euclidean_distance([node_attributes[u].x, node_attributes[u].y],
                                               [node_attributes[v].x, node_attributes[v].y])
        f = -self.c * weight / node_attributes[u].mass
        node_attributes[u].dx += x_dist * f
        node_attributes[u].dy += y_dist * f

        node_attributes[v].dx -= x_dist * f
        node_attributes[v].dy -= y_dist * f


class LogAttraction(AttractionForce):
    def __init__(self, coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist, y_dist, dist = euclidean_distance([node_attributes[u].x, node_attributes[u].y],
                                                  [node_attributes[v].x, node_attributes[v].y])
        if dist > 0:
            f = -self.c * weight * np.log(1 + dist) / dist
            node_attributes[u].dx += x_dist * f
            node_attributes[u].dy += y_dist * f

            node_attributes[v].dx -= x_dist * f
            node_attributes[v].dy -= y_dist * f


class LogAttractionDegreeDistributed(AttractionForce):
    def __init__(self, coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist, y_dist, dist = euclidean_distance([node_attributes[u].x, node_attributes[u].y],
                                                  [node_attributes[v].x, node_attributes[v].y])
        if dist > 0:
            f = -self.c * weight * np.log(1 + dist) / dist / node_attributes[u].mass
            node_attributes[u].dx += x_dist * f
            node_attributes[u].dy += y_dist * f

            node_attributes[v].dx -= x_dist * f
            node_attributes[v].dy -= y_dist * f


class LinAttractionAntiCollision(AttractionForce):
    def __init__(self, coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist, y_dist, dist = euclidean_distance([node_attributes[u].x, node_attributes[u].y],
                                                  [node_attributes[v].x, node_attributes[v].y])
        dist = dist - node_attributes[u].size - node_attributes[v].size
        if dist > 0:
            f = -self.c * weight
            node_attributes[u].dx += x_dist * f
            node_attributes[u].dy += y_dist * f

            node_attributes[v].dx -= x_dist * f
            node_attributes[v].dy -= y_dist * f


class LinAttractionAntiCollisionDegreeDistributed(AttractionForce):
    def __init__(self, coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist, y_dist, dist = euclidean_distance([node_attributes[u].x, node_attributes[u].y],
                                                  [node_attributes[v].x, node_attributes[v].y])
        dist = dist - node_attributes[u].size - node_attributes[v].size
        if dist > 0:
            f = -self.c * weight / node_attributes[u].mass
            node_attributes[u].dx += x_dist * f
            node_attributes[u].dy += y_dist * f

            node_attributes[v].dx -= x_dist * f
            node_attributes[v].dy -= y_dist * f


class LogAttractionAntiCollision(AttractionForce):
    def __init__(self, coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist, y_dist, dist = euclidean_distance([node_attributes[u].x, node_attributes[u].y],
                                                  [node_attributes[v].x, node_attributes[v].y])

        dist = dist- node_attributes[u].size - node_attributes[v].size

        if dist > 0:
            f = -self.c * weight * np.log(1 + dist) / dist
            node_attributes[u].dx += x_dist * f
            node_attributes[u].dy += y_dist * f

            node_attributes[v].dx -= x_dist * f
            node_attributes[v].dy -= y_dist * f


class LogAttractionAntiCollisionDegreeDistributed(AttractionForce):
    def __init__(self, coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist, y_dist, dist = euclidean_distance(node_attributes[u].coord,node_attributes[v].coord)

        dist = dist - node_attributes[u].size + node_attributes[v].size

        if dist > 0:
            f = -self.c * weight * np.log(1 + dist) / dist / node_attributes[u].mass
            node_attributes[u].dx += x_dist * f
            node_attributes[u].dy += y_dist * f

            node_attributes[v].dx -= x_dist * f
            node_attributes[v].dy -= y_dist * f

def factory_repulsion(is_adjust_size, scaling_ratio):
    if is_adjust_size:
        return LinRepulsionwithAntiCollision(scaling_ratio)
    return LinRepulsion(scaling_ratio)


def factory_gravity(coefficient):
    return StrongGravity(coefficient)


def attraction_factory(lin_log_mode, distributed, adjust_sizes, coefficient):
    if adjust_sizes:
        if lin_log_mode:
            if distributed:
                return LogAttractionAntiCollisionDegreeDistributed(coefficient)
            return LogAttractionAntiCollision(coefficient)
        else:
            if distributed:
                return LinAttractionAntiCollisionDegreeDistributed(coefficient)
            return LinAttractionAntiCollision(coefficient)
    else:
        if lin_log_mode:
            if distributed:
                return LogAttractionDegreeDistributed(coefficient)
            return LogAttraction(coefficient)
        else:
            if distributed:
                return LinAttractionMassDistributed(coefficient)
            return LinAttraction(coefficient)
