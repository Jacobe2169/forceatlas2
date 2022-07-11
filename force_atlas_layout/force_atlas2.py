import numpy as np
import networkx as nx
from .utils import *


class RootRegion():
    def __init__(self, nodes,nodes_attributes):
        self.nodes_attributes = nodes_attributes
        self.nodes = nodes

        self.massCenterX = 0
        self.massCenterY = 0
        self.size = 0
        self.mass = 0
        self.subregions = []

        self.update_mass_geometry()

    def update_mass_geometry(self):
        if len(self.nodes) >1:
            self.mass =0
            massSumX = 0
            massSumY = 0
            for n in self.nodes:
                self.mass += self.nodes_attributes[n]["mass"]
                massSumX += self.nodes_attributes[n]["x"] * self.nodes_attributes[n]["mass"]
                massSumY += self.nodes_attributes[n]["y"] * self.nodes_attributes[n]["mass"]
            self.massCenterY = massSumY / self.mass
            self.massCenterX = massSumX / self.mass
        self.size = -1000000000.0
        for n in self.nodes:
            dist = np.sqrt((self.nodes_attributes[n]["x"]-self.massCenterX)**2 + (self.nodes_attributes[n]["x"]-self.massCenterY)**2)
            self.size = np.max([self.size,2*dist])

    def build_sub_region(self):
        if len(self.nodes) >1:
            left_nodes = []
            right_nodes = []
            for n in self.nodes:
                if self.nodes_attributes[n]["x"] < self.massCenterX:
                    left_nodes.append(n)
                else:
                    right_nodes.append(n)
            top_left_nodes = []
            bottom_left_nodes = []
            for n in left_nodes:
                if self.nodes_attributes[n]["y"] < self.massCenterY:
                    top_left_nodes.append(n)
                else:
                    bottom_left_nodes.append(n)

            top_right_nodes = []
            bottom_right_nodes = []
            for n in right_nodes:
                if self.nodes_attributes[n]["y"] < self.massCenterY:
                    top_right_nodes.append(n)
                else:
                    bottom_right_nodes.append(n)

            for node_grp in [top_right_nodes,top_left_nodes,bottom_right_nodes,bottom_left_nodes]:
                if len(node_grp) > 0:
                    if len(node_grp) < len(self.nodes):
                        self.subregions.append(RootRegion(node_grp,self.nodes_attributes))
                    else:
                        for n in node_grp:
                            self.subregions.append(RootRegion(n,self.nodes_attributes))
            for region in self.subregions:
                region.build_sub_region()


    def apply_force(self, n, repulsion_funtion:RepulsionForce, barnes_hut_theta):
        if len(self.nodes) <2 and len(self.subregions) >0:
            repulsion_funtion.apply_r(self.nodes[0],self.subregions[0],self.nodes_attributes)
        else:
            dist = np.sqrt((self.nodes_attributes[n]["x"] - self.massCenterX)**2 + (self.nodes_attributes[n]["y"] - self.massCenterY)**2)
            if dist * barnes_hut_theta > self.size :
                repulsion_funtion.apply_r(n,self,self.nodes_attributes)
            else:
                for region in self.subregions:
                    region.apply_force(n, repulsion_funtion, barnes_hut_theta)


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


class ForceAtlas2(object):

    def __init__(self,
                 graph: nx.Graph,
                 edge_weight_influence: float = 1,
                 jitter_tolerance: float = 1.0,
                 scaling_ratio: float = 2,
                 gravity: float = 1,
                 speed: float = 1,
                 speed_efficiency: float = 1,
                 outbound_attraction_distribution: bool = False,
                 adjust_sizes: bool = False,
                 barnes_hut_optimize: bool = False,
                 barnes_hut_theta: float = 1.2,
                 lin_log_mode: bool = False,
                 normalize_edge_weights: bool = False,
                 strong_gravity_mode: bool = False,
                 n_jobs: int = -1
                 ):

        self.root_region = None
        self.n_jobs = n_jobs
        self.strong_gravity_mode = strong_gravity_mode
        self.normalize_edge_weights = normalize_edge_weights
        self.lin_log_mode = lin_log_mode
        self.barnes_hut_theta = barnes_hut_theta
        self.barnes_hut_optimize = barnes_hut_optimize
        self.adjust_sizes = adjust_sizes
        self.outbound_attraction_distribution = outbound_attraction_distribution
        self.speed_efficiency = speed_efficiency
        self.speed = speed
        self.gravity = gravity
        self.scaling_ratio = scaling_ratio
        self.jitter_tolerance = jitter_tolerance
        self.edge_weight_influence = edge_weight_influence

        self.graph = graph
        for src,tar,attr in self.graph.edges(data=True):
            if not "weight" in attr:
                self.graph.edges[src,tar]["weight"] = 1

        self.nodes_attributes = {n: {} for n in self.graph}

        for node in self.graph:
            self.nodes_attributes[node] = {
                "mass": 1 + self.graph.degree(node),
                "old_dx": 0,
                "old_dy": 0,
                "dx": 0,
                "dy": 0,
                "x": np.random.rand()*5,
                "y": np.random.rand()*5,
                "size":self.graph.degree(node)
            }

        if self.normalize_edge_weights:
            edges_weights = list(nx.get_edge_attributes(self.graph, 'weight').values())
            self.weight_min = np.min(edges_weights)
            self.weight_max = np.max(edges_weights)

    def get_positions(self):
        positions = {}
        for n in self.graph:
            positions[n] = [self.nodes_attributes[n]["x"],self.nodes_attributes[n]["y"]]
        return positions
    def iteration(self):
        for node in self.graph:
            self.nodes_attributes[node].update({
                "mass": 1 + self.graph.degree(node),
                "old_dx": self.nodes_attributes[node]["dx"],
                "old_dy": self.nodes_attributes[node]["dy"],
                "dx": 0,
                "dy": 0,
            })
        if self.barnes_hut_optimize:
            self.root_region = RootRegion(list(self.graph.nodes()),self.nodes_attributes)
            self.root_region.build_sub_region()

        outbound_compensation = 0
        if self.outbound_attraction_distribution:
            for n in self.graph:
                outbound_compensation += self.nodes_attributes[n]["mass"]

            outbound_compensation /= len(self.graph)

        repulsion_function = factory_repulsion(self.adjust_sizes, self.scaling_ratio)
        gravity_function = factory_gravity(self.scaling_ratio) if self.strong_gravity_mode else repulsion_function

        # Repulsion
        if self.barnes_hut_optimize:
            for n in self.graph:
                self.root_region.apply_force(n, repulsion_function, self.barnes_hut_theta)
        else:
            for n1 in self.graph:
                for n2 in self.graph:
                    repulsion_function.apply(n1, n2,self.nodes_attributes)

        for n in self.graph:
            try:
                gravity_function.apply(n,self.gravity/self.scaling_ratio,self.nodes_attributes)
            except:
                gravity_function.apply_g(n, self.gravity / self.scaling_ratio, self.nodes_attributes)

        # Attraction
        attraction = attraction_factory(self.lin_log_mode, outbound_compensation, self.adjust_sizes,
                                        (outbound_compensation if self.outbound_attraction_distribution else 1))

        if self.edge_weight_influence == 0:
            for src, tar in self.graph.edges():
                attraction.apply(src,tar,1,self.nodes_attributes)

        elif self.edge_weight_influence == 1:
            if self.normalize_edge_weights:
                if self.weight_min < self.weight_max:
                    for src, tar, attr in self.graph.edges(data=True):
                        w = attr['weight']
                        w = (w - self.weight_min) / (self.weight_max - self.weight_min)
                        attraction.apply(src,tar,w,self.nodes_attributes)
                else:
                    for src, tar in self.graph.edges():
                        attraction.apply(src,tar,1,self.nodes_attributes)
            else:
                for src, tar, attr in self.graph.edges(data=True):
                    w = attr['weight']
                    attraction.apply(src,tar,w,self.nodes_attributes)
        else:
            if self.normalize_edge_weights:
                if self.weight_min < self.weight_max:
                    for src, tar, attr in self.graph.edges(data=True):
                        w = attr['weight']
                        w = (w - self.weight_min) / (self.weight_max - self.weight_min)
                        attraction.apply(src,tar,w,self.nodes_attributes)
                else:
                    for src, tar in self.graph.edges(data=True):
                        attraction.apply(src,tar,1,self.nodes_attributes)
            else:
                for src, tar, attr in self.graph.edges(data=True):
                    w = attr['weight']
                    attraction.apply(src,tar,w**(self.edge_weight_influence),self.nodes_attributes)

        total_swinging = 0.0
        total_effective_traction = 0.0

        for n in self.graph:
            swinging = np.sqrt((self.nodes_attributes[n]["old_dx"] - self.nodes_attributes[n]["dx"]) ** 2 + (
                        self.nodes_attributes[n]["old_dy"] - self.nodes_attributes[n]["dy"]) ** 2)
            total_swinging += swinging * self.nodes_attributes[n]["mass"]

            total_effective_traction += self.nodes_attributes[n]["mass"] * 0.5 * np.sqrt(
                (self.nodes_attributes[n]["old_dx"] + self.nodes_attributes[n]["dx"]) ** 2 + (
                            self.nodes_attributes[n]["old_dy"] + self.nodes_attributes[n]["dy"]) ** 2)

        estimated_optimal_jitter_tolerance = 0.05 * np.sqrt(len(self.graph))
        min_jt = np.sqrt(estimated_optimal_jitter_tolerance)
        max_jt = 10.0
        N =len(self.graph)
        c= (estimated_optimal_jitter_tolerance * total_effective_traction / N**2)
        b = np.min([max_jt,float(c)])
        a = np.max([min_jt, b])
        jt = self.jitter_tolerance * a

        min_speed_efficiency = 0.05

        if (total_swinging / total_effective_traction) > 2:
            if self.speed_efficiency > min_speed_efficiency:
                self.speed_efficiency *= 0.5
            jt = np.max([jt, self.jitter_tolerance])

        target_speed = jt * self.speed_efficiency * total_effective_traction / total_swinging

        if total_swinging > jt * total_effective_traction:
            if self.speed_efficiency > min_speed_efficiency:
                self.speed_efficiency *= 0.7
            elif self.speed < 1000:
                self.speed_efficiency *= 1.3

        max_rise = 0.5
        self.speed = self.speed + np.min([target_speed - self.speed, max_rise * self.speed])

        if self.adjust_sizes:
            for n in self.graph:
                swinging = self.nodes_attributes[n]["mass"] * np.sqrt(
                    (self.nodes_attributes[n]["old_dx"] - self.nodes_attributes[n]["dx"]) * (
                                self.nodes_attributes[n]["old_dx"] - self.nodes_attributes[n]["dx"]) +
                    (self.nodes_attributes[n]["old_dy"] - self.nodes_attributes[n]["dy"]) * (
                                self.nodes_attributes[n]["old_dy"] - self.nodes_attributes[n]["dy"]))
                factor = 0.1 * self.speed / (1.0 + np.sqrt(self.speed * swinging))

                df = np.sqrt(self.nodes_attributes[n]["dx"] ** 2 + self.nodes_attributes[n]["dy"] ** 2)
                factor = np.min([factor * df, 10.]) / df

                x = self.nodes_attributes[n]["x"] + self.nodes_attributes[n]["dx"] * factor
                y = self.nodes_attributes[n]["y"] + self.nodes_attributes[n]["dy"] * factor

                self.nodes_attributes[n]["x"] = x
                self.nodes_attributes[n]["y"] = y
        else:
            for n in self.graph:
                swinging = self.nodes_attributes[n]["mass"] * np.sqrt(
                    (self.nodes_attributes[n]["old_dx"] - self.nodes_attributes[n]["dx"]) * (
                                self.nodes_attributes[n]["old_dx"] - self.nodes_attributes[n]["dx"]) +
                    (self.nodes_attributes[n]["old_dy"] - self.nodes_attributes[n]["dy"]) * (
                                self.nodes_attributes[n]["old_dy"] - self.nodes_attributes[n]["dy"]))
                factor = 0.1 * self.speed / (1.0 + np.sqrt(self.speed * swinging))

                x = self.nodes_attributes[n]["x"] + self.nodes_attributes[n]["dx"] * factor
                y = self.nodes_attributes[n]["y"] + self.nodes_attributes[n]["dy"] * factor

                self.nodes_attributes[n]["x"] = x
                self.nodes_attributes[n]["y"] = y
