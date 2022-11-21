import numpy as np
import networkx as nx

from .node import NodeCollection, Node

# from .utils import *
from .region import RootRegion
from .energy_function import attraction, repulsion, gravity

import random
from joblib import Parallel,delayed


class ForceAtlas2(object):
    def __init__(
        self,
        graph: nx.Graph,
        edge_weight_influence: float = 10,
        jitter_tolerance: float = 1.0,
        scaling_ratio: float = 2.0,
        gravity: float = 1.0,
        speed: float = 1.0,
        speed_efficiency: float = 1.0,
        outbound_attraction_distribution: bool = False,
        adjust_sizes: bool = False,
        barnes_hut_optimize: bool = False,
        barnes_hut_theta: float = 1.2,
        lin_log_mode: bool = False,
        normalize_edge_weights: bool = False,
        strong_gravity_mode: bool = False,
        n_jobs: int = -1,
        positions={},
        sizes={},
    ):

        self.root_region = None
        self.n_jobs = n_jobs
        self.strong_gravity_mode = strong_gravity_mode
        self.normalize_edge_weights = normalize_edge_weights
        self.lin_log_mode = lin_log_mode
        self.barnes_hut_theta = barnes_hut_theta
        self.barnes_hut_optimize = barnes_hut_optimize
        self.prevent_overlap = adjust_sizes
        self.outbound_attraction_distribution = outbound_attraction_distribution
        self.speed_efficiency = speed_efficiency
        self.speed = speed
        self.gravity = gravity
        self.scaling_ratio = scaling_ratio
        self.jitter_tolerance = jitter_tolerance
        self.edge_weight_influence = edge_weight_influence


        self.graph = graph

        

        # if no weight associated to an edge, set its value to min
        min_weight = np.inf
        edges_weights = list(nx.get_edge_attributes(self.graph, "weight").values())
        if len(edges_weights) == 0:
            min_weight = 1
        else:
            min_weight = np.min(edges_weights)

        for src, tar, attr in self.graph.edges(data=True):
            if not "weight" in attr:
                self.graph.edges[src, tar]["weight"] = min_weight

        # Initialize node attributes
        self.nodes_attributes = NodeCollection()

        
        for node in self.graph:
            param = {
                "id": node,
                "dx": 0,
                "dy": 0,
                "mass": 1 + self.graph.degree(node),
                "x": np.random.rand()*100 if not node in positions else positions[node][0],
                "y": np.random.rand()*100 if not node in positions else positions[node][1],
                "size": self.graph.degree(node) if not node in sizes else sizes[node],
            }
            self.nodes_attributes + Node(**param)
        
        self.root_region: RootRegion = None

        # if normalization activated, we pre-compute edge weights' minimum and maximum
        if self.normalize_edge_weights:
            edges_weights = list(nx.get_edge_attributes(self.graph, "weight").values())
            self.weight_min = np.min(edges_weights)
            self.weight_max = np.max(edges_weights)

    def get_positions(self):
        """
        Return computed positions of the graph's node using
        the ForceAtlas2 algorithm

        Returns
        -------
        dict
            dict with key corresponding to node id and the value corresponding
            to its positions in a 2D space
        """
        positions = {}
        for n in self.graph:
            positions[n] = [self.nodes_attributes[n].x, self.nodes_attributes[n].y]
        return positions

    def iteration(self):
        """
        Update the positions of the nodes by applying the Force2Atlas algorithm.
        In order to exploit the potential of the ForceAtlas2 algorithm, it's common to run
        multiple times the algorithm
        """

        # Update nodes attributes by storing previous state information and reinitialize
        # node mass
        for node in self.graph:
            self.nodes_attributes[node].mass = 1 + self.graph.degree(node)
            self.nodes_attributes[node].old_dx = self.nodes_attributes[node].dx
            self.nodes_attributes[node].old_dy = self.nodes_attributes[node].dy

            self.nodes_attributes[node].dx = 0
            self.nodes_attributes[node].dy = 0

        # If Barnes Hut active, initialize root region
        if self.barnes_hut_optimize:
            self.root_region = RootRegion(
                list(self.graph.nodes()), self.nodes_attributes
            )
            self.root_region.build_sub_region()

        # If outbound_attraction_distribution active, compensate
        outbound_compensation = 0
        if self.outbound_attraction_distribution:
            for n in self.graph:
                outbound_compensation += self.nodes_attributes[n].mass

            outbound_compensation /= len(self.graph)

        # Apply Repulsion
        if self.barnes_hut_optimize:
            for n in self.graph:
                self.root_region.apply_force(
                    n, self.barnes_hut_theta, self.scaling_ratio, self.prevent_overlap
                )
        else:
            for n1 in self.graph:
                for n2 in self.graph:
                    if n1 == n2:
                        continue
                    factor = repulsion(
                        self.nodes_attributes[n1],
                        self.nodes_attributes[n2],
                        self.scaling_ratio,
                        prevent_overlap=self.prevent_overlap,
                    )
                    self.nodes_attributes.apply(n1, n2, factor)

        # Apply Gravity
        for n in self.graph:
            factor = gravity(self.nodes_attributes[n], gravity=self.gravity/self.scaling_ratio,scaling_ratio=self.scaling_ratio,strong_gravity=self.strong_gravity_mode)
            self.nodes_attributes.apply_g(n, factor)

        for src, tar, attr in self.graph.edges(data=True):  # type: ignore
            w = None
            if self.edge_weight_influence > 0:
                w = attr["weight"]
                if self.normalize_edge_weights:
                    w = (w - self.weight_min) / (self.weight_max - self.weight_min)

            factor = attraction(
                self.nodes_attributes[src],
                self.nodes_attributes[tar],
                outbound_compensation if self.outbound_attraction_distribution else 1,
                self.lin_log_mode,
                self.prevent_overlap,
                weight=w,distributed=self.outbound_attraction_distribution
            )
            self.nodes_attributes.apply(src, tar, factor)

        # Adjust speed automatically
        total_swinging = 0.0
        total_effective_traction = 0.0

        for n in self.graph:
            swinging = np.sqrt(
                (self.nodes_attributes[n].old_dx - self.nodes_attributes[n].dx) ** 2
                + (self.nodes_attributes[n].old_dy - self.nodes_attributes[n].dy) ** 2
            )
            total_swinging += swinging * self.nodes_attributes[n].mass

            total_effective_traction += (
                self.nodes_attributes[n].mass
                * 0.5
                * np.sqrt(
                    (self.nodes_attributes[n].old_dx + self.nodes_attributes[n].dx) ** 2
                    + (self.nodes_attributes[n].old_dy + self.nodes_attributes[n].dy) ** 2
                )
            )

        #  swinging_movement shoud be < to tolerance * convergence moment
        # optimize jitter tolerance

        estimated_optimal_jitter_tolerance = 0.05 * np.sqrt(len(self.graph))
        min_jt = np.sqrt(estimated_optimal_jitter_tolerance)
        max_jt = 10.0
        N = len(self.graph)
        c = estimated_optimal_jitter_tolerance * total_effective_traction / N**2
        b = np.min([max_jt, float(c)])
        a = np.max([min_jt, b])
        jt = self.jitter_tolerance * a

        min_speed_efficiency = 0.05

        # Protection against erratic behavior
        if (total_swinging / total_effective_traction) > 2.0:
            if self.speed_efficiency > min_speed_efficiency:
                self.speed_efficiency *= 0.5
            jt = np.max([jt, self.jitter_tolerance])

        target_speed = (
            jt * self.speed_efficiency * total_effective_traction / total_swinging
        )

        # Speed efficiency is how the speed really corresponds to the swinging vs. convergence tradeoff
        if total_swinging > jt * total_effective_traction:
            if self.speed_efficiency > min_speed_efficiency:
                self.speed_efficiency *= 0.7
            elif self.speed < 1000:
                self.speed_efficiency *= 1.3

        # but not too much !
        max_rise = 0.5
        self.speed = self.speed + np.min(
            [target_speed - self.speed, max_rise * self.speed]
        )

        # Apply forces to avoid nodes overlapping

        for n in self.graph:
            swinging = self.nodes_attributes[n].mass * np.sqrt(
                (self.nodes_attributes[n].old_dx - self.nodes_attributes[n].dx) ** 2
                + (self.nodes_attributes[n].old_dy - self.nodes_attributes[n].dy) ** 2
            )

            if self.prevent_overlap:
                factor = 0.1 * self.speed / (1.0 + np.sqrt(self.speed * swinging))

                df = np.sqrt(
                    self.nodes_attributes[n].dx ** 2 + self.nodes_attributes[n].dy ** 2
                )
                factor = np.min([factor * df, 1.0]) / df
            else:
                factor = self.speed / (1.0 + np.sqrt(self.speed * swinging))

            self.nodes_attributes[n].x = (
                self.nodes_attributes[n].x + self.nodes_attributes[n].dx * factor
            )
            self.nodes_attributes[n].y = (
                self.nodes_attributes[n].y + self.nodes_attributes[n].dy * factor
            )
