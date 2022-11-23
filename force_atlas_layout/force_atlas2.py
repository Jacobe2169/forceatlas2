import numpy as np
import networkx as nx

from .node import NodeCollection, Node

# from .utils import *
from .region import RootRegion
from .energy_function import attraction, repulsion, gravity

import random
from joblib import Parallel,delayed




class ForceAtlas2(object):
    """
    Main class that runs the ForceAtlas2 algorithm
    """
    def __init__(
        self,
        graph: nx.Graph,
        edge_weight_influence: float = 1,
        scaling_ratio: float = 2.0,
        gravity: float = 1.0,
        speed: float = 1.0,
        max_force:float = 20.,
        quadtree_maxsize:int = 1000,
        outbound_attraction_distribution: bool = False,
        prevent_overlap: bool = True,
        barnes_hut_optimize: bool = False,
        barnes_hut_theta: float = 1.2,
        lin_log_mode: bool = False,
        normalize_edge_weights: bool = False,
        strong_gravity_mode: bool = False,
        n_jobs: int = -1,
        positions={},
        sizes={},
    ):

        """

        Parameters
        ----------
        graph : nx.Graph
            input graph
        edge_weight_influence : int
            influence of edges' weight in the spatialisation
        scaling_ratio : float
            force coefficient
        gravity : float
            gravity
        speed : float
            coefficient applied to nodes' movement vector each iteration
        max_force : float|int
            maximum magnitude of a movement vector
        quadtree_maxsize : int
            maximum number of nodes in the quadtree used when `barnes_hut_optimize = True`
        outbound_attraction_distribution: bool
            use a node mass (=degree) in the attraction computation
        prevent_overlap : bool
            prevent node overlapping (use node size)
        barnes_hut_optimize : bool
            enable fast computation of ForceAtlas2 by subdividing the graph into a Quadtree. Each node's new position is
             computed based on nodes in the Quadtree where it is.
        barnes_hut_theta : float
            constant (by default set to 1.2)
        lin_log_mode : bool
            multiply the attraction force by the logarithm of the distance
        normalize_edge_weights : bool
            if True, normalize edge weights. Formula w = (w-min)/(max-min)
        strong_gravity_mode : bool
            enable strong gravity mode
        n_jobs : int
            number of thread use in the computation (not implement for now)
        positions : dict
            dictionary containing initial positions for the graph's nodes. Initialized with random values if empty
        sizes : dict
            dictionary that contains each node's size (by default, degree of each node)
        """
        self.root_region = None
        self.n_jobs = n_jobs
        self.strong_gravity_mode = strong_gravity_mode
        self.normalize_edge_weights = normalize_edge_weights
        self.lin_log_mode = lin_log_mode
        self.barnes_hut_theta = barnes_hut_theta
        self.barnes_hut_optimize = barnes_hut_optimize
        self.prevent_overlap = prevent_overlap
        self.outbound_attraction_distribution = outbound_attraction_distribution
        self.speed = speed
        self.gravity = gravity
        self.scaling_ratio = scaling_ratio
        self.edge_weight_influence = edge_weight_influence
        self.max_force = max_force
        self.quadtree_maxsize = quadtree_maxsize


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
        In order to exploit the potential of the ForceAtlas2 algorithm, it's common to run the algorithm 
        multiple times 
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
            # limit the quadtree size for performance issue
            RootRegion.REGION_LEFT = self.quadtree_maxsize
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
            w = 1
            if self.edge_weight_influence > 0:
                w = attr["weight"]
                if self.normalize_edge_weights:
                    w = (w - self.weight_min) / (self.weight_max - self.weight_min)

            factor = attraction(
                self.nodes_attributes[src],
                self.nodes_attributes[tar],
                outbound_compensation if self.outbound_attraction_distribution else 1,
                lin_log=self.lin_log_mode,
                prevent_overlap=self.prevent_overlap,
                weight=w,distributed=self.outbound_attraction_distribution
            )
            self.nodes_attributes.apply(src, tar, factor)

        # Adjust speed automatically
        for n in self.graph:
            force = np.sqrt(self.nodes_attributes[n].dx ** 2 + self.nodes_attributes[n].dy ** 2 )
            if force > self.max_force:
                self.nodes_attributes[n].dx = (self.nodes_attributes[n].dx *self.max_force)/force
                self.nodes_attributes[n].dy = (self.nodes_attributes[n].dy *self.max_force)/force

            swinging = np.sqrt(
                (self.nodes_attributes[n].old_dx - self.nodes_attributes[n].dx) ** 2
                + (self.nodes_attributes[n].old_dy - self.nodes_attributes[n].dy) ** 2
            )* self.nodes_attributes[n].mass
           

            traction= 0.5* np.sqrt(
                    (self.nodes_attributes[n].old_dx + self.nodes_attributes[n].dx) ** 2
                    + (self.nodes_attributes[n].old_dy + self.nodes_attributes[n].dy) ** 2
                )
            
            node_speed = 1
            if self.prevent_overlap:
                node_speed = (0.1 * np.log(1 + traction)) / (1 + np.sqrt(swinging))
            else:
                node_speed = self.nodes_attributes[n].convergence * np.log(1+traction)/(1+np.sqrt(swinging))
                self.nodes_attributes[n].convergence = min(
                    1, np.sqrt(node_speed*(self.nodes_attributes[n].dx**2 + self.nodes_attributes[n].dy**2)/(1+np.sqrt(swinging)))
                )

            self.nodes_attributes[n].x = (
                self.nodes_attributes[n].x + self.nodes_attributes[n].dx * node_speed * self.speed
            )
            self.nodes_attributes[n].y = (
                self.nodes_attributes[n].y + self.nodes_attributes[n].dy * node_speed * self.speed
            )