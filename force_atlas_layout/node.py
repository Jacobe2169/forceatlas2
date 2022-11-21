import numpy as np
from .region import RootRegion


class Node():
    def __init__(self, id, x, y, dx, dy, size, mass):
        self.id = id
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.size = size
        self.mass = mass

        self.old_dx = 0
        self.old_dy = 0
        self.convergence = 1
        
    @property
    def coord(self):
        return np.array([self.x,self.y])

class NodeCollection():
    def __init__(self, nodes={}):
        self.nodes = nodes

    def __add__(self, othernode):
        assert type(othernode) == Node
        self.nodes[othernode.id] = othernode

    def __getitem__(self, id_):
        return self.nodes[id_]

    def apply(self, u, v, factor):
        nu = self.nodes[u]
        nv = self.nodes[v]
        x_dist = nu.x - nv.x
        y_dist = nu.y - nv.y
        self.nodes[u].dx += factor * x_dist
        self.nodes[u].dy += factor * y_dist

        self.nodes[v].dx -= factor * x_dist
        self.nodes[v].dy -= factor * y_dist

    def apply_r(self,u,region : RootRegion,factor):
        nu = self.nodes[u]
        x_dist = nu.x - region.massCenterX
        y_dist = nu.y - region.massCenterY

        self.nodes[u].dx += factor * x_dist
        self.nodes[u].dy += factor * y_dist

    def apply_g(self,u,gravity_factor):
        nu = self.nodes[u]
        x_dist = nu.x
        y_dist = nu.y 

        self.nodes[u].dx += gravity_factor * x_dist
        self.nodes[u].dy += gravity_factor * y_dist