
import numpy as np
from .energy_function import repulsion_region,repulsion
from numba import jit, vectorize,njit


class RootRegion():
    REGION_LEFT=2000
    def __init__(self, nodes: list, nodes_attributes):
        
        self.nodes_attributes = nodes_attributes
        self.nodes = nodes

        self.massCenterX = 0
        self.massCenterY = 0
        self.size = 0
        self.mass = 0
        self.subregions : list [RootRegion] = []

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

            self.massCenterX = mass_sum_x / self.mass
            self.massCenterY = mass_sum_y / self.mass

            self.size = float('-inf')
            for n in self.nodes:
                dist = np.sqrt((self.nodes_attributes[n].x - self.massCenterX) ** 2 + (
                            self.nodes_attributes[n].y - self.massCenterY) ** 2)
                self.size = max(self.size, 2 * dist)





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
                        #RootRegion.REGION_LEFT-=1
                    else:
                        for n in node_grp:
                            self.subregions.append(RootRegion([n], self.nodes_attributes))
                            #RootRegion.REGION_LEFT-=1

            for region in self.subregions:
                #if RootRegion.REGION_LEFT >0:
                region.build_sub_region()

    def apply_force(self, n, barnes_hut_theta,scaling_ratio,prevent_overlap):
        nu = self.nodes_attributes[n]
        if len(self.nodes) <2 and len(self.subregions) >0:
            ni = self.nodes_attributes[self.nodes[0]]
            factor = repulsion(nu, ni, scaling_ratio=scaling_ratio, prevent_overlap=prevent_overlap)
            self.nodes_attributes.apply_r(n,self.subregions[0],factor)
        else:
            dist = np.sqrt((nu.x - self.massCenterX) ** 2 + (nu.y - self.massCenterY) **2)
            if dist * barnes_hut_theta > self.size:
                factor = repulsion_region(nu,self,scaling_ratio=scaling_ratio,prevent_overlap=prevent_overlap)
                self.nodes_attributes.apply_r(n,self,factor)
            else:
                for region in self.subregions:
                    region.apply_force(n,barnes_hut_theta,scaling_ratio,prevent_overlap)