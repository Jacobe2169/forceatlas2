import numpy as np

def euclidean_distance(x,y):
    x_dist = x[0]-y[0]
    y_dist = x[1]-y[1]
    return x_dist,y_dist,np.sqrt(x_dist**2 + y_dist**2)


class AttractionForce:
    def apply(self,u,v,weight,node_attributes):
        raise NotImplemented

class RepulsionForce:
    def apply(self,u,v,node_attributes):
        raise NotImplemented
    def apply_r(self,u,region,node_attributes):
        raise NotImplemented

    def apply_g(self,u,gravity,node_attributes):
        raise NotImplemented


class LinRepulsion(RepulsionForce):
    def __init__(self,coefficient):
        RepulsionForce.__init__(self)
        self.c = coefficient

    def apply(self,u,v,node_attributes):
        x_dist ,y_dist, dist  = euclidean_distance([node_attributes[u]["x"],node_attributes[u]["y"]],[node_attributes[v]["x"],node_attributes[v]["y"]])
        if dist > 0:
            f = self.c * node_attributes[u]["mass"] * node_attributes[v]["mass"] / dist / dist
            node_attributes[u]["dx"] += x_dist * f
            node_attributes[u]["dy"] += y_dist * f

            node_attributes[v]["dx"] -= x_dist * f
            node_attributes[v]["dy"] -= y_dist * f

    def apply_r(self, u, region, node_attributes):
        mass_center_x,mass_center_y = region.massCenterX,region.massCenterY
        x_dist = node_attributes[u]["x"] - mass_center_x
        y_dist = node_attributes[u]["y"] - mass_center_y
        dist = np.sqrt(x_dist**2 + y_dist **2)
        if dist >0:
            f = self.c * node_attributes[u]["mass"] * region.mass /dist /dist
            node_attributes[u]["dx"] += x_dist * f
            node_attributes[u]["dy"] += y_dist * f

    def apply_g(self,u,gravity,node_attributes):
        x_dist = node_attributes[u]["x"]
        y_dist = node_attributes[u]["y"]
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
        if dist > 0:
            f = self.c * node_attributes[u]["mass"] * gravity / dist
            node_attributes[u]["dx"] -= x_dist * f
            node_attributes[u]["dy"] -= y_dist * f

class LinRepulsionwithAntiCollision(LinRepulsion):

    def __init__(self,coefficient):
        LinRepulsion.__init__(self,coefficient)

    def apply(self,u,v,node_attributes):
        x_dist ,y_dist, dist  = euclidean_distance([node_attributes[u]["x"],node_attributes[u]["y"]],[node_attributes[v]["x"],node_attributes[v]["y"]])
        dist -= node_attributes[u]["size"] + node_attributes[v]["size"]
        if dist > 0:
            f = self.c * node_attributes[u]["mass"] * node_attributes[v]["mass"] / dist / dist
            node_attributes[u]["dx"] += x_dist * f
            node_attributes[u]["dy"] += y_dist * f

            node_attributes[v]["dx"] -= x_dist * f
            node_attributes[v]["dy"] -= y_dist * f
        elif dist < 0 :
            f = 100 * self.c * node_attributes[u]["mass"] * node_attributes[v]["mass"]
            node_attributes[u]["dx"] += x_dist * f
            node_attributes[u]["dy"] += y_dist * f

            node_attributes[v]["dx"] -= x_dist * f
            node_attributes[v]["dy"] -= y_dist * f

    def apply_r(self, u, region, node_attributes):
        mass_center_x,mass_center_y = region.massCenterX,region.massCenterY
        x_dist = node_attributes[u]["x"] - mass_center_x
        y_dist = node_attributes[u]["y"] - mass_center_y
        dist = np.sqrt(x_dist**2 + y_dist **2)
        if dist >0:
            f = self.c * node_attributes[u]["mass"] * region.mass /dist /dist
            node_attributes[u]["dx"] += x_dist * f
            node_attributes[u]["dy"] += y_dist * f

        elif dist <0:
            f = -self.c * node_attributes[u].mass * mass_center_x/dist
            node_attributes[u]["dx"] += x_dist * f
            node_attributes[u]["dy"] += y_dist * f


class StrongGravity(RepulsionForce):

    def __init__(self,coefficient):
        RepulsionForce.__init__(self)
        self.c = coefficient

    def apply_g(self,u,gravity,node_attributes):
        x_dist = node_attributes[u]["x"]
        y_dist = node_attributes[u]["y"]
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
        if dist > 0:
            f = self.c * node_attributes[u]["mass"] * gravity
            node_attributes[u]["dx"] -= x_dist * f
            node_attributes[u]["dy"] -= y_dist * f

class LinAttraction(AttractionForce):
    def __init__(self,coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist,y_dist, _ =euclidean_distance([node_attributes[u]["x"],node_attributes[u]["y"]],[node_attributes[v]["x"],node_attributes[v]["y"]])
        f = -self.c * weight
        node_attributes[u]["dx"] += x_dist * f
        node_attributes[u]["dy"] += y_dist * f

        node_attributes[v]["dx"] -= x_dist * f
        node_attributes[v]["dy"] -= y_dist * f

class LinAttractionMassDistributed(AttractionForce):
    def __init__(self,coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist,y_dist, _ = euclidean_distance([node_attributes[u]["x"],node_attributes[u]["y"]],[node_attributes[v]["x"],node_attributes[v]["y"]])
        f = -self.c * weight / node_attributes[u]["mass"]
        node_attributes[u]["dx"] += x_dist * f
        node_attributes[u]["dy"] += y_dist * f

        node_attributes[v]["dx"] -= x_dist * f
        node_attributes[v]["dy"] -= y_dist * f

class LogAttraction(AttractionForce):
    def __init__(self,coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist,y_dist, dist=euclidean_distance([node_attributes[u]["x"],node_attributes[u]["y"]],[node_attributes[v]["x"],node_attributes[v]["y"]])
        if dist >0:
            f = -self.c * weight *np.log(1+dist)/dist
            node_attributes[u]["dx"] += x_dist * f
            node_attributes[u]["dy"] += y_dist * f

            node_attributes[v]["dx"] -= x_dist * f
            node_attributes[v]["dy"] -= y_dist * f


class LogAttractionDegreeDistributed(AttractionForce):
    def __init__(self, coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist, y_dist, dist = euclidean_distance([node_attributes[u]["x"], node_attributes[u]["y"]],
                                                  [node_attributes[v]["x"], node_attributes[v]["y"]])
        if dist > 0:
            f = -self.c * weight * np.log(1 + dist) / dist/ node_attributes[u]["mass"]
            node_attributes[u]["dx"] += x_dist * f
            node_attributes[u]["dy"] += y_dist * f

            node_attributes[v]["dx"] -= x_dist * f
            node_attributes[v]["dy"] -= y_dist * f


class LinAttractionAntiCollision(AttractionForce):
    def __init__(self, coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist, y_dist, dist = euclidean_distance([node_attributes[u]["x"], node_attributes[u]["y"]],
                                               [node_attributes[v]["x"], node_attributes[v]["y"]])
        dist = node_attributes[u]["size"] - node_attributes[v]["size"]
        if dist > 0:

            f = -self.c * weight
            node_attributes[u]["dx"] += x_dist * f
            node_attributes[u]["dy"] += y_dist * f

            node_attributes[v]["dx"] -= x_dist * f
            node_attributes[v]["dy"] -= y_dist * f


class LinAttractionAntiCollisionDegreeDistributed(AttractionForce):
    def __init__(self, coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist, y_dist, dist = euclidean_distance([node_attributes[u]["x"], node_attributes[u]["y"]],
                                                  [node_attributes[v]["x"], node_attributes[v]["y"]])
        dist = node_attributes[u]["size"] - node_attributes[v]["size"]
        if dist > 0:
            f = -self.c * weight / node_attributes[u]["mass"]
            node_attributes[u]["dx"] += x_dist * f
            node_attributes[u]["dy"] += y_dist * f

            node_attributes[v]["dx"] -= x_dist * f
            node_attributes[v]["dy"] -= y_dist * f


class LogAttractionAntiCollision(AttractionForce):
    def __init__(self, coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist, y_dist, dist = euclidean_distance([node_attributes[u]["x"], node_attributes[u]["y"]],
                                                  [node_attributes[v]["x"], node_attributes[v]["y"]])

        dist = node_attributes[u]["size"] - node_attributes[v]["size"]

        if dist > 0:
            f = -self.c * weight * np.log(1 + dist) / dist
            node_attributes[u]["dx"] += x_dist * f
            node_attributes[u]["dy"] += y_dist * f

            node_attributes[v]["dx"] -= x_dist * f
            node_attributes[v]["dy"] -= y_dist * f


class LogAttractionAntiCollisionDegreeDistributed(AttractionForce):
    def __init__(self, coefficient):
        AttractionForce.__init__(self)
        self.c = coefficient

    def apply(self, u, v, weight, node_attributes):
        x_dist, y_dist, dist = euclidean_distance([node_attributes[u]["x"], node_attributes[u]["y"]],
                                                  [node_attributes[v]["x"], node_attributes[v]["y"]])

        dist = node_attributes[u]["size"] - node_attributes[v]["size"]

        if dist > 0:
            f = -self.c * weight * np.log(1 + dist) / dist/ node_attributes[u]["mass"]
            node_attributes[u]["dx"] += x_dist * f
            node_attributes[u]["dy"] += y_dist * f

            node_attributes[v]["dx"] -= x_dist * f
            node_attributes[v]["dy"] -= y_dist * f

