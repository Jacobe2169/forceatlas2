from .force_atlas2 import ForceAtlas2
from tqdm import tqdm

def forceatlas2(graph,nb_iter=100,verbose = False,**parameters):
    if not "barnes_hut_optimize" in parameters:
        parameters["barnes_hut_optimize"] = True if len(graph) >=500 else False
    if not "scaling_ratio" in parameters:
        parameters["scaling_ratio"] = 2 if len(graph) <=100 else 10

    fa = ForceAtlas2(graph,**parameters)
    for _ in tqdm(range(nb_iter),disable=(not verbose)):
        fa.iteration()
    return fa.get_positions()
