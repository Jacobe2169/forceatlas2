from .force_atlas2 import ForceAtlas2
from tqdm import tqdm

def forceatlas2(graph,nb_iter=100,verbose = False,**parameters):
    fa = ForceAtlas2(graph,**parameters)
    for iter in tqdm(range(nb_iter),disable=(not verbose)):
        fa.iteration()
    return fa.get_positions()
