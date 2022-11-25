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


def forceatlas2widget(G,nodes_color=[]):
    try:
        import ipywidgets as widgets
        from ipywidgets import interact, interact_manual, fixed
        import networkx as nx
    except:
        raise ImportError("ipywidgets is required !")

    pos = {}
    def plot(pos,widgets_gui): 
        pos = forceatlas2(G,nb_iter=widgets_gui["nb_iter"].value,speed=widgets_gui["speed"].value,prevent_overlap=widgets_gui["prevent_overlap"].value,verbose=widgets_gui["verbose"].value,positions=pos,gravity=widgets_gui["gravity"].value)
        return nx.draw_networkx_nodes(G,pos,node_color=nodes_color,node_size=[G.degree(n) for n in G])
        
    def reset(b):
        global pos
        pos= {}
        
    widgets_gui = dict(
    nb_iter=widgets.IntText(min=0, max=1000, step=1, value=10,description="Nb. Iteration"),
    speed=widgets.IntText(min=0, max=100, step=1, value=1,description="Speed"),
    gravity=widgets.IntText(min=0, max=100, step=1, value=1,description="Gravity"),
    prevent_overlap = widgets.ToggleButton(description="Prevent Overlap"),
    verbose = widgets.ToggleButton(description="Verbose"),
    )
    reset_button = widgets.Button(description="Reset ↺")
    run_button = interact_manual(plot,pos=fixed(pos),widgets_gui=fixed(widgets_gui))
    run_button.widget.children[0].description = 'Run ▶️'

    param = widgets.HBox(list(widgets_gui.values()))
    command = widgets.HBox([reset_button])
    all_ = widgets.VBox([param,command])

    reset_button.on_click(reset)

    return all_