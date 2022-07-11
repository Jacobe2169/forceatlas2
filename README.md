# ForceAtlas2 for Python

This package is an implementation of the ForceAtlas2 algorithm available in Gephi. Since, this is an early work, some issues 
concerning performances have to be attended to ! 


## Setup

    sudo pip install git+https://github.com/Jacobe2169/forceatlas2py

or 

    git clone https://github.com/Jacobe2169/forceatlas2py
    cd forceatlas2py
    python setup.py install

## Usage 

```python
import networkx as nx
from force_atlas_layout import forceatlas2
import matplotlib.pyplot as plt

G = nx.les_miserables_graph()
pos = forceatlas2(G)
print(pos)

nx.draw(G,pos=pos,node_size=[G.degree(n) for n in G])
plt.show()
```