from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
from joblib import Parallel, delayed
import networkx as nx
from . import graph2vec
import pandas as pd

def extract_features(graph, rounds, name, features=None):
    if features is None:
        features = graph.degree()
        features = {int(k): v for k, v in features}
    else:
        features = {int(k): v for k, v in features.items()}
    
    machine = graph2vec.WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + str(name)])

    return doc

def EmbedGraphs(graphs, dimensions=128, workers=4, epochs=10, min_count=5, wl_iterations=2, learning_rate=0.025, down_sampling=0.0001, silent=False):

    if type(graphs) == nx.Graph:
        raise ValueError("Multiple graphs are required to create an embedding!")

    if silent:
        document_collections = Parallel(n_jobs=workers)(delayed(extract_features)(g, wl_iterations, i) for i, g in enumerate(graphs))
    else:
        document_collections = Parallel(n_jobs=workers)(delayed(extract_features)(g, wl_iterations, i) for i, g in tqdm(enumerate(graphs)))

    model = Doc2Vec(document_collections, 
                    vector_size=dimensions, 
                    window=0, 
                    min_count=min_count, 
                    dm=0, 
                    sample=down_sampling, 
                    workers=workers, 
                    epochs=epochs, 
                    alpha=learning_rate)
    
    out = []
    for i in range(len(graphs)):
        out.append([int(i)] + list(model.docvecs["g_"+str(i)]))
    column_names = ["graph"]+["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["graph"])

    return out