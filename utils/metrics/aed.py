"""

    aed
    ~~~

    Provides a wrapper over the implementation of the Approximated Edit Distance
    metric from the ged module by Pau Riba and Anjan Dutta.

    Author: Kinshuk Vasisht
    Version: 1.0

"""

import pandas
import networkx

from ged.VanillaAED import VanillaAED

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def create_matching_dict(nodes):
    """
    Given a set of nodes, it stems the node strings and creates a dict with the stemmed strings as keys
    and an index as the values.
    :param nodes:   networkx.NodeView
    :return:        dict
    """
    keys = set()
    for token in nodes:
        stemmed = stemmer.stem(token)
        keys.add(stemmed)

    return {y: x for x, y in enumerate(keys)}

def from_str_to_ids(edges: list, matching_dict: dict):
    """
    Creates a network with int node names instead of strings.
    :param edges:           networkx.EdgeView
    :param matching_dict:   dict, with <stemmed string> -> <id>
    :return:                networkx.Graph
    """

    def get_similar_id(node_str):
        """
        It matches the stemmed node string with respective id
        """
        stemmed = stemmer.stem(node_str)
        for stem in matching_dict.keys():
            if stemmed == stem:
                return matching_dict[stemmed]

        raise ValueError('Node id not found')

    stemmed_pairs = list()
    for pair in edges:
        source_id = get_similar_id(pair[0])
        target_id = get_similar_id(pair[1])

        stemmed_pairs.append((source_id, target_id))

    kg_df = pandas.DataFrame(stemmed_pairs, columns=["source", "target"])
    kg = networkx.from_pandas_edgelist(kg_df, "source", "target", create_using=networkx.Graph())

    return kg

def triples_to_kg(flat_triples):
    """ Converts a list of triples to a knowledge graph represented as a networkx graph. """
    df = pandas.DataFrame.from_records(
        flat_triples, columns=[ 'head', 'relation', 'tail' ]
    )
    graph = networkx.from_pandas_edgelist(
        df, source='head', target='tail', edge_attr=True,
        create_using=networkx.DiGraph()
    )
    return graph

def to_kg(triples):
    """ From a list of triples, generates a knowledge graph
        and a knowledge graph with stemmed entity nodes mapped to integer IDs,
        and returns the generated graphs
    """
    kg    = triples_to_kg(triples)
    id_kg = from_str_to_ids(kg.edges(), create_matching_dict(kg.nodes))
    return kg, id_kg

class ApproximatedEditDistance:
    """ Approximation of the Graph Edit Distance metric,
        implemented as a wrapping over the implementation
        of AED from the ged module. """

    def __init__(self, threshold=None):
        """ Initializes a new AED metric object.

        Args:
            threshold (float, optional): Log score threshold to filter
                low confidence triples. Defaults to None.
        """
        self.threshold = threshold
        self.aed = VanillaAED()

    def __call__(self, inputs, outputs):
        """ Computes the value of the metric, given expected input values
            and prediction outputs.

            Args:
                inputs (list[dict]): Expected input values containing target and gold standard triples.
                outputs (list[list[dict]]): Model predictions for every input prompt.

            Returns:
                float: Metric value.
            """
        base_triples = []
        predicted_triples = []

        for input_data, output_data in zip(inputs, outputs):
            base_triples.append(input_data['triple'])
            best_prediction = max(
                (prompt_outputs[0] for prompt_outputs in output_data),
                key=lambda x: x['log_score']
            )
            if self.threshold is None or best_prediction['log_score'] >= self.threshold:
                predicted_triples.append(best_prediction['triple'])

        _, base_kg      = to_kg(base_triples)
        _, predicted_kg = to_kg(predicted_triples)

        return self.aed.ged(base_kg, predicted_kg)[0]
