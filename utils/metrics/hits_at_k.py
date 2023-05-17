"""

    hits_at_k
    ~~~~~~~~~

    Implements the Hits@k / MeanPrecision@K metric for
    triple prediction result evaluation.

    Author: Kinshuk Vasisht
    Version: 1.0

"""

from ..common import preprocess

def evaluate_strict(target, output):
    """ Strict evaluation strategy: expects target to be equal to the prediction for a hit. """
    target = target.lower()
    return any(target == sample['prediction'].lower() for sample in output)

def evaluate_lax(target, output):
    """ Lax evaluation strategy: expects target to be contained in the predictions for a hit. """
    target = preprocess(target.lower())
    return any(target in preprocess(sample['prediction'].lower()) for sample in output)

class HitsAtK:
    """ Implementation of the Hits@K / MeanPrecision@K metric. """

    EVALUATION_STRATEGIES = {
        'strict': evaluate_strict,
        'lax'   : evaluate_lax
    }

    def __init__(self, K, evaluation_strategy='strict'):
        """ Initializes a new HitsAtK metric object.

        Args:
            K (int): Maximum output rank to consider for hits.
            evaluation_strategy (str, optional): Evaluation strategy to use. Defaults to 'strict'.

        Raises:
            ValueError: Invalid evaluation strategy specified.
        """
        self.max_rank = K
        self.evaluate_fx = self.EVALUATION_STRATEGIES.get(evaluation_strategy)

        if self.evaluate_fx is None:
            raise ValueError("HitsAtK: invalid evaluation strategy: " + evaluation_strategy)

    def __call__(self, inputs, outputs):
        """ Computes the value of the metric, given expected input values
            and prediction outputs.

            Args:
                inputs (list[dict]): Expected input values containing target and gold standard triples.
                outputs (list[list[dict]]): Model predictions for every input prompt.

            Returns:
                float: Metric value.
            """
        count_at_k, max_points = 0, len(outputs)
        for input_data, output_data in zip(inputs, outputs):
            could_predict_target = False
            for output in output_data:
                output = output[:self.max_rank]
                if self.evaluate_fx(input_data['target'], output):
                    could_predict_target = True
                    break
            if could_predict_target:
                count_at_k += 1
        return count_at_k/max_points
