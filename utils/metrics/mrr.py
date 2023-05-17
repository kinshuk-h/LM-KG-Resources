"""

    mrr
    ~~~

    Provides an implementation of the Mean Reciprocal Rank metric for
    triple prediction result evaluation.

    Author: Kinshuk Vasisht
    Version: 1.0

"""

from ..common import preprocess

def evaluate_strict(target, output):
    """ Strict evaluation strategy: expects target to be equal to the prediction for a hit. """
    target = target.lower()
    for i, sample in enumerate(output, 1):
        if target == sample['prediction'].lower(): return i
    return None

def evaluate_lax(target, output):
    """ Lax evaluation strategy: expects target to be contained in the predictions for a hit. """
    target = preprocess(target.lower())
    for i, sample in enumerate(output, 1):
        if target in preprocess(sample['prediction'].lower()): return i
    return None

class MeanReciprocalRank:
    """ Implements the Mean Reciprocal Rank metric for
        evaluation of triple prediction results. """

    EVALUATION_STRATEGIES = {
        'strict': evaluate_strict,
        'lax'   : evaluate_lax
    }

    def __init__(self, evaluation_strategy='strict'):
        """ Initailizes a new MRR metric object.

        Args:
            evaluation_strategy (str, optional): Evaluation strategy to use. Defaults to 'strict'.

        Raises:
            ValueError: Invalid evaluation strategy specified.
        """
        self.evaluate_fx = self.EVALUATION_STRATEGIES.get(evaluation_strategy)

        if self.evaluate_fx is None:
            raise ValueError("MeanReciprocalRank: invalid evaluation strategy: " + evaluation_strategy)

    def __call__(self, inputs, outputs):
        """ Computes the value of the metric, given expected input values
            and prediction outputs.

            Args:
                inputs (list[dict]): Expected input values containing target and gold standard triples.
                outputs (list[list[dict]]): Model predictions for every input prompt.

            Returns:
                float: Metric value.
            """
        rank_sum, max_points = 0, len(outputs)
        for input_data, output_data in zip(inputs, outputs):
            for output in output_data:
                if rank := self.evaluate_fx(input_data['target'], output):
                    rank_sum += (1 / rank)
                    break
        return rank_sum/max_points
