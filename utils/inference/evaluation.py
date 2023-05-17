"""

    evaluation
    ~~~~~~~~~~

    Provides an implementation of an evaluator to evaluate a given
    inference model object for specified metrics over a dataset of triples.

    Author: Kinshuk Vasisht
    Version: 1.0

"""

import os
import sys
import json
import random
import timeit
from collections.abc import Callable

import pandas
from tqdm.auto import tqdm

from ..common import format_duration, pathsafe, preprocess
from ..inference.model import Model

MASK_TO_CODE = {
    'head'    : 0,
    'relation': 1,
    'tail'    : 2
}

class Evaluator:
    """ Implements an evaluator for evaluating inference models.

        Evaluators are bound to specific datasets of triples, and can be used to
        evaluate models over dynamically specified metrics.

        Conversion of inputs and model outputs to formats expected by metrics,
        generation of triple masks and collection of results are
        handled by the evaluator itself."""

    PromptCallback = Callable[[Model, dict[str, str], str, int], list[str] | list[str]]

    def __init__(
        self, dataset_path,
        maskeable_fields=[ 'head', 'relation', 'tail' ],
        random_state=42,
        prompt_generation: PromptCallback=None
    ):
        """ Initializes a new Evaluator object.

        Args:
            dataset_path (str): Path to a CSV file to use for evaluation.
            maskeable_fields (list, optional): Valid fields which may be masked for predictions.
                Defaults to [ 'head', 'relation', 'tail' ].
            random_state (int, optional): Internal random state for determinism. Defaults to 42.
            prompt_generation (PromptCallback, optional): Custom function for prompt generation
                from raw inputs and masks. Defaults to None.
        """
        random.seed(random_state)

        self.name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.dataset = pandas.read_csv(dataset_path)
        self.field_masks = [
            mask for _ in range((len(self.dataset) // len(maskeable_fields)) + 5)
            for mask in maskeable_fields
        ]
        random.shuffle(self.field_masks)

        self.prompt_gen_fx = prompt_generation

    def make_prompt(self, model: Model, row: dict[str, str], mask: str):
        """ Generates a prompt from given row data according to the inference model.

        Args:
            model (Model): Inference model to be used for inference.
            row (dict[str, str]): Input data for the triple.
            mask (str): Field to be masked in the input.

        Returns:
            tuple[list, str]: List of prompts and the expected target.
        """
        model_name = model.name.lower()
        prompts, target = [], row[mask]

        if self.prompt_gen_fx:
            prompts = self.prompt_gen_fx(
                model, row, mask, MASK_TO_CODE[mask]
            ) or []

        if len(prompts) > 0:
            return prompts, target

        add_context = 'w-context' in model_name
        context = row['context'] if add_context else None

        if "bert" in model_name:
            triple = [ row['head'], row['relation'], row['tail'] ]
            if "qa" in model_name and add_context:
                question = ', '.join(entity for i, entity in enumerate(triple) if i != MASK_TO_CODE[mask])
                prompts.append({
                    'context': context,
                    'question': f"for {question}, identify {mask}?"
                })
            else:
                triple[0], triple[2] = preprocess(triple[0]), preprocess(triple[2])
                triple[1] = triple[1].replace('_', ' ')
                target    = triple[MASK_TO_CODE[mask]]
                num_masks = len(model.tokenizer.encode(
                    row[mask], return_tensors='pt', add_special_tokens=False
                ))
                triple[MASK_TO_CODE[mask]] = ' '.join('[MASK]' for _ in range(num_masks))
                prompts.append(' '.join(triple))
        else:
            if "mlm" in model_name or "kg" not in model_name:
                task_prefix = "fill-mask: " if add_context else ""
                triple = [ row['head'], row['relation'], row['tail'] ]
                triple[MASK_TO_CODE[mask]] = "<extra_id_0>"
                prompt = task_prefix + ' '.join(triple)
                if add_context: prompt += "\ncontext: " + context
                prompts.append(prompt)

            if "mlm" not in model_name:
                prompt = "predict " + mask + ": "
                if mask=='head':   prompt += row['tail'] + ' | ' + row['relation']
                elif mask=='tail': prompt += row['head'] + ' | ' + row['relation']
                else:              prompt += row['head'] + ' | ' + row['tail']
                if add_context: prompt += "\ncontext: " + context
                prompts.append(prompt)

        return prompts, target

    def make_inputs(self, model):
        """ Generates a list of inputs for a given inference model.
            Effectively maps dataset rows to prompt, target pairs
            constituting the inputs.

        Args:
            model (Model): Inference model to consider during prompt generation.

        Returns:
            list: List of inputs for inference.
        """
        inputs = []
        for (_, row), mask in zip(self.dataset.iterrows(), self.field_masks):
            prompts, target = self.make_prompt(model, row, mask)
            inputs.append({
                'triple': (row['head'], row['relation'], row['tail']),
                'prompts': prompts, 'target': target, 'mask': MASK_TO_CODE[mask]
            })
        return inputs

    def evaluate(self, model, metrics, num_samples=5, debug=True):
        """ Evaluates a given model over the bound dataset using specified metrics,
        returning the metric scores.

        Args:
            model (Model): Inference model to use for prediction.
            metrics (list): List of metric objects.
            num_samples (int, optional): Number of predictions per prompt. Defaults to 5.
            debug (bool, optional): Whether to dump debug data such as model predictions. Defaults to True.

        Returns:
            dict: Dictionary of metrics and metric values for model predictions.
        """
        inputs = self.make_inputs(model)
        data, outputs, metric_result = [], [], {}

        tic = timeit.default_timer()
        for input_row in tqdm(inputs, desc='Generating predictions'):
            model_outputs = [
                [
                    {
                        'prediction': prediction,
                        'log_score': score,
                        'triple': [
                            entity if i != input_row['mask'] else prediction
                            for i, entity in enumerate(input_row['triple'])
                        ]
                    }
                    for prediction, score in model.generate(prompt, num_samples=num_samples)
                ]
                for prompt in input_row['prompts']
            ]
            outputs.append(model_outputs)
            data.append({
                'inputs' : input_row,
                'outputs': model_outputs
            })
        toc = timeit.default_timer()
        metric_result['pred_time'] = f"{toc-tic:.6f}"
        metric_result['pred_time_fmtd'] = format_duration(round(toc-tic, 6))

        for metric, metric_fx in metrics.items():
            try:
                metric_result[metric] = metric_fx(inputs, outputs)
            except Exception as exc:
                print(exc, file=sys.stderr)
                metric_result[metric] = float('NaN')

        if debug:
            os.makedirs(os.path.join("predictions", self.name), exist_ok=True)
            with open(
                os.path.join("predictions", self.name, pathsafe(model.name)+".json"),
                "w+", encoding='utf-8'
            ) as ofile:
                json.dump(data, ofile, indent=2, ensure_ascii=False)

        return metric_result
