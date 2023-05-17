"""

    bert
    ~~~~

    Implementation of inference methods for predictions using BERT-based models.
    Supports predictions for MLM based models (with multiple mask prediction) and
    downstream task predictions, such as QA.

    Based on the huggingface NLP pipeline implementations and the methodology
    described in the huggingface NLP course, chapters 6 and 7.

    Author: Kinshuk Vasisht
    Version: 1.0

"""

import copy
import heapq
import random

import numpy
import torch

def get_predictions(tokenizer, masked_index, input_ids, logits, top_k):
    """ Generate top-k predictions for a single mask.
        Adapted from huggingface.transformers.pipeline.fill_mask.FillMaskPipeline.preprocess
        (https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/fill_mask.py#L105)
    """
    probs  = logits.softmax(dim=-1)
    values, predictions = probs.topk(top_k)
    row = []
    for v, p in zip(values.tolist(), predictions.tolist()):
        tokens = numpy.array(input_ids)
        tokens[masked_index] = p
        # Filter padding out:
        tokens = tokens[numpy.where(tokens != tokenizer.pad_token_id)]
        # Originally we skip special tokens to give readable output.
        # For multi masks though, the other [MASK] would be removed otherwise
        # making the output look odd, so we add them back
        sequence = tokenizer.decode(tokens, skip_special_tokens=True)
        proposition = {
            "score": v, "token": p,
            # "token_str": tokenizer.decode([p]),
            "sequence": sequence,
            "sequence_tokens": tokens.tolist()
        }
        row.append(proposition)
    return sorted(row, reverse=True, key=lambda x: x['score'])

def combine_predictions(main_proposition, beam_propositions, top_k, net_proposition):
    """ Combines root prediction with beam predictions and computes overall score for
        combined predictions.
    """
    lprop = main_proposition
    for rprop in beam_propositions:
        if 'scores' in lprop:
            new_prop = copy.deepcopy(lprop)
            new_prop.update(rprop)
            new_prop['scores'].append(rprop['score'])
            new_prop['score'] = float(numpy.prod(new_prop['scores']))
            new_prop['tokens'].append(rprop['token'])
            # new_prop['token_strs'].append(rprop['token_str'])
        else:
            new_prop = rprop
            new_prop.update(
                score=lprop['score']*rprop['score'],
                scores=[ lprop['score'], rprop['score'] ],
                tokens=[ lprop['token'], rprop['token'] ],
                # token_strs=[ lprop['token_str'], rprop['token_str'] ],
            )
        if len(net_proposition) >= top_k:
            if new_prop['score'] > net_proposition[0][0]:
                heapq.heapreplace(net_proposition, (new_prop['score'] * (1e-9 * random.random()), new_prop))
        else:
            heapq.heappush(net_proposition, (new_prop['score'] * (1e-9 * random.random()), new_prop))

def beam_predict(sequence, model, tokenizer, top_k=5, order="right-to-left"):
    """ Performs beam search prediction for multiple consecutive masks. """
    ids_main = tokenizer.encode(sequence, return_tensors="pt")

    position = torch.where(ids_main == tokenizer.mask_token_id)
    positions_list = position[1].numpy().tolist()

    if torch.cuda.is_available():
        ids_main = ids_main.to('cuda:0')

    if order == "left-to-right":
        positions_list.reverse()

    elif order == "random":
        random.shuffle(positions_list)

    predictions = []

    for i in range(len(positions_list)):
        # if it was the first prediction,
        # just go on and predict the first predictions
        if i == 0:
            model_logits = model(ids_main)["logits"][0][positions_list[0]]
            predictions.append(
                get_predictions(
                    tokenizer, positions_list[0],
                    ids_main[0].detach().tolist(),
                    model_logits, top_k
                )
            )

        # if we already have some predictions, go on and fill the rest of the masks
        # by continuing the previous predictions
        if i != 0:
            seq_ids = torch.tensor([ prop['sequence_tokens'] for prop in predictions[-1]])
            if torch.cuda.is_available():
                seq_ids = seq_ids.to('cuda:0')
            model_logits, net_predictions = model(seq_ids)["logits"], []

            for k, prediction in enumerate(predictions[-1]):
                logits = model_logits[k][positions_list[i]]
                # get the top k of this prediction and masked token
                new_predictions = get_predictions(
                    tokenizer, positions_list[i],
                    prediction['sequence_tokens'], logits, top_k
                )
                combine_predictions(
                    prediction, new_predictions, top_k, net_predictions
                )
            final_predictions = []
            while len(net_predictions) > 0:
                _, prop = heapq.heappop(net_predictions)
                final_predictions.append(prop)
            predictions.append(final_predictions[::-1])

    return predictions

def sample_top_k_outputs(input_prompt, model, tokenizer, num_samples):
    """ Samples top K outputs for mask predictions for a BERT-based model for MaskedLM.
        Utilizes beam search for predicting outputs for consecutive masks.

    Args:
        input_prompt (str): Input prompt with masks.
        model (AutoModelForMaskedLM): Model for mask prediction
        tokenizer (FastTokenizer): Fast tokenizer associated with the model (Rust-based)
        num_samples (int): Number of predictions to generate.

    Returns:
        list: List of predictions and score tuples.
    """
    results = beam_predict(input_prompt, model, tokenizer, top_k=num_samples)[-1]
    out_strs = tokenizer.batch_decode([
        prop.get('tokens', prop['token']) for prop in results
    ], skip_output_tokens=True)
    return list(zip(out_strs, (numpy.log(prop['score']) for prop in results)))

def sample_top_k_answers_for_qa(input_prompt, model, tokenizer, num_samples):
    """ Samples top K answer predictions from a BertForQuestionAnswering style model.

    Args:
        input_prompt (dict): Dictionary of question and context.
        model (AutoModelForQuestionAnswering): Extractive question answering model.
        tokenizer (FastTokenizer): Fast tokenizer (Rust-based implementation).
        num_samples (int): Number of predictions to generate.

    Returns:
        list: List of predictions and log score tuples.
    """
    question, context = input_prompt['question'], input_prompt['context']
    inputs = tokenizer(
        question, context,
        stride=50, padding="max_length",
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    _ = inputs.pop("overflow_to_sample_mapping")
    offsets = inputs.pop("offset_mapping")

    inputs = inputs.convert_to_tensors("pt").to(model.device)
    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits   = outputs.end_logits

    unwanted_tokens = [ i != 1 for i in inputs.sequence_ids() ]
    unwanted_tokens[0] = False # unmask [CLS]
    unwanted_tokens = torch.logical_or(
        torch.tensor(unwanted_tokens).to(model.device)[None],
        (inputs['attention_mask']==0)
    )

    start_logits[unwanted_tokens] = -1_00_00
    end_logits[unwanted_tokens] = -1_00_00

    start_probs = torch.nn.functional.softmax(start_logits, dim=-1)
    end_probs   = torch.nn.functional.softmax(end_logits, dim=-1)

    top_samples = []
    for start_prob, end_prob, offset in zip(start_probs, end_probs, offsets):
        # Score for prediction: p(start) * p(end) (independent predictions)
        scores = start_prob[:, None] * end_prob[None, :]
        # Select top K scores which satisfy start_idx <= end_idx
        candidate_idx = torch.topk(torch.triu(scores).flatten(), num_samples).indices.detach().tolist()
        for idx in candidate_idx:
            start_idx = idx // scores.shape[1]
            end_idx   = idx % scores.shape[1]

            start_pos, _ = offset[start_idx]
            _, end_pos   = offset[end_idx]

            answer = context[start_pos:end_pos]
            sample = ( answer, float(torch.log(scores[start_idx, end_idx]).item()) )

            if len(top_samples) >= num_samples:
                if sample[1] > top_samples[0][0]:
                    heapq.heapreplace(top_samples, (sample[1] + (1e-9 * random.random()), sample))
            else:
                heapq.heappush(top_samples, (sample[1] + (1e-9 * random.random()), sample))

    # Select top K candidates from heap filled with K^2 candidates
    best_samples = list(heapq.heappop(top_samples)[1] for _ in range(num_samples))
    best_samples.reverse()
    return best_samples