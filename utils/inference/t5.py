"""

    t5
    ~~

    Provides the methods for inference via T5 models.
    Based on the inference code for KGT-5 by Apoorv Saxena
    (https://github.com/apoorvumang/kgt5)

    Author: Kinshuk Vasisht
    Version: 1.0

"""

import torch

def get_output_scores(ids, scores, pad_token_id):
    """get sequence scores from model.generate output"""
    scores = torch.stack(scores, dim=1)
    log_probs = torch.log_softmax(scores, dim=2)
    ids = ids[:,1:] # remove start token
    x = ids.unsqueeze(-1).expand(log_probs.shape) # gather needed probs
    needed_logits = torch.gather(log_probs, 2, x)
    final_logits = needed_logits[:, :, 0]
    padded_mask = (ids == pad_token_id)
    final_logits[padded_mask] = 0
    final_scores = final_logits.sum(dim=-1)
    return final_scores.cpu().detach().numpy()

def sample_top_k_outputs(
    input_prompt, model, tokenizer, num_samples=5,
    num_beams=1, max_output_length=30
):
    """ Sample top K predictions from a T5 based model,
        and return a list of prediction and log score tuples. """

    tokenized = tokenizer(input_prompt, return_tensors="pt")
    if torch.cuda.is_available():
      tokenized = tokenized.to('cuda:0')
    out = model.generate(
        **tokenized, do_sample=True,
        num_return_sequences = num_samples,
        num_beams=num_beams,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        output_scores=True,
        return_dict_in_generate=True,
        max_length=max_output_length
    )

    out_tokens = out.sequences
    out_str = tokenizer.batch_decode(out_tokens, skip_special_tokens=True)
    out_scores = get_output_scores(out_tokens, out.scores, tokenizer.pad_token_id)

    pair_list = [ (x[0], float(x[1])) for x in zip(out_str, out_scores) ]
    return sorted(pair_list, key=lambda x: x[1], reverse=True)
