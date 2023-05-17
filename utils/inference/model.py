"""

    model
    ~~~~~

    Provides an implementation of a Model wrapper class which allows
    for seamless inference from huggingface transformers models.

"""

import peft
import torch
import transformers

from . import t5, bert

class Model:
    """ Implementation of a wrapper class over huggingface transformers models.
        This class allows defining objects which correctly initialize transformer
        models as per properties deduced from the model names and correctly
        sets up the tokenizer and device for inference. """

    def __init__(self, name):
        """ Creates a new inference model object.
            Sets up the model, tokenizer and places the model onto the
            best available device (GPU or CPU).
            Supports PEFT-optimized models.

        Args:
            name (str): Name of the model (including ORG prefix)
        """
        self.name = name

        if "peft" in name:
            self.config = peft.PeftConfig.from_pretrained(name)
            name = self.config.base_model_name_or_path

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)

        if "bert" in name.lower():
            if "qa" in name.lower():
                self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(name)
            else:
                self.model = transformers.AutoModelForMaskedLM.from_pretrained(name)
        else:
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(name)

        if "peft" in self.name:
            self.model = peft.PeftModel.from_pretrained(self.model, self.name)

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def generate(self, input_prompt, num_samples=5):
        """ Generates inferences from the model using the given prompt,
            as per the model's characteristics.

        Args:
            input_prompt (any): Input prompt to use for generation.
            num_samples (int, optional): Number of required predictions. Defaults to 5.

        Returns:
            list: List of tuple of prediction and scores.
        """
        if "bert" in self.name.lower():
            if "qa" in self.name.lower():
                return bert.sample_top_k_answers_for_qa(
                    input_prompt, self.model, self.tokenizer, num_samples
                )
            else:
                return bert.sample_top_k_outputs(
                    input_prompt, self.model, self.tokenizer, num_samples
                )
        else:
            return t5.sample_top_k_outputs(
                input_prompt, self.model, self.tokenizer, num_samples
            )
