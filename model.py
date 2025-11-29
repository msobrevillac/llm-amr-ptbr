import random
import jinja2
from pathlib import Path
from transformers import GenerationConfig
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

random.seed(1234)

JINJA_ENV = jinja2.Environment(
    loader=jinja2.BaseLoader,
    undefined=jinja2.StrictUndefined,
)


def render_prompt(template_path: str, sample):
    """Render the Jinja prompt using dataset metadata."""
    template_text = Path(template_path).read_text().strip()

    return JINJA_ENV.from_string(template_text).render(
        sample=sample
    )
    
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.2,
    top_p=0.75,
    num_beams=4,
    max_new_tokens=60
)

class Sabia_Generator:
    
    def __init__(self):

        self.tokenizer = LlamaTokenizer.from_pretrained("maritaca-ai/sabia-7b")
        self.model = LlamaForCausalLM.from_pretrained(
            "maritaca-ai/sabia-7b",
            device_map="auto",  # Automatically loads the model in the GPU, if there is one. Requires pip install acelerate
            #low_cpu_mem_usage=True,
            dtype=torch.bfloat16   # If your GPU does not support bfloat16, change to torch.float16
        )

    def evaluate(self, instruction):
        inputs = self.tokenizer(instruction, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=self.tokenizer.encode("\n")
        )
        for s in generation_output.sequences:
            output = self.tokenizer.decode(s)
            #print("Resposta:", output.split("Resposta:")[1].strip())
            return output.split("Resposta:")[1].strip()