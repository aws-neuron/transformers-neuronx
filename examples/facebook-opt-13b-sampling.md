## Run HuggingFace `facebook/opt-13b` autoregressive sampling on trn1.2xlarge

The first step is to download and construct the `facebook/opt-13b` model using the HuggingFace
`from_pretrained/save_pretrained` methods, and save the model as a local folder `./opt-13b-local`.

```
# download.py
from transformers.models.opt import OPTForCausalLM

hf_model = OPTForCausalLM.from_pretrained('facebook/opt-13b', low_cpu_mem_usage=True)
hf_model.save_pretrained('./opt-13b-local')
```

For the sake of reducing host memory usage, it is recommended to save the model `state_dict` as
multiple files, as opposed to one monolithic file given by `torch.save`. This "split-format"
`state_dict` can be created using the `save_pretrained_split` function. With this checkpoint format,
the Neuron model loader can "load parameters to Neuron device high-bandwidth memory (HBM) directly"
by keeping at most one layer of model parameters in the CPU main memory.

To reduce the memory usage, we cast the attention and mlp layers to `float16` precision before
saving them. We keep the layernorms in `float32`. To do this, we implement a callback function
that casts each layer in the model.

```
# split_checkpoint.py
import torch
from transformers.models.opt import OPTForCausalLM
from transformers_neuronx.module import save_pretrained_split

def amp_callback(model, dtype):
    # cast attention and mlp to low precision only; layernorms stay as f32
    for block in model.model.decoder.layers:
        block.self_attn.to(dtype)
        block.fc1.to(dtype)
        block.fc2.to(dtype)
    model.lm_head.to(dtype)

hf_model = OPTForCausalLM.from_pretrained('./opt-13b-local', low_cpu_mem_usage=True)
amp_callback(hf_model, torch.float16)
save_pretrained_split(hf_model, './opt-13b-split')
```

Now we have all necessary files for running `facebook/opt-13b` autoregressive sampling. We will need
the Neuron `OPTForSampling` class for this purpose. The default model config supports sampling up to
sequence length 2048, and we set batch size to 2. Tensor-parallelism is enabled through the argument
`tp_degree=2`. Under the hood, the Neuron tensor manipulator can shard/duplicate tensors to multiple
NeuronCores (2 in this case) to enable tensor-parallel computations on multiple NeuronCores.

```
# demo.py
import time
import torch
from transformers import AutoTokenizer
from transformers_neuronx.opt.model import OPTForSampling

# load facebook/opt-13b to NeuronCores with 2-way tensor parallel
# enable float16 casting
neuron_model = OPTForSampling.from_pretrained('./opt-13b-split', batch_size=2, tp_degree=2, amp='f16')
neuron_model.to_neuron()

# construct a tokenizer and encode prompt text
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-13b')
batch_prompts = [
    "Hello, I'm a language model,",
    "Welcome to Amazon Elastic Compute Cloud,",
]
input_ids = torch.as_tensor([tokenizer.encode(text) for text in batch_prompts])

with torch.inference_mode():
    start = time.time()
    generated_sequences = neuron_model.sample(input_ids, sequence_length=2048)
    elapsed = time.time() - start

generated_sequences = [tokenizer.decode(seq) for seq in generated_sequences]
print(f'generated sequences {generated_sequences} in {elapsed} seconds')
```

Larger batch sizes won't fit into a trn1.2xlarge instance. The instance has 32 GB of HBM, and
`facebook/opt-13b` has ~26 GB of model parameters. With batch size 3, after storing model parameters
and key-value caches, there will be less than 1 GB of HBM left, and that is not enough for storing
code and temporary data generated during the sampling computation.
