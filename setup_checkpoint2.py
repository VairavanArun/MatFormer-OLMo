#This file setups Matformer from checkpoint 2: Matformer-OLMo-460M
from olmo import Olmo, Tokenizer
import torch

checkpoint = "Matformer_OLMO_460M"
model = Olmo.from_checkpoint(checkpoint, device = "cuda")
tokenizer = Tokenizer.from_checkpoint(checkpoint)

input_ids = tokenizer.encode("I'm a large language model, ", add_special_tokens=False)
# `model.generate()` expects a batch.
input_tensor = torch.tensor(input_ids).unsqueeze(0)
input_tensor = input_tensor.cuda()

# Run beam search.
outputs = model.generate(input_tensor, max_steps=3, beam_size=3)

# The output token IDs are shape (batch_size, beam_size, max_steps)
best_generation = outputs.token_ids[0][0].tolist()
print(tokenizer.decode(best_generation))