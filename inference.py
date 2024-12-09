from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# from transformers import BitsAndBytesConfig

# config = BitsAndBytesConfig(
#     load_in_4bit=True
# )

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.bfloat16, # loads model in 16bit
    device_map="auto", # checks if cpu or gpu is avaiable to load model on
    trust_remote_code=True
)

# print(model.eval())

input_texts = tokenizer(text="What is cricket?", return_tensors="pt") # return_tensor='pt' means return tensor in torch format
print(input_texts) # {'input_ids': tensor([[    1,  2592,  1117, 26791, 29572]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]])}
# output = model.generate(**input_texts) # Tensor([[1,2,3,4,....,10]])

# print(output)
# print(tokenizer.decode(token_ids=output[0], skip_special_tokens=True))

# torch.no_grad is used when we dont need to calculate weights and save them like in training, 
# so we can have faster inference as it is not necessary there
with torch.no_grad():
    output = model.generate(**input_texts)
    print(output) # Tensor([[1,2,3,4,....,10]])
    print(tokenizer.decode(token_ids=output[0], skip_special_tokens=True).split("\n"))


for i in range(5):
    print(i)
    
for i in range(10):
    print(i)





















"""
`torch.no_grad()` is a context manager in PyTorch that temporarily sets all the `requires_grad` flags to `False`. This means that during the execution of operations within this context, PyTorch will not compute gradients for any of the tensors involved.

### What is `torch.no_grad()`?

- **Purpose:** To disable gradient calculation, which is used mainly during inference when you don't need gradients to be computed or stored.
- **Use Case:** During inference or evaluation, when you are running a model to make predictions and not updating model parameters (weights), you don't need gradients. Computing gradients during inference consumes additional memory and computational resources without any benefit.

### Key Benefits of Using `torch.no_grad()`:

1. **Memory Efficiency:** Reduces memory consumption since no intermediate results are stored for gradient computation.
2. **Speed:** Speeds up computations by skipping the backpropagation step and avoiding the creation of gradient tensors.
3. **Safety:** Helps avoid accidental modifications of model parameters during inference.

### Example Usage

Here's an example that demonstrates the use of `torch.no_grad()`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize input text
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors='pt')

# Model evaluation mode (disable dropout and other training-specific layers)
model.eval()

# Inference with torch.no_grad() to save memory and computation
with torch.no_grad():
    outputs = model(**inputs)

# Convert the output to text
print(tokenizer.decode(outputs.logits.argmax(-1)[0], skip_special_tokens=True))
```

### Explanation:

1. **Model Evaluation Mode (`model.eval()`):** This method is called to set the model to evaluation mode. It disables layers like dropout and batch normalization that behave differently during training.
   
2. **`torch.no_grad()` Context Manager:** The block of code inside `with torch.no_grad()` will not track operations for gradient computation. This is ideal for inference since gradients are not needed.

3. **Inside the Block:** During inference, the model forward pass is done without tracking gradients, leading to reduced memory usage and faster computation.

### Common Scenarios for `torch.no_grad()`:

- **Inference:** When generating predictions from a model, as shown in the example.
- **Validation:** When evaluating a model on a validation dataset after each epoch during training to monitor performance.
- **Memory-Constrained Environments:** When running models on environments with limited memory (e.g., CPUs or edge devices), avoiding unnecessary gradient calculations helps save resources.

### Summary

- `torch.no_grad()` is used to disable gradient tracking, which is beneficial for inference and evaluation.
- It improves memory efficiency and computation speed by preventing unnecessary gradient computations.

Would you like more information on how gradients work in PyTorch or any related topics?

"""