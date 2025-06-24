"""
Interactive Q&A with trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT


# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# model
if init_from == 'resume':
   # init from a model saved in a specific directory
   ckpt_path = os.path.join(out_dir, 'ckpt.pt')
   checkpoint = torch.load(ckpt_path, map_location=device)
   gptconf = GPTConfig(**checkpoint['model_args'])
   model = GPT(gptconf)
   state_dict = checkpoint['model']
   unwanted_prefix = '_orig_mod.'
   for k,v in list(state_dict.items()):
       if k.startswith(unwanted_prefix):
           state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
   model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
   # init from a given GPT-2 model
   model = GPT.from_pretrained(init_from, dict(dropout=0.0))


model.eval()
model.to(device)
if compile:
   model = torch.compile(model) # requires PyTorch 2.0 (optional)


# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
   meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
   load_meta = os.path.exists(meta_path)
if load_meta:
   print(f"Loading meta from {meta_path}...")
   with open(meta_path, 'rb') as f:
       meta = pickle.load(f)
   # TODO want to make this more general to arbitrary encoder/decoder schemes
   stoi, itos = meta['stoi'], meta['itos']
   encode = lambda s: [stoi[c] for c in s]
   decode = lambda l: ''.join([itos[i] for i in l])
else:
   # ok let's assume gpt-2 encodings by default
   print("No meta.pkl found, assuming GPT-2 encodings...")
   enc = tiktoken.get_encoding("gpt2")
   encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
   decode = lambda l: enc.decode(l)


def generate_response(prompt, max_new_tokens=500):
   """Generate a response to the given prompt"""
   # Format the prompt to match training data format
   formatted_prompt = f"{prompt}" + "{}"
  
   # encode the prompt
   start_ids = encode(formatted_prompt)
   x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
  
   # generate response
   with torch.no_grad():
       with ctx:
           y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
           response = decode(y[0].tolist())
  
   return response


def extract_response(full_text, prompt):
   """Extract just the response part, removing the prompt"""
   # Remove the formatted prompt (question + ?)
   formatted_prompt = f"{prompt}" + "{}"
   if full_text.startswith(formatted_prompt):
       response = full_text[len(formatted_prompt):]
   else:
       response = full_text
  
   # Remove the [] end block if present
   if '[]' in response:
       response = response.split('[]')[0]
  
   return response.strip()


def main():
   print("Interactive Q&A with trained model")
   print("Type 'quit' to exit")
   print("-" * 50)
  
   while True:
       # Get user question
       question = input("\nYour question: ").strip()
      
       if question.lower() in ['quit', 'exit', 'q']:
           print("Goodbye!")
           break
      
       if not question:
           continue
      
       print("\nGenerating response...")
      
    #    try:
       # Generate full response
       full_response = generate_response(question)
        
       # Extract just the response part
       response = extract_response(full_response, question)
        
       print(f"\nResponse: {response}")
          
    #    except Exception as e:
    #        print(f"Error generating response: {e}")
    #        continue


if __name__ == "__main__":
   main()