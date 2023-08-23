import torch  
from torchtext.vocab import build_vocab_from_iterator  
  
# Define the code snippet  
code_snippet = """  
 
"""  
  
# Tokenize the code snippet using whitespace as the delimiter  
tokens = code_snippet.split()  
  
# Build a vocabulary from the tokens  
vocab = build_vocab_from_iterator([tokens])  
  
# Convert tokens to numerical values using the vocabulary  
token_indices = [vocab[token] for token in tokens]  
  
# Convert token indices to PyTorch tensor  
token_tensors = torch.tensor(token_indices)  
  
# Print the tokens and their corresponding tensors  
print("Tokens:", tokens)  
print("Token Tensors:", token_tensors)  
