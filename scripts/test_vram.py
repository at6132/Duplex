"""Quick VRAM test: load model, do one forward+backward pass."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from duplex.config import DuplexConfig
from duplex.duplex_model import DuplexModel

print('Loading model...', flush=True)
model = DuplexModel(DuplexConfig())
print(f'Model VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB', flush=True)

print('Single forward pass (seq_len=32)...', flush=True)
tok = model.tokenizer
enc = tok('Hello world test', return_tensors='pt', max_length=32, padding='max_length', truncation=True).to('cuda:0')
prompt_enc = tok('Test prompt', return_tensors='pt', max_length=32, padding='max_length', truncation=True).to('cuda:0')

out = model(
    input_ids=enc['input_ids'],
    attention_mask=enc['attention_mask'],
    prompt_ids=prompt_enc['input_ids'],
    prompt_mask=prompt_enc['attention_mask'],
    labels=enc['input_ids'],
)
loss = out["loss"]
print(f'Forward done. Loss: {loss.item():.4f}', flush=True)
print(f'VRAM after forward: {torch.cuda.memory_allocated()/1024**3:.2f}GB', flush=True)

print('Backward pass...', flush=True)
loss.backward()
print(f'Backward done. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB', flush=True)
print('SUCCESS!', flush=True)
