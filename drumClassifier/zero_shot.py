import numpy as np
import torch
import laion_clap
import torchaudio
import torch.nn.functional as F

# requires 48.1 khz 
PATH = "/Users/rithuhegde/Desktop/toy_data/train/track_2/drums.wav"
wav, sr = torchaudio.load(PATH, normalize=True)
if sr != 48000:
    wav = torchaudio.functional.resample(wav, sr, 48000)

#chane stereo to mono
mono = wav.mean(dim=0)
batch = mono.unsqueeze(0)

audio_np = batch.numpy().astype(np.float32)
audio_t = batch.float()

# load model & pretrained checkpoint
model = laion_clap.CLAP_Module(enable_fusion=False)
checkpoint_path = "/Users/rithuhegde/checkpoints/laion-clap/630k-audioset-best.pt"
model.load_ckpt(checkpoint_path)  

# gets embeddings as a torch.tensor
audio_emb = model.get_audio_embedding_from_data(x=audio_np, use_tensor=False)
print(audio_emb.shape)  # shape (1, D) num samples, num features 

# text embeddings
texts = ["kick drum", "snare drum", "hi-hat"]
text_emb = model.get_text_embedding(x=texts, use_tensor=False)
print(text_emb.shape)   # shape (3, D)

text_emb_tensor = torch.from_numpy(text_emb)
audio_emb_tensor = torch.from_numpy(audio_emb)

cos = F.cosine_similarity(audio_emb_tensor, text_emb_tensor) # (3,)
pred_idx = cos.argmax().item()
print("predicted:", texts[pred_idx])
