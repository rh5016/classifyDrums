import os, glob
import numpy    as np
import torch
import torchaudio
import laion_clap


model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt("/Users/rithuhegde/checkpoints/laion-clap/630k-audioset-best.pt")
sr_target = 48000

def load_and_prepare(path):
    wav, sr = torchaudio.load(path, normalize=True)
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, sr, sr_target)
    mono  = wav.mean(dim=0, keepdim=True)       # (1,T)
    return mono.numpy().astype(np.float32)      # shape (1,T)

# Goes through dataset
labels = ["kick","snare","hihat"]
all_embs, all_lbls = [], []
for idx, lab in enumerate(labels):
    for fp in glob.glob(f"data/IDMT-SMT-Drums/{lab}/*.wav"):
        audio_np = load_and_prepare(fp)
        emb = model.get_audio_embedding_from_data(x=audio_np, use_tensor=False)  # (1,D)
        all_embs.append(emb[0])
        all_lbls.append(idx)


np.save("embs.npy", np.stack(all_embs))  # (N, D)
np.save("lbls.npy", np.array(all_lbls))
