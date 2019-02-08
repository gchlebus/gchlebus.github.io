import numpy as np

def read_file(p):
  loss = []
  g = []
  with open(p, "r") as f:
    for l in f.readlines():
      loss.append(float(l.strip().split(",")[0]))
      g.append(float(l.strip().split(",")[1]))
  return loss, g

files = [
  "dice_fgpatches.csv",
  "dice_bgpatches.csv",
  "diceallchannels_fgpatches.csv",
  "diceallchannels_bgpatches.csv",
  "cce_fgpatches.csv",
  "cce_bgpatches.csv",
  "topk25_bgpatches.csv",
  "topk25_fgpatches.csv"
]

for f in files:
  l, g = read_file(f)
  print f, "mean grad = ", np.mean(g), "mean loss = ", np.mean(l)


