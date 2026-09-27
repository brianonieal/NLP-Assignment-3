# GenAI disclosure: written by Claude (Anthropic) to test the Section 6.1.5 explanation for lr = 0.1.
# Trains the single-512 sigmoid MLP for 2 epochs at lr 0.1 and 0.02 and reports how many hidden
# sigmoid outputs are saturated (below 0.01 or above 0.99) on the dev set after each epoch.
import os, sys, torch, gensim.downloader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hw3"))
from mlp import load_data_mlp, create_tensor_dataset, create_dataloader, SentimentClassifier, evaluate
torch.manual_seed(42)
dev, train, _ = load_data_mlp()
emb = gensim.downloader.load("glove-twitter-50")
train_ds, dev_ds = create_tensor_dataset(train, emb), create_tensor_dataset(dev, emb)
for lr in [0.1, 0.02]:
    torch.manual_seed(42)
    model = SentimentClassifier(50, 2, [512], "Sigmoid")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(2):
        model.train()
        for x, y in create_dataloader(train_ds, 64, shuffle=True):
            opt.zero_grad(); loss = model.loss(model(x), y); loss.backward(); opt.step()
        with torch.no_grad():
            h = torch.sigmoid(model.linears[0](dev_ds.tensors[0]))
            sat = ((h < 0.01) | (h > 0.99)).float().mean().item()
            w_norm = model.linears[0].weight.norm(dim=1).mean().item()
        dl, da = evaluate(model, create_dataloader(dev_ds, 64, shuffle=False))
        print(f"RESULT lr={lr} epoch={epoch} saturated_frac={sat:.3f} mean_row_norm_W1={w_norm:.1f} dev_loss={dl:.3f} dev_acc={da:.3f}", flush=True)
