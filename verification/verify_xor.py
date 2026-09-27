# GenAI disclosure: written by Claude (Anthropic) to check the Assignment 3 answers numerically.
import torch
torch.set_default_dtype(torch.float64)
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]); Y = torch.tensor([0.,1.,1.,0.])
# Kronecker check for Section 2 EC
torch.manual_seed(0); n,d,kk = 3,4,2; Xm = torch.randn(n,d); W = torch.randn(kk,d)
J = torch.autograd.functional.jacobian(lambda W: Xm@W.T, W)   # (a,b,p,q)
rows_rowmajorF = J.permute(0,1,3,2).reshape(n*kk, d*kk)          # rows a*k+b, cols q*k+p (= column-major vec of W)
print("dvec_r(XW^T)/dvec_c(W) == kron(X, I_k):", torch.allclose(rows_rowmajorF, torch.kron(Xm, torch.eye(kk))))

# 4.2 hand-picked weights
W1 = torch.tensor([[1.,1.],[1.,1.]]); b1 = torch.tensor([0.,-1.]); W2 = torch.tensor([1.,-2.]); b2 = 0.
H = torch.relu(X@W1.T + b1); print("4.2 z1:", (X@W1.T+b1).tolist(), " h:", H.tolist(), " yhat:", (H@W2+b2).tolist())

def train(act, init, steps=5000, lr=0.1, hidden=2):
    p = {k_: torch.full(s, init, requires_grad=True) for k_, s in
         dict(W1=(hidden,2), b1=(hidden,), W2=(hidden,), b2=()).items()}
    opt = torch.optim.SGD(p.values(), lr=lr)
    for t in range(steps):
        opt.zero_grad()
        yhat = act(X@p['W1'].T + p['b1'])@p['W2'] + p['b2']
        loss = ((yhat-Y)**2).mean(); loss.backward(); opt.step()
    return {k_: v.detach() for k_,v in p.items()}, yhat.detach(), loss.item()

print("\n4.3 zero init, ReLU, MSE, full-batch GD, 5000 steps")
p, yhat, loss = train(torch.relu, 0.0)
print(" W1", p['W1'].tolist(), " b1", p['b1'].tolist(), " W2", p['W2'].tolist(), " b2", round(p['b2'].item(),4))
print(" yhat", [round(v,4) for v in yhat.tolist()], " loss", round(loss,4))

print("\n4.4 constant init, several activations")
acts = {'sigmoid': torch.sigmoid, 'tanh': torch.tanh, 'relu': torch.relu,
        'gelu': torch.nn.functional.gelu, 'square': lambda z: z**2}
for name, a in acts.items():
    for c in ([0.5, -0.3] if name == 'square' else [0.5, -0.3, 1.0]):  # square diverges at c=1.0 with lr 0.05
        try:
            p, yhat, loss = train(a, c, steps=20000, lr=0.05)
            rows_equal = torch.allclose(p['W1'][0], p['W1'][1]) and torch.allclose(p['b1'][0], p['b1'][1]) and torch.allclose(p['W2'][0], p['W2'][1])
            cols_equal = torch.allclose(p['W1'][:,0], p['W1'][:,1])
            print(f" {name:7s} c={c:5}: hidden units identical={rows_equal} input cols equal={cols_equal} "
                  f"yhat={[round(v,3) for v in yhat.tolist()]} loss={loss:.3e}")
        except Exception as e:
            print(name, c, 'err', e)
