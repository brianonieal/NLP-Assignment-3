# GenAI disclosure: written by Claude (Anthropic) to check the Assignment 3 answers numerically.
import torch, math, numpy as np
from torch.autograd.functional import jacobian
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
d,n,k = 4,3,2
w = torch.randn(d); X = torch.randn(n,d); xi = X[0]; W = torch.randn(k,d)
print("== Section 2 Jacobians (shape, matches closed form) ==")
cases = {
 "c":        (lambda w: torch.tensor(5.0)+0*w.sum(), torch.zeros(d)),
 "||w||^2":  (lambda w: (w*w).sum(), 2*w),
 "w^T x_i":  (lambda w: w@xi, xi),
 "Xw":       (lambda w: X@w, X),
 "w":        (lambda w: w, torch.eye(d)),
 "w^2":      (lambda w: w**2, torch.diag(2*w)),
}
for name,(f,closed) in cases.items():
    J = jacobian(f, w)
    print(f"{name:9s} autograd shape {tuple(J.shape)!s:8s} match={torch.allclose(J,closed)}")
J = jacobian(lambda W: X@W.T, W)          # shape (n,k,k,d)
closed = torch.einsum('bp,aq->abpq', torch.eye(k), X)   # delta_{bp} X_{aq}
print("XW^T      autograd shape", tuple(J.shape), "match delta_bp X_aq:", torch.allclose(J, closed))
# F flattened row by row (index a*k+b), W flattened column by column (index q*k+p)
Jflat = J.permute(0, 1, 3, 2).reshape(n*k, d*k)
print("flattened Jacobian == kron(X, I_k):", torch.allclose(Jflat, torch.kron(X, torch.eye(k))))

print("\n== Section 3 softmax ==")
z = torch.randn(5); s = torch.softmax(z,0)
J = jacobian(lambda z: torch.softmax(z,0), z)
print("diag(s)-ss^T match:", torch.allclose(J, torch.diag(s)-torch.outer(s,s)))
print("elementwise s_i(delta_ij - s_j) match:", torch.allclose(J, s[:,None]*(torch.eye(5)-s[None,:])))
print("rows of J sum to 0:", torch.allclose(J.sum(1), torch.zeros(5)))
xs = torch.linspace(-10,10,20001)
sig = torch.sigmoid(xs); print("max sigmoid' =", float((sig*(1-sig)).max()), "at x =", float(xs[(sig*(1-sig)).argmax()]))
print("max tanh' =", float((1-torch.tanh(xs)**2).max()))
print("sigmoid'(10) =", float(torch.sigmoid(torch.tensor(10.))*(1-torch.sigmoid(torch.tensor(10.)))))
print("sigmoid'(5) =", float(torch.sigmoid(torch.tensor(5.))*(1-torch.sigmoid(torch.tensor(5.)))))

print("\n== Section 5 computation graph ==")
x = torch.tensor(1., requires_grad=True); y = torch.tensor(3., requires_grad=True); zz = torch.tensor(2., requires_grad=True)
h1 = torch.log(x); h2 = torch.exp(y); h3 = h2*zz; f = h1+h3
f.backward()
print("h1,h2,h3,f =", float(h1), float(h2), float(h3), float(f))
print("df/dx, df/dy, df/dz =", float(x.grad), float(y.grad), float(zz.grad))
print("e^3 =", math.exp(3), " 2e^3 =", 2*math.exp(3))
