import torch
from value import Value

def test_sanity_check():
    # Custom Value class
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    # PyTorch equivalent
    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # Check forward pass
    assert ymg.data == ypt.data.item(), f"Expected {ypt.data.item()}, but got {ymg.data}"
    # Check backward pass
    assert xmg.grad == xpt.grad.item(), f"Expected {xpt.grad.item()}, but got {xmg.grad}"

def test_more_ops():
    # Custom Value class
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    # PyTorch equivalent
    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # Check forward pass
    assert abs(gmg.data - gpt.data.item()) < tol, f"Expected {gpt.data.item()}, but got {gmg.data}"
    # Check backward pass
    assert abs(amg.grad - apt.grad.item()) < tol, f"Expected {apt.grad.item()}, but got {amg.grad}"
    assert abs(bmg.grad - bpt.grad.item()) < tol, f"Expected {bpt.grad.item()}, but got {bmg.grad}"

if __name__ == "__main__":
    print("Running sanity check...")
    test_sanity_check()
    print("Sanity check passed!")

    print("Running more operations check...")
    test_more_ops()
    print("More operations check passed!")
