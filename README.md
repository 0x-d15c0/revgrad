# revgrad
![image](https://github.com/user-attachments/assets/c049a50c-b85e-4f1a-8b8f-e84c5f0f5c7c)

<b>A simple backprop library heavily inspired from Andrej Karpathy's micrograd. Users can build a simple neural network using the library . Tests involve comparison with pytorch's autodiff functions as well.</b>

## Usage
```bash
git clone https://github.com/0x-d15c0/revgrad.git
cd revgrad
pip install - r requirements.txt
```

## Test Case
```py
a = Value(-3.0)
b = Value(1.0)
c = a + b
d = a * b + b**2
c += c + 1
c += 1 + c + (-a)
d += d * 3 + (b + a).relu()
d += 2 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # outcome of this forward pass = 162.0309
g.backward()
print(f'{a.grad:.4f}') # value of dg/da = -143.9726
print(f'{b.grad:.4f}') # value of dg/db =269.9486
```

## Sanity Check [requires pytorch to work!!]
```bash
python test.py
```

## Upcoming features 
1. Visualization using graphviz
2. More activation functions
