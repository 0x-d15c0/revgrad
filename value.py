class Value:

    def __init__(self, data, _children=(), op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data} | grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        if isinstance(other, int):
            out = Value(self.data**other, (self,), f'**{other}')

            def _backward():
                self.grad += other * (self.data ** (other - 1)) * out.grad
            out._backward = _backward
            return out
        else:
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data**other.data, (self, other), '**')

            def _backward():
                self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
                other.grad += self.data ** other.data * math.log(self.data) * out.grad
            out._backward = _backward
            return out

    def __truediv__(self, other):
        return self * other**-1
  
    def __rtruediv__(self, other):
        return other * self**-1

    def __neg__(self):
        return self * -1

    def __exp__(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topology = []
        visited_nodes = set()

        def build_topology(v):
            if v not in visited_nodes:
                visited_nodes.add(v)
                for child in v._prev:
                    build_topology(child)
                topology.append(v)

        build_topology(self)
        self.grad = 1.0

        for node in reversed(topology):
            node._backward()
