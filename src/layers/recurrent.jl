
gate(h, n) = (1:h) .+ h*(n-1)
gate(x::AbstractVector, h, n) = @view x[gate(h,n)]
gate(x::AbstractMatrix, h, n) = x[gate(h,n),:]

# Stateful recurrence

"""
    Recur(cell)

`Recur` takes a recurrent cell and makes it stateful, managing the hidden state
in the background. `cell` should be a model of the form:

    h, y = cell(h, x...)

For example, here's a recurrent network that keeps a running total of its inputs:

```julia
accum(h, x) = (h + x, x)
rnn = Flux.Recur(accum, 0)
rnn(2)      # 2
rnn(3)      # 3
rnn.state   # 5
rnn.(1:10)  # apply to a sequence
rnn.state   # 60
```
"""
mutable struct Recur{T,S}
  cell::T
  state::S
end

function (m::Recur)(xs...)
  h, y = m.cell(m.state, xs...)
  m.state = h
  return y
end

@functor Recur
trainable(a::Recur) = (a.cell,)

Base.show(io::IO, m::Recur) = print(io, "Recur(", m.cell, ")")

"""
    reset!(rnn)

Reset the hidden state of a recurrent layer back to its original value.

Assuming you have a `Recur` layer `rnn`, this is roughly equivalent to:
```julia
rnn.state = hidden(rnn.cell)
```
"""
reset!(m::Recur) = (m.state = m.cell.state)
reset!(m) = foreach(reset!, functor(m)[1])

flip(f, xs) = reverse(f.(reverse(xs)))

# Vanilla RNN

struct RNNCell{F,A,V,S}
  σ::F
  Wi::A
  Wh::A
  b::V
  state::S
end

RNNCell(in::Integer, out::Integer, σ=tanh; init=Flux.glorot_uniform, initb=zeros, init_state=zeros) = 
  RNNCell(σ, init(out, in), init(out, out), initb(out), init_state(out,1))

function (m::RNNCell)(h, x)
  σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
  h = σ.(Wi*x .+ Wh*h .+ b)
  return h, h
end

@functor RNNCell

function Base.show(io::IO, l::RNNCell)
  print(io, "RNNCell(", size(l.Wi, 2), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    RNN(in::Integer, out::Integer, σ = tanh)

The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.
"""
Recur(m::RNNCell) = Recur(m, m.state)
RNN(a...; ka...) = Recur(RNNCell(a...; ka...))

# LSTM

struct LSTMCell{A,V,S}
  Wi::A
  Wh::A
  b::V
  state::S
end

function LSTMCell(in::Integer, out::Integer;
                  init = glorot_uniform,
                  initb = zeros,
                  init_state = zeros)
  cell = LSTMCell(init(out * 4, in), init(out * 4, out), initb(out * 4), (init_state(out), init_state(out)))
  cell.b[gate(out, 2)] .= 1
  return cell
end

function (m::LSTMCell)((h, c), x)
  b, o = m.b, size(h, 1)
  g = m.Wi*x .+ m.Wh*h .+ b
  input = σ.(gate(g, o, 1))
  forget = σ.(gate(g, o, 2))
  cell = tanh.(gate(g, o, 3))
  output = σ.(gate(g, o, 4))
  c = forget .* c .+ input .* cell
  h′ = output .* tanh.(c)
  return (h′, c), h′
end

@functor LSTMCell

Base.show(io::IO, l::LSTMCell) =
  print(io, "LSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4, ")")

"""
    LSTM(in::Integer, out::Integer)

[Long Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
recurrent layer. Behaves like an RNN but generally exhibits a longer memory span over sequences.

See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
# Recur(m::LSTMCell) = Recur(m, (zeros(length(m.b)÷4), zeros(length(m.b)÷4)),
#   (zeros(length(m.b)÷4), zeros(length(m.b)÷4)))
Recur(m::LSTMCell) = Recur(m, m.state)
LSTM(a...; ka...) = Recur(LSTMCell(a...; ka...))

# GRU

struct GRUCell{A,V,S}
  Wi::A
  Wh::A
  b::V
  state::S
end

GRUCell(in, out; init = glorot_uniform, initb = zeros, init_state = zeros) =
  GRUCell(init(out * 3, in), init(out * 3, out), initb(out * 3), init_state(out,1))

function (m::GRUCell)(h, x)
  b, o = m.b, size(h, 1)
  gx, gh = m.Wi*x, m.Wh*h
  r = σ.(gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
  z = σ.(gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
  h̃ = tanh.(gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
  h′ = (1 .- z).*h̃ .+ z.*h
  return h′, h′
end

@functor GRUCell

Base.show(io::IO, l::GRUCell) =
  print(io, "GRUCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ")")

"""
    GRU(in::Integer, out::Integer)

[Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.

See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
Recur(m::GRUCell) = Recur(m, m.state)
GRU(a...; ka...) = Recur(GRUCell(a...; ka...))

@adjoint function Broadcast.broadcasted(f::Recur, args...)
  Zygote.∇map(__context__, f, args...)
end
