using Flux, CUDA, Test
using Flux: pullback

using Flux
m = RNN(5, 5)
loss(x) = sum(Flux.stack(m.(x), 2))
x = [rand(5) for i in 1:2]
Flux.train!(loss, Flux.params(m), x, ADAM())

@testset for R in [RNN, GRU, LSTM]
  m = R(10, 5) |> gpu
  x = gpu(rand(10))
  (m̄,) = gradient(m -> sum(m(x)), m)
  Flux.reset!(m)
  θ = gradient(() -> sum(m(x)), params(m))
  @test x isa CuArray
  @test_broken θ[m.cell.Wi] isa CuArray
  @test_broken collect(m̄[].cell[].Wi) == collect(θ[m.cell.Wi])
end

@testset "RNN" begin
  @testset for R in [RNN, GRU, LSTM], batch_size in (1, 5)
    rnn = R(10, 5)
    curnn = fmap(gpu, rnn)

    Flux.reset!(rnn)
    Flux.reset!(curnn)
    x = batch_size == 1 ?
      rand(10) :
      rand(10, batch_size)
    cux = gpu(x)

    y, back = pullback((r, x) -> r(x), rnn, x)
    cuy, cuback = pullback((r, x) -> r(x), curnn, cux)

    @test y ≈ collect(cuy)

    @test haskey(Flux.CUDAint.descs, curnn.cell)

    ȳ = randn(size(y))
    m̄, x̄ = back(ȳ)
    cum̄, cux̄ = cuback(gpu(ȳ))

    @test x̄ ≈ collect(cux̄)
    @test_broken m̄[].cell[].Wi ≈ collect(cum̄[].cell[].Wi)
    @test_broken m̄[].cell[].Wh ≈ collect(cum̄[].cell[].Wh)
    @test_broken m̄[].cell[].b ≈ collect(cum̄[].cell[].b)
    if m̄[].state isa Tuple
      for (x, cx) in zip(m̄[].state, cum̄[].state)
        @test x ≈ collect(cx)
      end
    else
      @test m̄[].state ≈ collect(cum̄[].state)
    end

    Flux.reset!(rnn)
    Flux.reset!(curnn)
    ohx = batch_size == 1 ?
      Flux.onehot(rand(1:10), 1:10) :
      Flux.onehotbatch(rand(1:10, batch_size), 1:10)
    cuohx = gpu(ohx)
    y = (rnn(ohx); rnn(ohx))

    # TODO: FIX ERROR
    @test_broken 1 == 2
    # cuy = (curnn(cuohx); curnn(cuohx))
    # @test y ≈ collect(cuy)
  end
end
