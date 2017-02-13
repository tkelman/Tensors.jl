# Benchmarks

Here are some benchmark timings for tensors in 3 dimensions. For comparison
the timings for the same operations using standard Julia `Array`s are also
presented.

| Operation  | `Tensor` | `Array` | speed-up |
|:-----------|---------:|--------:|---------:|
| **Single contraction** | | | |
| a ⋅ a | 1.823 ns | 13.557 ns | 7.4× |
| A ⋅ a | 2.807 ns | 79.758 ns | 28.4× |
| A ⋅ A | 7.394 ns | 59.886 ns | 8.1× |
| As ⋅ As | 6.379 ns | 60.087 ns | 9.4× |
| **Double contraction** | | | |
| A ⊡ A | 3.911 ns | 18.570 ns | 4.7× |
| As ⊡ As | 2.781 ns | 18.333 ns | 6.6× |
| AA ⊡ A | 22.955 ns | 109.811 ns | 4.8× |
| AA ⊡ AA | 264.948 ns | 391.900 ns | 1.5× |
| AAs ⊡ AAs | 58.656 ns | 364.040 ns | 6.2× |
| **Outer product** | | | |
| a ⊗ a | 3.585 ns | 299.176 ns | 83.5× |
| A ⊗ A | 43.960 ns | 449.223 ns | 10.2× |
| **Other operations** | | | |
| det(A) | 2.698 ns | 246.061 ns | 91.2× |
| det(As) | 2.611 ns | 238.947 ns | 91.5× |
| inv(A) | 8.380 ns | 859.526 ns | 102.6× |
| inv(As) | 6.425 ns | 843.810 ns | 131.3× |
| norm(a) | 2.169 ns | 11.912 ns | 5.5× |
| norm(A) | 3.931 ns | 21.261 ns | 5.4× |
| norm(As) | 2.637 ns | 21.216 ns | 8.0× |
| norm(AA) | 58.575 ns | 35.292 ns | 0.6× |
| norm(AAs) | 20.150 ns | 35.411 ns | 1.8× |
| a × a | 7.675 ns | 48.481 ns | 6.3× |
