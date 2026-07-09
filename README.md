# Results

Comparison between PyTorch's Flash Attention 2 implementation (`F.scaled_dot_product_attention`)
and my Triton kernel.

- **Hardware:** NVIDIA RTX 5070 Ti (Blackwell, 70 SMs, theoretical throughput of FP16
tensor cores with FP32 accumulation with 2.30 GHz clock : `82.5 TFLOP/s`)
- **Precision:** BF16 inputs
- **Dimensions:** Q, K, V of shape `(B, H, N, d)` with `B = H = 32`, `d = 64`.
- **Masking:** none (non-causal) - the full S matrix contributes to the computation
- **Software:** Python `3.13.13`, PyTorch `2.12.0`, Triton `3.7.0`

**Forward Pass benchmark (4-dimensional input tensors):**

<img src="benchmark/figures/forward_v2.png" alt="Comparison forward v2 FA2 triton vs Pytorch on RTX 5070 Ti" width="700">

**Backward Pass benchmark (4-dimensional input tensors):** 

<img src="benchmark/figures/backward.png" alt="Comparison backward FA2 triton vs Pytorch on RTX 5070 Ti" width="700">


# I/ 2-dimensional Forward Pass


> *WARNING :* The shape `(1, 1, N, d)` has **a single head**. It is not representative of
> a real workload (where `batch × heads` is in the tens to thousands): it underfills the GPU and forces
> PyTorch into a *split-KV* strategy.

<img src="benchmark/figures/forward_v1.png" alt="Comparison forward v1 FA2 triton vs Pytorch on RTX 5070 Ti" width="700">

### A) Reading the plot

At small `N`, my kernel edges out PyTorch - most likely because PyTorch takes a *split-KV* path here in
two kernels (partial compute + recombination, see §D) whose overhead isn't amortized when there's
little work, whereas my autotuned kernel runs in a single launch.

From `N ≈ 1024` onward, PyTorch pulls ahead: my kernel is only a direct implementation of Tri Dao's
algorithm, with no further optimization, and leaves performance on the table.

**About the "bandwidth" axis.** The curve plots an *algorithmic* bandwidth: *modeled* bytes
(formula §B) ÷ measured time - **not** the real DRAM traffic, which the hardware caps at 896 GB/s.
This is why the PyTorch curve can "exceed" that ceiling: it simply means PyTorch moves *fewer* real
DRAM bytes than my formula assumes (its K/V tiles are served back from the L2 cache). Profiling
confirms it: on my own kernel, measured DRAM throughput is only **1.25%** - at these sizes,
Q/K/V essentially fit in cache and almost never hit HBM. The "bandwidth" metric is therefore a
throughput indicator, not a measure of memory saturation.

### B) Counting bytes transferred

Notation:
- `B_r`: block size along the rows (queries)
- `d`: hidden dimension shared by Q, K, V
- `N`: number of rows of each matrix (Q, K, V)

Bytes transferred for one *program* (one query block), in BF16 (2 bytes/element):
- load `Q_i` (HBM): `2·B_r·d`
- load all `K_j` (HBM): `2·N·d`
- load all `V_j` (HBM): `2·N·d`
- store `O_i` (HBM): `2·B_r·d`
- store `L_i` (HBM): `2·B_r`

**Total: `4·B_r·d + 4·N·d + 2·B_r`.**

### C) Memory-bound or compute-bound: the idealized roofline

Arithmetic intensity `AI = FLOPs / bytes` places the kernel relative to the roofline's ridge point.

**Ridge point.** The dominant operations are the two matmuls (tensor cores). For a standard BF16
attention (BF16 inputs, FP32 accumulation - the usual FA2 configuration), the card's tensor throughput
is 87.9 TFLOP/s, hence:

> ridge point = 87.9 TFLOP/s ÷ 896 GB/s ≈ **98 FLOPs/byte**

(With FP16 accumulation the throughput would be ~176 TFLOP/s, i.e. ~196 FLOPs/byte; that's not the
regime targeted here.)

**Kernel FLOPs.** Only the two matmuls matter; the rest (exp, online-softmax rescaling, index
arithmetic) is `O(B_r·N)`, negligible against `O(B_r·N·d)` as soon as `d ≫ 1`:
- `2·N·B_r·d` for `S = Q·Kᵀ`
- `2·N·B_r·d` for `O += P·V`
- **total: `FLOPs = 4·N·B_r·d`**

**Bytes** (dropping the `2·B_r` term): `4·B_r·d + 4·N·d = 4·d·(B_r + N)`.

Hence:

> `AI = 4·N·B_r·d / (4·d·(B_r + N)) = (B_r·N) / (B_r + N)`

Numerical application (`B_r = 64`, `N = 4096`, `d = 64`): **`AI ≈ 63 FLOPs/byte`**.

Since `63 < 98`, this **idealized model** predicts a memory-bound regime - *provided the kernel
saturates the bandwidth*. Profiling (§D) shows it does not: the kernel saturates **neither** compute
**nor** memory. The roofline therefore describes a bound my kernel doesn't reach, because its real
bottleneck lies elsewhere.

Après avoir appliqué l'algorithme de Tri Dao à la lettre et de n'accepter que des tenseurs de 2 dimensions,
on peut élargir le scope et rendu possible des tenseurs à 4 dimensions : (B, H, N, d).

# II/ 4-dimensional Forward Pass

Avant de faire quelconque profiling que ce soit, faisons d'abord un benchmark pour voir comment s'en sort le
kernel. J'ai mis en place de l'autotunue en couvrant une assez grande plage de `block sizes` et de `num_stages`.

Avant de lancer le benchmark, il faut savoir dans quel régime on est. En effet, même si, d'après l'algorithme,
il y a 2 produits matriciels, on a vu dans la première version que l'on peut-être memory-bound.

Par conséquent, reprenons les données et calculs importants du GPU sur lequel je fais tourner ces kernels :
- Peak Bandwidth de `896 GB/s`
- Peak Tensor Cores FP16 avec accumulation FP32 de `87.9 TLFOP/s`
- Nombre total de bytes transférés de `8*B*H*N*d`
- Nombre total de FLOPs de `4*B*H*N*N*d`

Avec les deux premières caractéristiques du GPU, on sait que le ridge point est à `98 FLOPs / byte`.
Or, dans notre cas, on va s'intéresser aux dimensions suivantes : 
- `B = 32`
- `H = 32`
- `N = 4096`
- `d = 64`

Soit, `IA =  4*B*H*N*N*d / 8*B*H*N*d`.

D'où, `IA = N / 2`.

Pour ces dimensions, nous sommes dans un régime complètement compute-bound. On l'est même à partir de `N = 256`,
ce qui représente `2⁸`, qui est la première puissance de 2 à partir de laquelle on est dans le régime
compute-bound.

Dans toute la suite, que ce soit benchmark ou profiling, forward ou backward, le fréquence du GPU est bloqué
à 2.30 GHz. Par conséquent, le nombre de Peak TFLOPs est diminué à `82.5 TFLOP/s` car la donnée `87.9 TLFOP/s`
est pour la fréquence boostée de `2.452 GHz`.

Les calculs de ridge point et d'intensité arithmétique sont toujours valables. En effet, le facteur de baisse de
fréquence est valable pour le nombre de FLOPs et le nombre de bytes transférés. Les facteurs aux numérateur et 
au dénominateur se compensent donc. La valeur du ridge point est donc toujours la même.

On montre donc le benchmark suivant, qui utilise le throughput comme axe de comparaison.

<img src="benchmark/figures/forward_v2.png" alt="Comparison forward v2 FA2 triton vs Pytorch on RTX 5070 Ti" width="700">


Mon kernel et l'implémentation de Pytorch sont très proches. Il y a quelques pourcentages d'écart entre les deux.

La ligne en pointillé

Après ce benchmark, qui est très encourageant dans le fait qu'il y a une légère différence entre les deux implémentations,
on fait un profiling sur Nsight Compute pour comprendre ce qu'il se passe.

| Metric | Value Custom kernel | Value PyTorch |
|---|---|---|
| Compute (SM) throughput | 93.52% | 96.50% |
| Memory throughput | 37.16% | 23.11% |
| DRAM throughput | 4.49% | 4.69% |
| L2 Hit Rate | 96.88% | 94.80% |

Le profiling sur Nsight Compute confirme bien la tendance que l'on est bien compute-bound. Il y a aussi un fait intéressant 
qui distingue les deux kernels. Le kernel PyTorch a une utilisation légèrement meilleure des SMs, ce qui explique
en partie son léger avantage. 

On a donc un forward kernel qui n'est pas parfaitement au niveau de PyTorch mais qui s'en sort très bien.

On peut donc passer au backward car c'est là où il y a le plus de gains par rapport à une implémentation de base.


# II/ 4-dimensional Backward Pass

On reprend la méthode de la partie précédente mais en adaptant les données.

Sur la fiche technique de la RTX 5070 Ti, le peak Tensor Cores FP16 avec accumulation FP32 est de `87.9 TLFOP/s`.
Cependant, cela est valable que pour une fréquence boostée de `2.45 GHz`.

Puisque la fréquence de base de cette carte graphique est de `2.30 GHz`, j'ai décidé de la bloquer à cette fréquence.

Par conséquent, la nouvelle valeur du peak Tensor Cores FP16 avec accumulation FP32 est de `82.5 TFLOP/s`.

Reprenons les données et calculs importants du GPU sur lequel je fais tourner ces kernels :
- Peak Bandwidth de `840.5 GB/s`
- Peak Tensor Cores FP16 avec accumulation FP32 de `82.5 TLFOP/s`
- Nombre total de bytes transférés de `18*B*H*N*d`
- Nombre total de FLOPs de `10*B*H*N*N*d`

Le facteur `10` pour le nombre de bytes transférés vient du fait que l'on ne prend en compte que les 5 transferts les
plus importants : `q_tensor`, `k_tensor`, `v_tensor`, `o_tensor`, `do_tensor`, `dq_tensor` (compte double car 
accumulation en `FP32`), `dk_tensor` et `dv_tensor`.


Le facteur `10` pour le nombre total de FLOPs vient du fait que l'on a 5 produits matriciels.

Par conséquent, l'intensité arithmétique vaut : `IA = (10*N)/18`.

Ainsi, pour se donner une marge de manoeuvre avec les approximations mises en place, on peut dire que l'on est en régime
compute-bound pour `N = 256`.

<img src="benchmark/figures/backward.png" alt="Comparison backward FA2 triton vs Pytorch on RTX 5070 Ti" width="700">


Le benchmark est très intéressant à lire. Premièrement, l'implémentation PyTorch a du mal à atteindre le pic du 
throughput. À voir au moment du profiling si le kernel est potentiellement latency-bound. Deuxièmement, mon kernel
est en moyenne deux fois plus lent que l'implémentation de PyTorch.

Une potentielle cause est l'utilisation des atomic add qui peuvent stall le processus le temps de faire le trajet en
mémoire.

Pour cela, on fait le profiling de la shape suivante : `(32, 32, 4096, 64)`.

En premier lieu, on obtient plusieurs kernels qui s'exécutent (je ne prends pas en compte pas les 
`vectorized_elementwise_kernel` de PyTorch). Les trois colonnes proviennent du résumé de Nsight Compute.

| Estimated Speedup (%) | Function Name | Duration (ms) |
|---|---|---|
| 5.55% | _kernel_D_fa2 | 1.31 ms |
| 31.69% | _kernel_fa2_backward | 276.91 ms |
| 22.32% | flash_bwd_dot_do_o_kernel | 3.13 ms |
| 7.95% | flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel | 134.81 ms |
| 1.29% | flash_bwd_convert_dq_kernel | 2.09 ms |

À l'aide du résumé du profiling, on peut déjà voir se confirmer la tendance que mon kernel principal `_kernel_fa2_backward`
est environ 2 fois plus lent que l'implémentation PyTorch.

Analysons un peu plus en profondeur les métriques de Nisght Compute de mon kernel pour voir son comportement.

| Metric | Value |
|---|---|
| Compute (SM) throughput | 72.24% |
| Memory throughput | 32.71% |
| L1 Hit Rate | 32.73% |
| L2 Hit Rate | 22.17% |
| DRAM throughput | 2.66% | 

Le kernel est bel et bien pas memory-bound, mais on ne peut cependant pas affirmer qu'il soit compute-bound non plus.
`72%` de compute throughput est bien mais pas suffisant pour dire que le compute est le facteur limitant ici.

La latence peut potentiellement être le problème mais peu de métriques permettent de s'assurer que c'est bien le cas.
Cependant, dans Nsight Compute, il est indiqué que l'on a des warp stalls.

On peut donc regarder dans le code source SASS pour voir quelles sont les instructions qui stall.

On observe que les instructions Global Atomic suivantes sont responsables d'un stall moyen de 11% : 
```
ATOMG.E.ADD.F32*4.FTZ.RN.STRONG.GPU PT, RZ, desc[UR16][R112.64], R148
```

On a donc bien des atomic adds qui sont problèmétiques et stall le kernel.