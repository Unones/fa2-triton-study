Version 2 avec forward et backward prenant en entrée des tenseurs de dimension 4.

Après avoir appliqué l'algorithme de Tri Dao à la lettre et de n'accepter que des tenseurs de 2 dimensions,
j'ai élargi le scope et rendu possible des tenseurs à 4 dimensions : (B, H, N, d).

# I/ Forward Pass

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

Les calculs de ridge point et d'intensité arithmétique sont toujours valables.

On montre donc le benchmark suivant, qui utilise le throughput comme axe de comparaison.

Données pour reproductibilité du benchmark:
- Python 3.13.13
- PyTorch 2.12.0
- Triton 3.7.0
- Dimensions d'entrée : `(32, 32, *, 64)`
- Carte Graphique : RTX 5070 Ti
- Fréquence : 2.30 GHz
- dtype : `BF16`

<img src="benchmark/figures/forward_v2.png" alt="Comparison FA2 triton vs Pytorch on RTX 5070 Ti" width="700">


Mon kernel et l'implémentation de Pytorch sont très proches. Il y a quelques pourcentages d'écart entre les deux.

La ligne en pointillé

Après ce benchmark, qui est très encourageant dans le fait qu'il y a une légère différence entre les deux implémentations,
on fait un profiling sur Nsight Compute pour comprendre ce qu'il se passe.

| Metric | Value Custom kernel | Value PyTorch |
|---|---| --- |
| Compute (SM) throughput | 93.52% | 96.50 |
| Memory throughput | 37.16% | 23.11% |
| DRAM throughput | 4.49% | 4.69% |
| L2 Hit Rate | 96.88% | 94.80% |

Le profiling sur Nsight Compute confirme bien la tendance que l'on est bien compute-bound. Il y a aussi un fait intéressant 
qui distingue les deux kernels. Le kernel PyTorch a une utilisation légèrement meilleure des SMs, ce qui explique
en partie son léger avantage. 

On a donc un forward kernel qui n'est pas parfaitement au niveau de PyTorch mais qui s'en sort très bien.

On peut donc passer au backward car c'est là où il y a le plus de gains par rapport à une implémentation de base.