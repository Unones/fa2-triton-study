Version 2 avec forward et backward prenant en entrée des tenseurs de dimension 4.

Après avoir appliqué l'algorithme de Tri Dao à la lettre et de n'accepter que des tenseurs de 2 dimensions,
j'ai élargi le scope et rendu possible des tenseurs à 4 dimensions : (B, H, N, d).

# I/ Forward Pass

Avant de faire quelconque profiling que ce soit, faisons d'abord un benchmark pour voir comment s'en sort le
kernel. J'ai mis en place de l'autotunue en couvrant une assez grande plage de `block sizes` et de `num_stages`.

Avant de lancer le benchmark, il faut savoir dans quel régime on est. En effet, même si, d'après l'algorithme,
il y a 3 produits matriciels, on a vu dans la première version que l'on peut-être memory-bound.

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
ce qui représente `2⁸`.

On montre donc le benchmark suivant, qui utilise le trhoughput comme axe de comparaison.

<img src="benchmark/figures/forward_v2.png" alt="Comparison FA2 triton vs Pytorch on RTX 5070 Ti" width="700">


Mon kernel et l'implémentation de Pytorch sont très proches. Il y a quelques pourcentages d'écart entre les deux.

La ligne de pointillés est dépassée par les deux kernels, mais j'ai utilisé la formule suivante : 

```python
flops = 4*N*N*d*H*B
```

Il y a d'autres opérations coûteuses et non optimisées dans le kernel, tel que des sommes, des exponentielles et autres.
Par conséquent, la ligne est pointillée se fait dépasser artificiellement par les deux kernels car le formule du décompte
des flops n'est pas exacte.