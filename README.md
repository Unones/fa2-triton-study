# fa2-triton-study

Implementation of Flash Attention 2 in Triton kernels to better understand how it works.

## I/ Les résultats

Comparaison entre l'implémentation Flash Attention 2 de PyTorch (`F.scaled_dot_product_attention`)
et mon kernel Triton.

- **Matériel :** NVIDIA RTX 5070 Ti (Blackwell, 70 SM, bande passante HBM théorique 896 GB/s)
- **Précision :** entrées BF16
- **Dimensions :** Q, K, V de forme `(N, d)` avec `d = 64`, soit `(1, 1, N, d)` au format FA2
- **Masquage :** aucun (cas non causal) — toute la matrice S contribue au calcul

> ⚠️ **Limite du benchmark.** La forme `(1, 1, N, d)` n'a **qu'une seule tête**. Elle n'est pas
> représentative d'une charge réelle (où `batch × têtes` vaut des dizaines à des milliers) : elle
> sous-remplit le GPU et force PyTorch dans une stratégie *split-KV* (voir §D). Un benchmark
> multi-têtes est prévu (§E).

<img src="benchmark/figures/forward.png" alt="Comparison FA2 triton vs Pytorch on RTX 5070 Ti" width="700">

### A) Description du graphe

Aux petits `N`, mon kernel devance PyTorch — probablement parce que PyTorch emprunte ici un chemin
*split-KV* en deux kernels (calcul partiel + recombinaison, voir §D) dont l'overhead n'est pas amorti
quand il y a peu de travail, là où mon kernel autotuné tient en un seul lancement.

À partir de `N ≈ 1024`, PyTorch repasse devant : mon kernel n'est qu'une implémentation directe de
l'algorithme de Tri Dao, sans optimisation supplémentaire, et il laisse de la performance sur la table
(voir le profiling §D).

**À propos de l'axe « bande passante ».** La courbe trace une bande passante *algorithmique* : octets
*modélisés* (formule §B) ÷ temps mesuré — **pas** le trafic DRAM réel, que le matériel plafonne à
896 GB/s. C'est pourquoi la courbe PyTorch peut « dépasser » ce plafond : cela signifie simplement que
PyTorch déplace *moins* d'octets réels en DRAM que ma formule ne le suppose (ses tuiles K/V sont
resservies par le cache L2). Le profiling le confirme : sur mon propre kernel, le débit DRAM mesuré
n'est que de **1,25 %** (§D) — à ces tailles, Q/K/V tiennent quasiment en cache et ne touchent presque
jamais la HBM. La métrique « bande passante » est donc un indicateur de débit, pas une mesure de
saturation mémoire.

### B) Calcul des octets transférés

Notations :
- `B_r` : taille de bloc suivant les lignes (requêtes)
- `d` : hidden dimension commune à Q, K, V
- `N` : nombre de lignes de chaque matrice (Q, K, V)

Octets transférés pour un *program* (un bloc de requêtes), en BF16 (2 octets/élément) :
- load `Q_i` (HBM) : `2·B_r·d`
- load tous les `K_j` (HBM) : `2·N·d`
- load tous les `V_j` (HBM) : `2·N·d`
- store `O_i` (HBM) : `2·B_r·d`
- store `L_i` (HBM) : `2·B_r`

**Total : `4·B_r·d + 4·N·d + 2·B_r`.**

### C) Memory-bound ou compute-bound : le roofline idéalisé

L'intensité arithmétique `IA = FLOPs / octets` situe le kernel par rapport au point d'inflexion du
roofline.

**Point d'inflexion.** Les opérations dominantes sont les deux matmuls (tensor cores). Pour une
attention BF16 standard (entrées BF16, accumulation FP32 — la configuration FA2 usuelle), le débit
tensoriel de la carte est de 87,9 TFLOP/s, d'où :

> point d'inflexion = 87,9 TFLOP/s ÷ 896 GB/s ≈ **98 FLOPs/octet**

(En accumulation FP16 le débit serait ~176 TFLOP/s, soit ~196 FLOPs/octet ; ce n'est pas le régime
visé ici.)

**FLOPs du kernel.** Seuls les deux matmuls comptent ; le reste (exp, rescaling de l'online softmax,
arithmétique d'indices) est en `O(B_r·N)`, négligeable devant `O(B_r·N·d)` dès que `d ≫ 1` :
- `2·N·B_r·d` pour `S = Q·Kᵀ`
- `2·N·B_r·d` pour `O += P·V`
- **total : `FLOPs = 4·N·B_r·d`**

**Octets** (en négligeant le terme `2·B_r`) : `4·B_r·d + 4·N·d = 4·d·(B_r + N)`.

D'où :

> `IA = 4·N·B_r·d / (4·d·(B_r + N)) = (B_r·N) / (B_r + N)`

Application numérique (`B_r = 64`, `N = 4096`, `d = 64`) : **`IA ≈ 63 FLOPs/octet`**.

Comme `63 < 98`, ce **modèle idéalisé** prédit un régime memory-bound — *à condition que le kernel
sature la bande passante*. Le profiling (§D) montre que ce n'est pas le cas : le kernel ne sature **ni**
le compute **ni** la mémoire. Le roofline décrit donc une borne que mon kernel n'atteint pas, parce que
son vrai goulot est ailleurs.

### D) Profiling (Nsight Compute, `N = 4096`, `d = 64`)

**Mon kernel `_kernel_fa2_forward` :**

| Métrique | Valeur |
|---|---|
| Compute (SM) throughput | 67,7 % |
| Memory throughput | 13,7 % |
| DRAM throughput | 1,25 % |
| Occupancy (théorique = atteinte) | 8,33 % (4 warps actifs/SM sur 48) |
| Grid | 64 blocs pour 70 SM → 0,91 wave/SM |
| Shared memory dynamique | 65,5 Ko/bloc |
| Limiteur d'occupancy | **shared memory** (1 bloc/SM ; les registres en autoriseraient 2) |

**Diagnostic.** Le kernel n'est ni memory-bound (mémoire 13,7 %, DRAM 1,25 %) ni compute-saturé
(67,7 %) : il est **latency-bound**. Avec seulement 4 warps actifs par SM, il n'y a pas assez de warps
en vol pour cacher la latence des `tl.dot` et des loads, ni pour saturer un pipe. L'occupancy de 8,33 %
est plafonnée par la **shared memory** (65,5 Ko/bloc ⇒ un seul bloc par SM ; Nsight estime un gain
local potentiel de ~92 % en levant cette contrainte).

S'ajoute un **sous-remplissage** : la grille ne compte que 64 blocs pour 70 SM (0,91 wave/SM), donc
certains SM restent inactifs (Nsight signale un déséquilibre de charge, instance minimale à −100 % de
la moyenne). C'est une conséquence directe de la forme single-head `(1, 1, N, d)`.

**PyTorch (`flash_fwd_splitkv_kernel` + `flash_fwd_splitkv_combine_kernel`).** PyTorch utilise une
stratégie *split-KV* (Flash-Decoding) en deux kernels : faute de parallélisme `batch × têtes` (ici = 1),
il découpe la dimension K/V pour occuper les SM, puis recombine les résultats partiels. Son kernel
principal mesure compute 71,0 % / mémoire 29,4 % — plus efficace que le mien sur les deux axes, d'où
l'écart à grand `N`.

### E) Limites et prochaines étapes

- **Benchmark single-head non représentatif.** Refaire avec un `batch × têtes` réaliste (p. ex.
  `B = 8, H = 16`) : PyTorch repassera sur son kernel standard (sans split-KV) et la comparaison sera
  plus juste.
- **Batching `(B, H, N, d)`** via une dimension de grille `B·H` (le cœur 2D du kernel reste inchangé) :
  remplit la grille et rend l'occupancy exploitable.
- **Précision des matmuls.** Le kernel up-converti actuellement Q/K/V en FP32 avant `tl.dot` : les
  matmuls tournent donc en FP32/TF32, pas en BF16. Cela abaisse le plafond compute réel sous les
  98 FLOPs/octet du roofline BF16, et double l'empreinte des tuiles en shared memory — contribuant
  probablement au plafond d'occupancy à 1 bloc/SM. Garder Q/K/V en BF16 pour les matmuls (accumulation
  FP32 via `tl.dot`) est une piste directe.
- **Occupancy.** Balayer `num_warps` (4 → 8) et `num_stages` ; réduire la shared memory par bloc pour
  faire entrer un 2ᵉ bloc/SM.
- **Roofline mesuré.** Tracer les TFLOP/s atteints face aux deux plafonds plutôt qu'une bande passante
  algorithmique.