Calculs de l'intensité arithmétique du forward pass de flash attention 2 en FP16.

Par définition, l'intensité arithmétique vaut : AI = (FLOPs / bytes transferred).
Dans le cas d'une RTX 5070 Ti, le max FLOPS théorique vaut `43,9 TFLOPs` et la 
bandwidth théorique max vaut `896 GB/s`.

Par conséquent, au point d'inflexion du roofline model, on a `IA = 49 FLOPs / byte`.

On utilise les notations suivantes : 
- `B_r` pour la taille du bloc suivant les lignes
- `d` la hidden dimension commune aux trois matrices d'entrée
- `

Calculons le nombre de `bytes trransferred` pour le forward pass (un program): 
- load `Q_i` depuis la HBM = `2*B_r*d` bytes 
- load par bout `K_j` depuis la HBM = `2*N*d` bytes
- load par bout `V_j` depuis la HBM = `2*N*d` bytes
- store `O_i` vers la HBM = `2*B_r*d` bytes
- store `L_i` depuis la HBM = `2*B_r` bytes

Ainsi, il y a un total de bytes transferred valant ` 4*B_r*d + 4*N*d + 2*B_r`.


------ à partir d'ici, les `B_c` sont à substituer par `N` car on ne regarde que pour une unique
itération de la boucle interne -----------

Maintenant, calculons le nombre d'opérations exécutées par un unique program :
- produit matriciel d'une matrice de dimension `(B_r, d)` et `(d, B_c)` = `2*B_c*B_r*d` opérations
- rowmax puis max pour calculer `m_i` = `négligeable`
- soustraction pour le calcul de `P_i` = `B_r*B_c` opérations
- prise en compte de l'exponentielle par éléments pour le calcule de `P_i` = `4*B_r*B_c` opérations
- calcul complet pour `l_i` = `5*B_r + B_r*B_c` opérations
- calcul du premier terme de `O_i` = `B_r*B_c` opérations
- calcul du deuxième terme de `O_i` = `2*B_r*B_c*d` opérations

------- on sort de la boucle interne ---------
- dernier calcul pour `O_i` = `B_r*B_c` opérations
- calcul pour `L_i` = `B_r + 4*B_r` opérations


Vu le coût des produits matriciels et qu'en moyenne : ` d >> 10`, alors on peut négliger toutes les 
opérations à part les deux produits matriciels à chaque boucle interne.

Aussi, on réajuste le calcul initial de l'intensité arithmétique.
Puisque l'on utilise les tensor cores, alors le max FLOPs théorique est de `IA = 98 FLOPs / byte`.

On reprend les FLOPs : 
- `2*N*B_r*d` FLOPs pour la première mutliplication matricielle
- `2*B_r*N*d` FLOPs pour la seconde multiplication matricielle.

D'où : FLOPs = `4*N*B_r*d`

Puis, en simplifiant le nombre total de bytes transferred : `4*B_r*d + 4*N*d`

Par conséquent : `IA_kernel` = ` (B_r*N) /(B_r + N)`.

Application numérique : 
- `B_r = 8`
- `N = 128`
- `d = 128` (limite la taille de `B_r` et `B_c`)

Ainsi, `IA_kernel = 7,53 FLOPs/byte`.

Nous sommes donc en régime memory-bound. Cependant, ça ne sera pas toujours le cas.
Pour `N` fixé, en augmentant `B_r`, on peut atteindre le pic de l'intensité arithmétique.

Fixons `N=1024`. Alors, pour avoir l'intensité arithmétique à `98 FLOPs / byte`, on a besoin de
`B_r = 128` au minimum.

