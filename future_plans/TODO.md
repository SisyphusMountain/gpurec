- [ ] Implement implicit gradient descent and test different algorithms
- [ ] Test performant methods for conjugate gradient solving
- [ ] Implement efficient data preparation: either program in Rust or reuse the ALERax pipeline (including using several gene trees and hashing them properly for CCP computations)
- [ ] Implement efficient sampling of reconciliations.
- [ ] Implement scenario sampling efficiently.
- [ ] Implement fixed-point iteration by gradient descent on the norm of F(Pi) - Pi
- [ ] Implement true second order method for efficiency
Pour les gros arbres, il faut de nombreuses itérations avant convergence totale de Pi... Comment faire pour que ce soit plus rapide? Dans ALERAX, ils n'ont besoin que d'environ 4 itérations, car ils utilisent la programmation dynamique pour la mise à jour?

On pourrait essayer d'obtenir une meilleure initialisation avant de faire le point fixe. Pour cela, on peut enlever la contrainte de transferts uniquement vers non-ancêtres?

Compiler avec torch AOT inductor : donne code C++

Il y a un préordre sur les clades qu'on peut utiliser tout en gardant une part de parallélisme.

BUG: Revoir la fonction ScatterLogSumExp: probablement inefficace, et recalcule 2 fois la même chose pour créer le contexte pour le backward.

On peut batcher en concaténant simplement les [G, S] selon l'axe des G. Les matrices qui s'appliquent sont de toute façon des opérations sparse, donc aucun problème. Le parallélisme devrait rendre plus efficace l'optimisation. Il faut essayer avec un ensemble d'arbres de gènes, et vérifier qu'on a bien les mêmes log-likelihoods que ALERax.


Observation : sur la trace JSON, on observe beaucoup de temps GPU idle. Il faut fusionner les kernels. Pour ce faire, on peut utiliser les [CUDA graphs](https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html)

Préallouer la mémoire pour tous les tenseurs afin de simplifier le code.