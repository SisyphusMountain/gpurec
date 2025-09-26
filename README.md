# gpurec
Porting the AleRax algorithm to GPU.


- On peut commencer l'optimisation en float32: ça donne une précision de 10^-7, puis float64 donne une précision de 10^-16
- Implémenté des kernels Triton pour logsumexp sur 4, 5, 7 termes (nécessaires pour faire certaines opérations dans Pi update et E update. J'ai fait ça car les opérations de stack et logsumexp de PyTorch prenaient beaucoup de temps. On pourrait dès le départ allouer un vecteur de taille [4, ...] par exemple et travailler sur les lignes. Pas sûr que ça ait un grand avantage cependant, on verra plus tard sur la trace)
- Problème à résoudre : ScatterLogSumExp = opération très lente. Solution en cours d'implémentation = écrire un kernel triton, grouper les Clade splits et utiliser un vecteur PTR qui donne le début et la fin de chaque section à réduire. Faire attention d'avoir trié ces segments par taille. Ensuite, on peut simplement appliquer la réduction uniquement aux segments de longueur supérieure ou égale à 2.
- Quand on sait qu'un tenseur va être aggrégé, par exemple avec logsumexp, stocker les différentes colonnes dans un même tenseur préalablement alloué.
- Implémenter descente de gradient et batching pour travailler sur plusieurs arbres de gènes à la fois. Etapes d'implémentation:
  - Utiliser repeat_interleave pour avoir pS, pD, pT, pL adaptés à l'arbre.
  - Si on veut granularité par branche + par arbre, alors on a vraiment des tenseurs de la taille nb_sp*nb_gene_trees qu'il faudra repeat_interleave avant de sommer dans dup_both_survive. 
  - S: repeat_interleave sur log_pS
  - SL: repeat_interleave sur log_pS, sur E_s1 et E_s2 (qu'on devrait concaténer en E_s12, repeat_interleave puis prendre chacune des composantes)
  - D: Repeat_interleave E et log_pD
  - T: fusionner le kernel max+matmul.


Etapes : 
1. Implémenter l'ouverture de l'arbre d'espèces et des arbres de gènes en C++. Rendre ça efficace.
   1. Pour l'instant, efficacité ok (0.5 s pour test_mixed_200 avec 200 espèces et 1000 feuilles d'arbre de gènes)
2. Faire en sorte que l'on puisse bien prendre plusieurs arbres de gènes par famille de gènes
   1. Pas encore implémenté cette possibilité.
   2. Pour avoir plusieurs familles, on concatènera les différentes matrices, et on verra comment modifier les updates pour que ça marche malgré tout.
3. Implémenter la descente de gradient efficace pour l'optimisation de la vraisemblance.
   1. Méthode du pseudo second ordre en regardant les gradients des paramètres passés?
   2. BFGS sans line search?
   3. Pour l'instant, LBFGS fonctionne plutôt bien.
   4. Le calcul des gradients est très coûteux vers la fin...
4. Implémenter le stochastic backtracking en Python puis en C++ pour obtenir les scénarios. Faire la partie output.
   1. Stochastic backtracking fonctionne pour un seul arbre de gènes
5. Optimiser la passe backward qui est coûteuse.
6. Il faudrait avoir une optimisation adaptative : 1 ère étape: on calcule le bon gradient. Etapes suivantes : on obtient des gradients approximatifs. Ensuite on termine avec des gradients exacts. On ne doit update la courbure que lorsqu'on calcule des vrais gradients. Vrais gradients = suffisamment d'étapes de vjp. Gradient approximatifs = on update un gradient précédent en faisant juste quelques passes de vjp.



ScatterLogSumExp triton : j'ai essayé d'utiliser torch.func.vjp, mais ça ne fonctionne pas en intégrant cette fonction, car l'autotuner de Triton voit des GradCheckTensor, ce qui est problématique.
J'ai donc utilisé torch.autograd.functional.vjp. Ca fonctionne plutôt bien. Si on veut une intégration avec torch.func, il faut implémenter la fonction fake_tensor comme expliqué dans la page tutoriel intégration à torch.func.