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
- Finir d'implémenter le log-matmul efficace pour calculer les probabilités de transfert. Pour avoir une bonne efficacité, on peut calculer un max. par ligne de la matrice A et par colonne de la matrice B et faire un matmul comme ça.