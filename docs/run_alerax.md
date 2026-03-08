Example commands to run AleRax. Will generate a folder /output in which you can find model_parameters/model_parameters.txt, where you can find inferred D, T, L rates (there are several lines but all branches have the same parameters)
Command:
```alerax -f families.txt -s sp.nwk -p output --gene-tree-samples 0 --species-tree-search SKIP```