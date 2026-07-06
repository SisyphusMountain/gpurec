import pathlib
import json

FX = pathlib.Path(__file__).parent / "fixtures" / "six"


def test_files_exist():
    for f in ["species.nwk", "true_gene.nwk", "start_gene.nwk", "aln.fasta", "map.link", "families.txt", "generax_ref.json"]:
        assert (FX / f).exists()


def test_start_differs_from_true():
    import dendropy
    tns = dendropy.TaxonNamespace()
    t = dendropy.Tree.get(path=str(FX / "true_gene.nwk"), schema="newick", taxon_namespace=tns)
    s = dendropy.Tree.get(path=str(FX / "start_gene.nwk"), schema="newick", taxon_namespace=tns)
    assert dendropy.calculate.treecompare.symmetric_difference(t, s) >= 2


def test_gpurec_preprocesses_six():
    # gpurec maps species by leaf-name prefix; confirm no mapping error + finite recon logL
    import sys, math, torch
    repo = str(pathlib.Path(__file__).resolve().parents[2])
    sys.path.insert(0, repo)
    from gpurec.api.model import GeneReconModel
    m = GeneReconModel(str(FX / "species.nwk"), [str(FX / "true_gene.nwk")], mode="global", device="cuda", dtype=torch.float64)
    tri = torch.tensor([math.log2(0.2), math.log2(0.3), math.log2(0.1)], dtype=m.theta.dtype, device=m.theta.device)
    with torch.no_grad():
        m.theta.copy_(tri)
        nll = float(m())
    assert math.isfinite(nll)


def test_generax_recovers_better_tree():
    ref = json.loads((FX / "generax_ref.json").read_text())
    # recorded during build: GeneRax output is closer to truth than the start
    assert ref["rf_generax_to_true"] < ref["rf_start_to_true"]
