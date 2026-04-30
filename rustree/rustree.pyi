"""Type stubs for the rustree Python module.

rustree provides Python bindings for birth-death species tree simulation,
DTL gene tree simulation, gene tree sampling, reconciliation with ALERax,
and ML/training tensor construction -- all implemented in Rust via PyO3.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

# =============================================================================
# Module-level functions
# =============================================================================

def simulate_species_tree(
    n: int,
    lambda_: float,
    mu: float,
    seed: Optional[int] = None,
) -> PySpeciesTree:
    """Simulate a birth-death species tree.

    Args:
        n: Number of extant species (must be > 0).
        lambda_: Speciation/birth rate (must be > 0).
        mu: Extinction/death rate (must be >= 0 and < lambda_).
        seed: Random seed for reproducibility.

    Returns:
        A PySpeciesTree with *n* extant species simulated under the birth-death
        process.

    Raises:
        ValueError: If parameters are out of range.
    """
    ...

def parse_species_tree(newick_str: str) -> PySpeciesTree:
    """Parse a Newick string or file into a species tree.

    Args:
        newick_str: A Newick formatted string **or** path to a Newick file.

    Returns:
        A PySpeciesTree parsed from the Newick input.

    Raises:
        ValueError: If parsing fails.
    """
    ...

def parse_recphyloxml(filepath: str) -> PyGeneTree:
    """Parse a RecPhyloXML file into a reconciled gene tree.

    Args:
        filepath: Path to the RecPhyloXML file (e.g. from ALERax).

    Returns:
        A PyGeneTree containing both the species tree and reconciled gene tree
        with event mappings.

    Raises:
        ValueError: If parsing fails.
    """
    ...

def reconcile_with_alerax(
    species_tree: Union[PySpeciesTree, str],
    gene_trees: Union[
        PyGeneTree,
        GeneForest,
        List[PyGeneTree],
        str,
        List[str],
        Dict[str, str],
    ],
    output_dir: Optional[str] = None,
    num_samples: int = 100,
    model: str = "PER-FAMILY",
    gene_tree_rooting: Optional[str] = None,
    seed: Optional[int] = None,
    keep_output: bool = False,
    alerax_path: str = "alerax",
) -> Dict[str, PyAleRaxResult]:
    """Reconcile gene trees with a species tree using ALERax.

    Calls the external ALERax tool to perform phylogenetic reconciliation,
    inferring duplication, transfer, and loss events.

    Args:
        species_tree: Species tree as a PySpeciesTree object, Newick string,
            or file path.
        gene_trees: Gene trees as a single PyGeneTree, GeneForest,
            list of PyGeneTree objects, Newick string, file path,
            list of Newick strings/paths, or dict mapping family names to
            Newick strings/paths.
        output_dir: Output directory (default: temporary directory).
        num_samples: Number of reconciliation samples per family.
        model: Model parametrization: ``"PER-FAMILY"`` or ``"GLOBAL"``.
        gene_tree_rooting: Gene tree rooting strategy (optional).
        seed: Random seed for reproducibility.
        keep_output: Whether to preserve ALERax output files.
        alerax_path: Path to the ``alerax`` executable.

    Returns:
        Dictionary mapping family names to PyAleRaxResult objects.

    Raises:
        ValueError: If inputs are invalid or ALERax execution fails.
    """
    ...

def compare_reconciliations(
    truth: PyGeneTree,
    inferred: PyGeneTree,
) -> PyReconciliationComparison:
    """Compare two reconciliations (truth vs inferred).

    Matches nodes by their clade (set of descendant extant leaf names).
    Both gene trees must have the same extant leaf set.

    Args:
        truth: Ground truth reconciliation (e.g. from simulation + sample_extant()).
        inferred: Inferred reconciliation (e.g. from ALERax).

    Returns:
        A PyReconciliationComparison with accuracy metrics and per-node details.

    Raises:
        ValueError: If trees cannot be compared.
    """
    ...

def compare_reconciliations_multi(
    truth: PyGeneTree,
    samples: List[PyGeneTree],
) -> PyMultiSampleComparison:
    """Compare truth reconciliation against multiple inferred samples.

    Computes per-sample metrics and a consensus comparison (majority vote
    per clade).

    Args:
        truth: Ground truth reconciliation.
        samples: List of inferred reconciliations.

    Returns:
        A PyMultiSampleComparison with per-sample and consensus accuracy.

    Raises:
        ValueError: If trees cannot be compared.
    """
    ...

def create_training_sample(
    sp_newick_path: str,
    g_newick_path: str,
    xml_path: str,
) -> dict:
    """Create a training sample from species tree, pruned gene tree, and reconciliation XML.

    Replaces the Python ``create_sample()`` function with a fast Rust
    implementation.  The Zombi-format XML (containing ``<recGeneTree>``
    without ``<spTree>``) is parsed for event annotations.

    Args:
        sp_newick_path: Path to species tree Newick file.
        g_newick_path: Path to pruned gene tree Newick file.
        xml_path: Path to reconciliation XML file (Zombi format).

    Returns:
        A dict with keys: ``species_names``, ``g_root_name``,
        ``_gene_root_name``, ``gene_names``, ``g_neighbors``,
        ``g_leaves_names``, ``sp_children``, ``sp_parents``,
        ``true_states``, ``true_events``, ``true_root``,
        ``nb_sp_leaves``, ``nb_g_leaves``, ``sp_tree_path``,
        ``g_tree_path``, ``xml_g_tree_path``.

    Raises:
        ValueError: If files cannot be read or parsed.
    """
    ...

def create_training_sample_from_sim(
    gene_tree: PyGeneTree,
) -> dict:
    """Create a training sample dict directly from simulated tree objects.

    Equivalent to ``create_training_sample`` but takes a PyGeneTree (after
    ``sample_extant()``) instead of file paths.

    Args:
        gene_tree: A PyGeneTree returned by ``sample_extant()``.

    Returns:
        A dict with the same keys as ``create_training_sample`` (minus the
        three ``*_path`` keys which are set to empty strings).

    Raises:
        ValueError: If gene tree data is incomplete.
    """
    ...

def build_training_tensors(
    base_sample: dict,
    seed: int,
    force_mask_last_added: bool,
    predict_root_position: bool,
    sample_order: str = "random",
) -> dict:
    """Build all tensors needed for a training sample in one Rust call.

    Replaces the Python functions: ``unroot_tree``, ``sample_gene_coloring``,
    ``_build_species_tensors``, ``_build_gene_edges``,
    ``_build_gene_event_types``, and the masking/assembly logic in
    ``get_sample``.

    Args:
        base_sample: A dict as returned by ``create_training_sample`` or
            ``create_training_sample_from_sim``.
        seed: RNG seed for sampling.
        force_mask_last_added: Whether to mask the last added node's event.
        predict_root_position: Whether to predict root position.
        sample_order: ``"random"`` or ``"bottom_up"``.

    Returns:
        A dict of numpy arrays ready to be wrapped into ``HeteroData``.
        Keys include: ``x_sp``, ``x_gene``, ``g_true_sp``, ``event_true``,
        ``event_input``, ``frontier_mask``, ``is_leaf``, ``mask_label_node``,
        ``last_added_index``, ``mask_last_added_event``, ``sp_child_edge``,
        ``sp_parent_edge``, ``g_edge``, ``g_dir_edge``, ``root_edge_target``.

    Raises:
        ValueError: If base_sample is missing required keys.
    """
    ...

def build_otf_batch(
    n_sp: int,
    lambda_birth: float,
    mu_death: float,
    sp_seed: int,
    n_gene_trees: int,
    gt_seeds: List[int],
    lambda_d: float,
    lambda_t: float,
    lambda_l: float,
    enable_event: bool,
    enable_root: bool,
    sample_order: str,
    map_coloring_seeds: List[int],
    evt_coloring_seeds: List[int],
    root_coloring_seeds: List[int],
    min_gene_leaves: int,
    max_gene_nodes: int,
    max_retries: int,
) -> dict:
    """Build a complete batched dict of numpy arrays for on-the-fly training.

    Simulates one species tree and *n_gene_trees* gene trees, builds all
    three task tensors (map / evt / root), collates them, and returns flat
    numpy arrays ready for ``batch_to_heterodata()`` in Python.

    The GIL is released during the Rust computation for true parallelism.

    Args:
        n_sp: Target number of extant species.
        lambda_birth: Speciation rate (must be > mu_death).
        mu_death: Extinction rate (must be >= 0).
        sp_seed: Seed for species tree simulation.
        n_gene_trees: Number of gene trees per batch.
        gt_seeds: Per-gene-tree simulation seeds (length ``n_gene_trees``).
        lambda_d: Duplication rate.
        lambda_t: Transfer rate.
        lambda_l: Loss rate.
        enable_event: Include ``evt_*`` keys in the returned dict.
        enable_root: Include ``root_*`` keys in the returned dict.
        sample_order: ``"random"`` or ``"bottom_up"``.
        map_coloring_seeds: Per-sample RNG seeds for map tensors.
        evt_coloring_seeds: Per-sample RNG seeds for event tensors.
        root_coloring_seeds: Per-sample RNG seeds for root tensors.
        min_gene_leaves: Minimum extant gene leaves required.
        max_gene_nodes: Maximum total gene nodes (0 = no limit).
        max_retries: Retry attempts per gene tree slot on simulation failure.

    Returns:
        A dict of numpy arrays for species and gene tensor data.

    Raises:
        ValueError: If inputs are invalid or simulation fails.
    """
    ...

def compute_gcn_norm(
    edge_index: np.ndarray,
    num_nodes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GCN normalization on edge indices.

    Adds self-loops and computes D^{-1/2} A D^{-1/2} edge weights.
    Matches PyG's ``gcn_norm(edge_index, None, num_nodes, improved=False,
    add_self_loops=True, flow="source_to_target")``.

    Args:
        edge_index: numpy array of shape ``(2, E)``, dtype ``int32``.
        num_nodes: Number of nodes in the graph.

    Returns:
        Tuple of ``(edge_index_with_self_loops, edge_weights)`` as numpy
        arrays.

    Raises:
        ValueError: If edge_index shape is invalid.
    """
    ...

def build_inference_batch(
    base_sample: dict,
    n_copies: int,
    seed: int,
    sample_order: str,
) -> dict:
    """Build a batched dict of numpy arrays for inference from a base_sample dict.

    Replaces the Python ``get_sample()`` + ``collate()`` path.  Builds task
    tensors, collates *n_copies*, applies GCN norm and computes varlen
    metadata.  Only mapping task tensors are returned.

    Args:
        base_sample: A dict as returned by ``create_training_sample``.
        n_copies: Number of copies to collate into the batch.
        seed: RNG seed.
        sample_order: ``"random"`` or ``"bottom_up"``.

    Returns:
        A dict of numpy arrays in the same format as ``build_otf_batch()``
        (mapping task only).

    Raises:
        ValueError: If base_sample is missing required keys.
    """
    ...

def from_reconciliation(
    sp_newick: str,
    g_newick: str,
    node_species: Dict[str, str],
    node_events: Dict[str, int],
) -> PyGeneTree:
    """Build a PyGeneTree from Newick strings and per-node reconciliation annotations.

    Allows constructing a fully annotated reconciled gene tree without going
    through RecPhyloXML, e.g. from model predictions.

    Args:
        sp_newick: Species tree in Newick format (string, not file path).
        g_newick: Gene tree in Newick format (string, not file path).
        node_species: Dict mapping gene node name to species node name.
        node_events: Dict mapping gene node name to event code
            (0=Speciation, 1=Duplication, 2=Transfer, 3=Leaf).

    Returns:
        A PyGeneTree with the given reconciliation annotations.

    Raises:
        ValueError: If parsing fails or species names do not match.
    """
    ...

# =============================================================================
# Classes
# =============================================================================

class PySpeciesTree:
    """A species tree simulated under the birth-death process."""

    def to_newick(self) -> str:
        """Convert the species tree to Newick format.

        Raises:
            ValueError: If conversion fails.
        """
        ...

    def num_nodes(self) -> int:
        """Get the number of nodes in the tree."""
        ...

    def num_leaves(self) -> int:
        """Get the number of extant species (leaves)."""
        ...

    def tree_height(self) -> float:
        """Get the total tree height (depth of deepest leaf)."""
        ...

    def root_index(self) -> int:
        """Get the root node index."""
        ...

    def leaf_names(self) -> List[str]:
        """Get leaf names."""
        ...

    def get_node(self, index: int) -> PySpeciesNode:
        """Get a node by its index.

        Raises:
            ValueError: If index is out of range.
        """
        ...

    def iter(self, order: str = "preorder") -> PySpeciesTreeIter:
        """Iterate over nodes in a given traversal order.

        Args:
            order: Traversal order: ``"preorder"``, ``"inorder"``, or
                ``"postorder"``.

        Raises:
            ValueError: If order is not recognized.
        """
        ...

    def __iter__(self) -> PySpeciesTreeIter:
        """Default iteration (preorder traversal)."""
        ...

    def sample_leaf_names(
        self,
        n: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Uniformly sample leaf names from extant species in the tree.

        Exactly one of *n* or *fraction* must be provided.

        Args:
            n: Number of leaf names to sample.
            fraction: Fraction of leaves to sample (0.0--1.0).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If arguments are invalid.
        """
        ...

    def extract_induced_subtree_by_names(self, names: List[str]) -> PySpeciesTree:
        """Extract an induced subtree keeping only the specified leaf names.

        Creates a new species tree containing only the specified leaves and
        their most recent common ancestors (MRCAs).

        Args:
            names: List of leaf names to keep in the subtree.

        Raises:
            ValueError: If names list is empty or no matching leaves found.
        """
        ...

    def sample_extant(self) -> PySpeciesTree:
        """Extract only extant species from the tree (remove extinct lineages).

        Only works on trees from ``simulate_species_tree()``.

        Raises:
            ValueError: If no extant species are found.
        """
        ...

    def save_newick(self, filepath: str) -> None:
        """Save the species tree to a Newick file.

        Raises:
            ValueError: If writing fails.
        """
        ...

    def to_svg(
        self,
        filepath: Optional[str] = None,
        open_browser: bool = False,
        internal_names: bool = True,
        color: Optional[str] = None,
        fontsize: Optional[float] = None,
        thickness: Optional[float] = None,
        symbol_size: Optional[float] = None,
        background: Optional[str] = None,
        landscape: bool = False,
        sampled_species_names: Optional[List[str]] = None,
        keep_color: str = "green",
        has_descendant_color: str = "orange",
        discard_color: str = "grey",
    ) -> str:
        """Generate an SVG visualization of the species tree using thirdkind.

        Requires ``thirdkind`` to be installed (``cargo install thirdkind``).

        Args:
            filepath: Path to save the SVG file. If None, only returns the
                SVG as a string.
            open_browser: Open the SVG in a web browser.
            internal_names: Display internal node names.
            color: Color for the tree (e.g. ``"blue"``, ``"#4A38C4"``).
            fontsize: Font size for node labels.
            thickness: Thickness of tree lines.
            symbol_size: Size of event symbols.
            background: Background color.
            landscape: Display in landscape orientation.
            sampled_species_names: Species leaf names to keep; when provided,
                node labels are colored by their NodeMark.
            keep_color: Color for Keep nodes.
            has_descendant_color: Color for HasDescendant nodes.
            discard_color: Color for Discard nodes.

        Returns:
            The SVG content as a string.

        Raises:
            ValueError: If thirdkind is not installed or fails.
        """
        ...

    def display(
        self,
        internal_names: bool = True,
        color: Optional[str] = None,
        fontsize: Optional[float] = None,
        thickness: Optional[float] = None,
        symbol_size: Optional[float] = None,
        background: Optional[str] = None,
        landscape: bool = False,
        sampled_species_names: Optional[List[str]] = None,
        keep_color: str = "green",
        has_descendant_color: str = "orange",
        discard_color: str = "grey",
    ) -> object:
        """Display the species tree visualization in a Jupyter notebook.

        Requires ``thirdkind`` and IPython/Jupyter.  Accepts the same styling
        options as ``to_svg()``.

        Returns:
            An ``IPython.display.SVG`` object.

        Raises:
            ValueError: If thirdkind is not installed or fails.
        """
        ...

    def save_bd_events_csv(
        self,
        filepath: str,
        eps: Optional[float] = None,
    ) -> None:
        """Save birth-death events to a CSV file.

        Raises:
            ValueError: If event generation or writing fails.
        """
        ...

    def get_bd_events(
        self,
        eps: Optional[float] = None,
    ) -> dict:
        """Get birth-death events as a dictionary.

        Returns:
            A dict with keys ``time``, ``node_name``, ``event_type``,
            ``child1_name``, ``child2_name``.

        Raises:
            ValueError: If event generation fails.
        """
        ...

    def get_ltt_data(
        self,
        eps: Optional[float] = None,
    ) -> dict:
        """Get Lineages Through Time (LTT) data for plotting.

        Returns:
            A dict with keys ``times`` and ``lineages``.

        Raises:
            ValueError: If event generation fails.
        """
        ...

    def plot_ltt(
        self,
        filepath: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ) -> None:
        """Plot Lineages Through Time (LTT) using matplotlib.

        Args:
            filepath: Path to save the plot. If None, displays interactively.
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.

        Raises:
            ValueError: If plotting fails.
        """
        ...

    def simulate_dtl(
        self,
        lambda_d: float,
        lambda_t: float,
        lambda_l: float,
        transfer_alpha: Optional[float] = None,
        replacement_transfer: Optional[float] = None,
        require_extant: bool = False,
        seed: Optional[int] = None,
    ) -> PyGeneTree:
        """Simulate a gene tree along this species tree using the DTL model.

        Args:
            lambda_d: Duplication rate per unit time.
            lambda_t: Transfer rate per unit time.
            lambda_l: Loss rate per unit time.
            transfer_alpha: Distance decay for assortative transfers
                (None = uniform).
            replacement_transfer: Probability of replacement transfer
                (0.0--1.0).
            require_extant: If True, retry until a tree with extant genes
                is produced.
            seed: Random seed for reproducibility.

        Returns:
            A PyGeneTree containing the simulated gene tree.

        Raises:
            ValueError: If rates are invalid or simulation fails.
        """
        ...

    def simulate_dtl_batch(
        self,
        n: int,
        lambda_d: float,
        lambda_t: float,
        lambda_l: float,
        transfer_alpha: Optional[float] = None,
        replacement_transfer: Optional[float] = None,
        require_extant: bool = False,
        seed: Optional[int] = None,
    ) -> GeneForest:
        """Simulate multiple gene trees efficiently with shared pre-computed data.

        Faster than calling ``simulate_dtl`` multiple times.

        Args:
            n: Number of gene trees to simulate.
            lambda_d: Duplication rate per unit time.
            lambda_t: Transfer rate per unit time.
            lambda_l: Loss rate per unit time.
            transfer_alpha: Distance decay for assortative transfers.
            replacement_transfer: Probability of replacement transfer.
            require_extant: If True, only include trees with extant genes.
            seed: Random seed for reproducibility.

        Returns:
            A GeneForest containing the simulated gene trees.

        Raises:
            ValueError: If rates are invalid or simulation fails.
        """
        ...

    def simulate_dtl_per_species(
        self,
        lambda_d: float,
        lambda_t: float,
        lambda_l: float,
        transfer_alpha: Optional[float] = None,
        replacement_transfer: Optional[float] = None,
        require_extant: bool = False,
        seed: Optional[int] = None,
    ) -> PyGeneTree:
        """Simulate a gene tree using the Zombi-style per-species DTL model.

        In this model, the event rate is proportional to the number of
        *species* with gene copies, NOT the number of gene copies.

        Args:
            lambda_d: Duplication rate per species per unit time.
            lambda_t: Transfer rate per species per unit time.
            lambda_l: Loss rate per species per unit time.
            transfer_alpha: Distance decay for assortative transfers.
            replacement_transfer: Probability of replacement transfer.
            require_extant: If True, retry until at least one extant gene
                exists.
            seed: Random seed for reproducibility.

        Returns:
            A PyGeneTree containing the simulated gene tree.

        Raises:
            ValueError: If rates are invalid or simulation fails.
        """
        ...

    def simulate_dtl_per_species_batch(
        self,
        n: int,
        lambda_d: float,
        lambda_t: float,
        lambda_l: float,
        transfer_alpha: Optional[float] = None,
        replacement_transfer: Optional[float] = None,
        require_extant: bool = False,
        seed: Optional[int] = None,
    ) -> GeneForest:
        """Simulate multiple gene trees using the Zombi-style per-species DTL model.

        Faster than calling ``simulate_dtl_per_species`` multiple times.

        Args:
            n: Number of gene trees to simulate.
            lambda_d: Duplication rate per species per unit time.
            lambda_t: Transfer rate per species per unit time.
            lambda_l: Loss rate per species per unit time.
            transfer_alpha: Distance decay for assortative transfers.
            replacement_transfer: Probability of replacement transfer.
            require_extant: If True, only include trees with extant genes.
            seed: Random seed for reproducibility.

        Returns:
            A GeneForest containing the simulated gene trees.

        Raises:
            ValueError: If rates are invalid or simulation fails.
        """
        ...

    def simulate_dtl_iter(
        self,
        lambda_d: float,
        lambda_t: float,
        lambda_l: float,
        transfer_alpha: Optional[float] = None,
        replacement_transfer: Optional[float] = None,
        n: int = 1,
        require_extant: bool = False,
        seed: Optional[int] = None,
    ) -> PyDtlSimIter:
        """Create a lazy iterator that generates gene trees one at a time (per-gene-copy model).

        Each call to ``next()`` runs one Gillespie simulation and yields a
        PyGeneTree.  Only one gene tree lives in memory at a time.

        Args:
            lambda_d: Duplication rate per unit time.
            lambda_t: Transfer rate per unit time.
            lambda_l: Loss rate per unit time.
            transfer_alpha: Distance decay for assortative transfers.
            replacement_transfer: Probability of replacement transfer.
            n: Number of gene trees to generate.
            require_extant: If True, retry until a tree with extant genes
                is produced.
            seed: Random seed for reproducibility.

        Returns:
            A PyDtlSimIter that implements the Python iterator protocol.

        Raises:
            ValueError: If rates are invalid.
        """
        ...

    def simulate_dtl_per_species_iter(
        self,
        lambda_d: float,
        lambda_t: float,
        lambda_l: float,
        transfer_alpha: Optional[float] = None,
        replacement_transfer: Optional[float] = None,
        n: int = 1,
        require_extant: bool = False,
        seed: Optional[int] = None,
    ) -> PyDtlSimIter:
        """Create a lazy iterator that generates gene trees one at a time (per-species model).

        In the per-species (Zombi-style) model, the DTL event rate scales
        with the number of alive species, NOT the number of gene copies.

        Args:
            lambda_d: Duplication rate per species per unit time.
            lambda_t: Transfer rate per species per unit time.
            lambda_l: Loss rate per species per unit time.
            transfer_alpha: Distance decay for assortative transfers.
            replacement_transfer: Probability of replacement transfer.
            n: Number of gene trees to generate.
            require_extant: If True, retry until a tree with extant genes
                is produced.
            seed: Random seed for reproducibility.

        Returns:
            A PyDtlSimIter that implements the Python iterator protocol.

        Raises:
            ValueError: If rates are invalid.
        """
        ...

    def pairwise_distances(
        self,
        distance_type: str,
        leaves_only: bool = True,
    ) -> pd.DataFrame:
        """Compute all pairwise distances between nodes in the species tree.

        Args:
            distance_type: ``"topological"`` (number of edges) or ``"metric"``
                (sum of branch lengths).
            leaves_only: If True, only compute distances between leaf nodes.

        Returns:
            A pandas DataFrame with columns: ``node1``, ``node2``,
            ``distance``.

        Raises:
            ValueError: If distance_type is invalid.
        """
        ...

    def save_pairwise_distances_csv(
        self,
        filepath: str,
        distance_type: str,
        leaves_only: bool = True,
    ) -> None:
        """Save pairwise distances between nodes to a CSV file.

        Args:
            filepath: Path to save the CSV file.
            distance_type: ``"topological"`` or ``"metric"``.
            leaves_only: If True, only compute distances between leaf nodes.

        Raises:
            ValueError: If distance_type is invalid or writing fails.
        """
        ...

    def compute_ghost_lengths(
        self,
        sampled_leaf_names: List[str],
    ) -> pd.DataFrame:
        """Compute ghost branch lengths for a sampled species tree.

        The ghost length of a sampled-tree branch is the sum of branch
        lengths of all non-sampled nodes in the complete tree that project
        onto that branch.

        Args:
            sampled_leaf_names: Names of species leaves to keep.

        Returns:
            A pandas DataFrame with columns: ``node_index``, ``node_name``,
            ``ghost_length``.
        """
        ...

class PySpeciesNode:
    """A single node from a species tree."""

    @property
    def name(self) -> str:
        """Node name."""
        ...

    @property
    def index(self) -> int:
        """Node index in the tree."""
        ...

    @property
    def depth(self) -> Optional[float]:
        """Node depth (distance from root)."""
        ...

    @property
    def length(self) -> float:
        """Branch length to parent."""
        ...

    @property
    def left_child(self) -> Optional[int]:
        """Left child index, or None for leaves."""
        ...

    @property
    def right_child(self) -> Optional[int]:
        """Right child index, or None for leaves."""
        ...

    @property
    def parent(self) -> Optional[int]:
        """Parent index, or None for root."""
        ...

    @property
    def bd_event(self) -> Optional[str]:
        """Birth-death event type, or None for parsed trees."""
        ...

class PySpeciesTreeIter:
    """Iterator over species tree nodes in a given traversal order."""

    def __iter__(self) -> PySpeciesTreeIter: ...
    def __next__(self) -> PySpeciesNode: ...

class PyGeneTree:
    """A gene tree simulated under the DTL model (or parsed from file).

    Contains the reconciled gene tree with species mapping, event annotations,
    and a reference to the associated species tree.
    """

    def to_newick(self) -> str:
        """Convert the gene tree to Newick format.

        Raises:
            ValueError: If conversion fails.
        """
        ...

    def save_newick(self, filepath: str) -> None:
        """Save the gene tree to a Newick file.

        Raises:
            ValueError: If writing fails.
        """
        ...

    def num_nodes(self) -> int:
        """Get the number of nodes in the gene tree."""
        ...

    def num_extant(self) -> int:
        """Get the number of extant genes (leaves that survived, not losses)."""
        ...

    def count_events(self) -> Dict[str, int]:
        """Get the number of events by type.

        Returns:
            Dict with keys: ``speciations``, ``duplications``, ``transfers``,
            ``losses``, ``leaves``.
        """
        ...

    def extant_gene_names(self) -> List[str]:
        """Get names of extant genes (genes that survived to present)."""
        ...

    def sample_extant(self) -> PyGeneTree:
        """Sample the gene tree by keeping only genes from extant species.

        Returns a new gene tree containing only the induced subtree of
        extant genes.

        Raises:
            ValueError: If no extant genes to sample.
        """
        ...

    def sample_by_names(self, names: List[str]) -> PyGeneTree:
        """Sample the gene tree by keeping only genes with the specified names.

        Args:
            names: List of gene names to keep.

        Raises:
            ValueError: If no matching genes found.
        """
        ...

    def sample_by_species_names(self, species_names: List[str]) -> PyGeneTree:
        """Sample the gene tree by keeping only genes from the specified species.

        Gene leaves are matched to species by the naming convention
        ``speciesName_geneId``.

        Args:
            species_names: List of species names to keep.

        Raises:
            ValueError: If no genes found for the specified species.
        """
        ...

    def to_xml(self) -> str:
        """Export the reconciled tree to RecPhyloXML format as a string."""
        ...

    def save_xml(self, filepath: str) -> None:
        """Save the reconciled tree to a RecPhyloXML file.

        Raises:
            ValueError: If writing fails.
        """
        ...

    def to_svg(
        self,
        filepath: Optional[str] = None,
        open_browser: bool = False,
        gene_colors: Optional[str] = None,
        species_color: Optional[str] = None,
        internal_gene_names: bool = False,
        internal_species_names: bool = False,
        gene_fontsize: Optional[float] = None,
        species_fontsize: Optional[float] = None,
        gene_thickness: Optional[float] = None,
        species_thickness: Optional[float] = None,
        symbol_size: Optional[float] = None,
        background: Optional[str] = None,
        landscape: bool = False,
        fill_species: bool = False,
        gene_only: bool = False,
        sampled_species_names: Optional[List[str]] = None,
        keep_color: str = "green",
        has_descendant_color: str = "orange",
        discard_color: str = "grey",
        color_transfers_by: Optional[str] = None,
    ) -> str:
        """Generate an SVG visualization of the reconciled tree using thirdkind.

        Requires ``thirdkind`` to be installed (``cargo install thirdkind``).

        Args:
            filepath: Path to save the SVG file.
            open_browser: Open the SVG in a web browser.
            gene_colors: Comma-separated colors for gene trees.
            species_color: Color for the species tree.
            internal_gene_names: Display internal gene node names.
            internal_species_names: Display internal species node names.
            gene_fontsize: Font size for gene tree labels.
            species_fontsize: Font size for species tree labels.
            gene_thickness: Thickness of gene tree lines.
            species_thickness: Thickness of species tree lines.
            symbol_size: Size of event symbols.
            background: Background color.
            landscape: Display in landscape orientation.
            fill_species: Fill the species tree.
            gene_only: Show only the gene tree (Newick mode).
            sampled_species_names: Species leaf names for NodeMark coloring.
            keep_color: Color for Keep nodes.
            has_descendant_color: Color for HasDescendant nodes.
            discard_color: Color for Discard nodes.
            color_transfers_by: ``"donor"`` or ``"recipient"`` to color
                transfers by NodeMark.

        Returns:
            The SVG content as a string.

        Raises:
            ValueError: If thirdkind is not installed or fails.
        """
        ...

    def display(
        self,
        gene_colors: Optional[str] = None,
        species_color: Optional[str] = None,
        internal_gene_names: bool = False,
        internal_species_names: bool = False,
        gene_fontsize: Optional[float] = None,
        species_fontsize: Optional[float] = None,
        gene_thickness: Optional[float] = None,
        species_thickness: Optional[float] = None,
        symbol_size: Optional[float] = None,
        background: Optional[str] = None,
        landscape: bool = False,
        fill_species: bool = False,
        gene_only: bool = False,
        sampled_species_names: Optional[List[str]] = None,
        keep_color: str = "green",
        has_descendant_color: str = "orange",
        discard_color: str = "grey",
        color_transfers_by: Optional[str] = None,
    ) -> object:
        """Display the reconciled tree visualization in a Jupyter notebook.

        Requires ``thirdkind`` and IPython/Jupyter.  Accepts the same
        styling options as ``to_svg()``.

        Returns:
            An ``IPython.display.SVG`` object.

        Raises:
            ValueError: If thirdkind is not installed or fails.
        """
        ...

    def to_csv(
        self,
        filepath: Optional[str] = None,
    ) -> pd.DataFrame:
        """Export gene tree data as a pandas DataFrame.

        Columns: ``node_id``, ``name``, ``parent``, ``left_child``,
        ``left_child_name``, ``right_child``, ``right_child_name``,
        ``length``, ``depth``, ``species_node``, ``species_node_left``,
        ``species_node_right``, ``event``.

        Args:
            filepath: Optional path to save as CSV file.

        Returns:
            A pandas DataFrame with the gene tree data.

        Raises:
            ValueError: If writing fails.
        """
        ...

    def transfers(self) -> pd.DataFrame:
        """Return a DataFrame of transfer events."""
        ...

    def duplications(self) -> pd.DataFrame:
        """Return a DataFrame of duplication events."""
        ...

    def losses(self) -> pd.DataFrame:
        """Return a DataFrame of loss events."""
        ...

    def speciations(self) -> pd.DataFrame:
        """Return a DataFrame of speciation events."""
        ...

    def leaves(self) -> pd.DataFrame:
        """Return a DataFrame of leaf events."""
        ...

    def pairwise_distances(
        self,
        distance_type: str,
        leaves_only: bool = True,
    ) -> pd.DataFrame:
        """Compute all pairwise distances between nodes in the gene tree.

        Args:
            distance_type: ``"topological"`` or ``"metric"``.
            leaves_only: If True, only compute distances between leaf nodes.

        Returns:
            A pandas DataFrame with columns: ``node1``, ``node2``,
            ``distance``.

        Raises:
            ValueError: If distance_type is invalid.
        """
        ...

    def save_pairwise_distances_csv(
        self,
        filepath: str,
        distance_type: str,
        leaves_only: bool = True,
    ) -> None:
        """Save pairwise distances between nodes in the gene tree to a CSV file.

        Args:
            filepath: Path to save the CSV file.
            distance_type: ``"topological"`` or ``"metric"``.
            leaves_only: If True, only compute distances between leaf nodes.

        Raises:
            ValueError: If distance_type is invalid or writing fails.
        """
        ...

    def compute_induced_transfers(
        self,
        sampled_leaf_names: List[str],
    ) -> pd.DataFrame:
        """Compute induced transfers by projecting transfers onto a sampled species tree.

        Args:
            sampled_leaf_names: Names of species leaves to keep.

        Returns:
            A pandas DataFrame with columns: ``time``, ``gene_id``,
            ``from_species_complete``, ``to_species_complete``,
            ``from_species_sampled``, ``to_species_sampled``.

        Raises:
            ValueError: If DTL events are not available.
        """
        ...

    def compare_reconciliation(
        self,
        other: PyGeneTree,
    ) -> PyReconciliationComparison:
        """Compare this reconciliation (truth) against another (inferred).

        Args:
            other: The inferred reconciliation to compare against.

        Returns:
            A PyReconciliationComparison with accuracy metrics.

        Raises:
            ValueError: If trees cannot be compared.
        """
        ...

    def compare_reconciliation_multi(
        self,
        samples: List[PyGeneTree],
    ) -> PyMultiSampleComparison:
        """Compare this reconciliation (truth) against multiple inferred samples.

        Args:
            samples: List of inferred reconciliations.

        Returns:
            A PyMultiSampleComparison with per-sample and consensus accuracy.

        Raises:
            ValueError: If trees cannot be compared.
        """
        ...

    def rf_distance(
        self,
        other: PyGeneTree,
        rooted: bool = False,
    ) -> int:
        """Compute Robinson-Foulds distance to another gene tree.

        Loss nodes are automatically stripped before comparison.

        Args:
            other: The other gene tree.
            rooted: If True, compare rooted clades. If False (default),
                compare unrooted bipartitions.

        Returns:
            The RF distance (number of differing splits).

        Raises:
            ValueError: If comparison fails.
        """
        ...

class PyDtlSimIter:
    """Lazy iterator for DTL gene tree simulation.

    Generates one gene tree per ``next()`` call using the Gillespie algorithm.
    Created by ``PySpeciesTree.simulate_dtl_iter()`` or
    ``PySpeciesTree.simulate_dtl_per_species_iter()``.
    """

    @property
    def remaining(self) -> int:
        """Number of simulations remaining."""
        ...

    @property
    def total(self) -> int:
        """Total number of simulations requested."""
        ...

    @property
    def completed(self) -> int:
        """Number of simulations completed so far."""
        ...

    def __iter__(self) -> PyDtlSimIter: ...

    def __next__(self) -> PyGeneTree:
        """Generate the next gene tree.

        Raises:
            ValueError: If the simulation fails.
            StopIteration: When all requested simulations are done.
        """
        ...

    def __len__(self) -> int: ...

    def single(self) -> PyGeneTree:
        """Run a single simulation and return the result directly.

        Raises:
            ValueError: If no simulations remain or the simulation fails.
        """
        ...

    def collect_all(self) -> GeneForest:
        """Collect all remaining simulations into a GeneForest.

        Warning: this loads all trees into memory at once.
        """
        ...

    def save_xml(self, dir: str) -> None:
        """Save each remaining gene tree as RecPhyloXML to the given directory.

        Files are named ``gene_0000.xml``, ``gene_0001.xml``, etc.

        Raises:
            ValueError: If writing fails.
        """
        ...

    def save_newick(self, dir: str) -> None:
        """Save each remaining gene tree as Newick to the given directory.

        Files are named ``gene_0000.nwk``, ``gene_0001.nwk``, etc.
        Newick format does not preserve reconciliation information.

        Raises:
            ValueError: If writing fails.
        """
        ...

class GeneForest:
    """A collection of gene trees associated with a single species tree.

    Provides methods for pruning by species tree or species leaf names,
    reconciliation with ALERax, and batch access to gene trees.
    """

    def __init__(
        self,
        species_tree: PySpeciesTree,
        gene_trees: List[PyGeneTree],
    ) -> None:
        """Create a new GeneForest from a species tree and list of gene trees."""
        ...

    def __len__(self) -> int:
        """Number of gene trees in the forest."""
        ...

    def __getitem__(self, idx: int) -> PyGeneTree:
        """Get a gene tree by index.

        Raises:
            ValueError: If index is out of range.
        """
        ...

    @property
    def species_tree(self) -> PySpeciesTree:
        """Get the species tree."""
        ...

    @property
    def gene_trees(self) -> List[PyGeneTree]:
        """Get all gene trees."""
        ...

    def sample_extant(self) -> GeneForest:
        """Return a new forest with each gene tree pruned to extant leaves only.

        Gene trees with zero extant leaves are dropped.

        Raises:
            ValueError: If sampling fails.
        """
        ...

    def prune_to_species_tree(self, target: PySpeciesTree) -> GeneForest:
        """Prune the forest to match a target species tree.

        The target species tree's leaves must be a subset of this forest's
        species tree leaves.

        Raises:
            ValueError: If pruning fails.
        """
        ...

    def sample_leaves(self, names: List[str]) -> GeneForest:
        """Sample leaves and filter all trees accordingly.

        Args:
            names: List of species leaf names to keep.

        Raises:
            ValueError: If sampling fails.
        """
        ...

    def reconcile_with_alerax(
        self,
        output_dir: Optional[str] = None,
        num_samples: int = 100,
        model: str = "PER-FAMILY",
        gene_tree_rooting: Optional[str] = None,
        seed: Optional[int] = None,
        keep_output: bool = False,
        alerax_path: str = "alerax",
    ) -> AleRaxForestResult:
        """Reconcile the gene forest with ALERax.

        Runs ALERax on all gene trees, parses all results, and auto-renames
        species tree nodes back to their original names.

        Args:
            output_dir: Output directory (default: temp directory).
            num_samples: Number of reconciliation samples per family.
            model: ``"PER-FAMILY"`` or ``"GLOBAL"``.
            gene_tree_rooting: Gene tree rooting strategy.
            seed: Random seed for reproducibility.
            keep_output: Preserve ALERax output files.
            alerax_path: Path to ``alerax`` executable.

        Returns:
            An AleRaxForestResult containing all reconciliation data.

        Raises:
            ValueError: If reconciliation fails.
        """
        ...

    def __iter__(self) -> Iterator[PyGeneTree]: ...

class PyAleRaxResult:
    """Result of ALERax reconciliation for a gene family.

    Contains all reconciliation samples, estimated evolutionary rates,
    log-likelihood, and summary statistics.
    """

    @property
    def gene_trees(self) -> List[PyGeneTree]:
        """All reconciliation samples (typically 100)."""
        ...

    @property
    def duplication_rate(self) -> float:
        """Estimated duplication rate."""
        ...

    @property
    def loss_rate(self) -> float:
        """Estimated loss rate."""
        ...

    @property
    def transfer_rate(self) -> float:
        """Estimated transfer rate."""
        ...

    @property
    def likelihood(self) -> float:
        """Log-likelihood of the reconciliation."""
        ...

    @property
    def statistics(self) -> PyReconciliationStatistics:
        """Summary statistics across all reconciliation samples."""
        ...

class AleRaxForestResult:
    """Comprehensive result of reconciling a GeneForest with ALERax.

    Contains per-family results, aggregate species event counts, and
    total transfers.
    """

    @property
    def family_results(self) -> Dict[str, PyAleRaxResult]:
        """Per-family reconciliation results."""
        ...

    @property
    def output_dir(self) -> Optional[str]:
        """Output directory path, if preserved."""
        ...

    def mean_species_event_counts(
        self,
        family_name: str,
    ) -> pd.DataFrame:
        """Get mean species event counts for a specific family as a pandas DataFrame.

        Columns: ``species_label``, ``speciations``, ``duplications``,
        ``losses``, ``transfers``, ``presence``, ``origination``, ``copies``,
        ``singletons``, ``transfers_to``.

        Raises:
            ValueError: If family_name is not found.
        """
        ...

    @property
    def total_species_event_counts(self) -> pd.DataFrame:
        """Get total (aggregate) species event counts as a pandas DataFrame.

        Columns: ``species_label``, ``speciations``, ``duplications``,
        ``losses``, ``transfers``, ``presence``, ``origination``, ``copies``,
        ``singletons``, ``transfers_to``.
        """
        ...

    @property
    def total_transfers(self) -> pd.DataFrame:
        """Get total (aggregate) transfers as a pandas DataFrame.

        Columns: ``source``, ``destination``, ``count``.
        """
        ...

    def family_names(self) -> List[str]:
        """List all family names."""
        ...

class PyEventCounts:
    """Event counts from reconciliation analysis."""

    @property
    def speciations(self) -> float:
        """Number of speciation events."""
        ...

    @property
    def speciation_losses(self) -> float:
        """Number of speciation loss events."""
        ...

    @property
    def duplications(self) -> float:
        """Number of duplication events."""
        ...

    @property
    def duplication_losses(self) -> float:
        """Number of duplication loss events."""
        ...

    @property
    def transfers(self) -> float:
        """Number of transfer events."""
        ...

    @property
    def transfer_losses(self) -> float:
        """Number of transfer loss events."""
        ...

    @property
    def losses(self) -> float:
        """Number of loss events."""
        ...

    @property
    def leaves(self) -> float:
        """Number of leaf nodes."""
        ...

class PyReconciliationStatistics:
    """Summary statistics for reconciliation analysis."""

    @property
    def mean_event_counts(self) -> PyEventCounts:
        """Mean event counts across all samples."""
        ...

    @property
    def mean_transfers(self) -> Dict[str, Dict[str, float]]:
        """Mean transfers between species pairs.

        Structure: ``{source_species: {dest_species: mean_count}}``.
        """
        ...

    @property
    def events_per_species(self) -> Dict[str, PyEventCounts]:
        """Mean events per species node.

        Structure: ``{species_name: PyEventCounts}``.
        """
        ...

    def transfers_df(self) -> pd.DataFrame:
        """Returns a pandas DataFrame of mean transfers between species pairs.

        Columns: ``source``, ``destination``, ``mean_count``.
        """
        ...

    def events_df(self) -> pd.DataFrame:
        """Returns a pandas DataFrame of mean events per species.

        Columns: ``species``, ``speciations``, ``speciation_losses``,
        ``duplications``, ``duplication_losses``, ``transfers``,
        ``transfer_losses``, ``losses``, ``leaves``.
        """
        ...

class PyReconciliationComparison:
    """Result of comparing two reconciliations (truth vs inferred).

    Provides accuracy metrics, per-node details, and confusion matrix.
    """

    @property
    def mapping_accuracy(self) -> float:
        """Fraction of correctly inferred species mappings."""
        ...

    @property
    def event_accuracy(self) -> float:
        """Fraction of correctly inferred events."""
        ...

    @property
    def both_accuracy(self) -> float:
        """Fraction of nodes correct on both mapping and event."""
        ...

    @property
    def nodes_compared(self) -> int:
        """Number of nodes matched by clade (topology agreement)."""
        ...

    @property
    def correct_mappings(self) -> int:
        """Number of correct species mappings."""
        ...

    @property
    def correct_events(self) -> int:
        """Number of correct events."""
        ...

    @property
    def mappings_evaluated(self) -> int:
        """Number of nodes where mapping was evaluable."""
        ...

    @property
    def unmatched_truth_clades(self) -> int:
        """Number of truth clades not found in inferred tree."""
        ...

    @property
    def unmatched_inferred_clades(self) -> int:
        """Number of inferred clades not found in truth tree."""
        ...

    @property
    def leaf_check(self) -> Tuple[int, int]:
        """Leaf mapping sanity check: ``(correct, total)``."""
        ...

    def to_dataframe(self) -> pd.DataFrame:
        """Per-node comparison as a pandas DataFrame.

        Columns: ``truth_node_idx``, ``truth_node_name``,
        ``inferred_node_idx``, ``inferred_node_name``, ``clade``,
        ``truth_species``, ``inferred_species``, ``truth_event``,
        ``inferred_event``, ``mapping_correct``, ``event_correct``.
        """
        ...

    def confusion_matrix(self) -> pd.DataFrame:
        """Event confusion matrix as a pandas DataFrame.

        Columns: ``truth_event``, ``inferred_event``, ``count``.
        """
        ...

    def to_svg(
        self,
        truth_color: str = "green",
        inferred_color: str = "red",
        internal_gene_names: bool = False,
        internal_species_names: bool = True,
        landscape: bool = False,
        fill_species: bool = True,
        species_color: str = "#cccccc",
        species_fontsize: float = 40.0,
        background: str = "white",
    ) -> str:
        """Generate SVG showing both reconciliations overlaid on the same species tree.

        Returns:
            The post-processed SVG string.

        Raises:
            ValueError: If trees are not available or thirdkind fails.
        """
        ...

    def display(
        self,
        truth_color: str = "green",
        inferred_color: str = "red",
        internal_gene_names: bool = False,
        internal_species_names: bool = True,
        landscape: bool = False,
        fill_species: bool = True,
        species_color: str = "#cccccc",
        species_fontsize: float = 40.0,
        background: str = "white",
    ) -> object:
        """Display both reconciliations overlaid in a Jupyter notebook.

        Returns:
            An ``IPython.display.SVG`` object.

        Raises:
            ValueError: If trees are not available or thirdkind fails.
        """
        ...

class PyMultiSampleComparison:
    """Result of comparing truth against multiple reconciliation samples.

    Provides per-sample metrics, consensus comparison, and mean accuracies.
    """

    @property
    def mean_mapping_accuracy(self) -> float:
        """Mean mapping accuracy across all samples."""
        ...

    @property
    def mean_event_accuracy(self) -> float:
        """Mean event accuracy across all samples."""
        ...

    @property
    def num_samples(self) -> int:
        """Number of samples compared."""
        ...

    @property
    def consensus(self) -> PyReconciliationComparison:
        """Consensus comparison (majority vote across samples)."""
        ...

    @property
    def sample_mapping_accuracies(self) -> List[float]:
        """Per-sample mapping accuracies."""
        ...

    @property
    def sample_event_accuracies(self) -> List[float]:
        """Per-sample event accuracies."""
        ...

    def get_sample(self, index: int) -> PyReconciliationComparison:
        """Get the comparison for a specific sample index.

        Raises:
            ValueError: If index is out of range.
        """
        ...

# Note: PyInducedTransfer is defined in mod.rs but not registered with
# m.add_class, so it is only returned as part of DataFrame results.
# Including it here for completeness in case it is exposed in the future.

class PyInducedTransfer:
    """An induced transfer: a transfer event projected onto a sampled species tree."""

    @property
    def time(self) -> float:
        """Time of the transfer event."""
        ...

    @property
    def gene_id(self) -> int:
        """Gene tree node index."""
        ...

    @property
    def from_species_complete(self) -> int:
        """Donor species index in the complete tree."""
        ...

    @property
    def to_species_complete(self) -> int:
        """Recipient species index in the complete tree."""
        ...

    @property
    def from_species_sampled(self) -> Optional[int]:
        """Induced donor: index in the sampled tree (None if projection fails)."""
        ...

    @property
    def to_species_sampled(self) -> Optional[int]:
        """Induced recipient: index in the sampled tree (None if projection fails)."""
        ...
