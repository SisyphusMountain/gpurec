// rustree/src/metric_functions.rs
// The functions in this repo act on the branch lengths
// and depths of nodes in phylogenetic trees.


use crate::node::{Node, TraversalOrder, FlatTree};
use std::str::FromStr;

/// Type of distance to compute between nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistanceType {
    /// Topological distance: number of edges between two nodes
    Topological,
    /// Metric distance: sum of branch lengths along the path
    Metric,
}

impl FromStr for DistanceType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "topological" => Ok(DistanceType::Topological),
            "metric" => Ok(DistanceType::Metric),
            _ => Err(format!("Unknown distance type '{}'. Valid values: topological, metric", s)),
        }
    }
}

/// A single pairwise distance entry between two nodes.
#[derive(Clone, Debug)]
pub struct PairwiseDistance<'a> {
    /// Name of the first node
    pub node1: &'a str,
    /// Name of the second node
    pub node2: &'a str,
    /// Distance between the two nodes
    pub distance: f64,
}

impl<'a> PairwiseDistance<'a> {
    /// Convert to CSV row format
    pub fn to_csv_row(&self) -> String {
        format!("{},{},{}", self.node1, self.node2, self.distance)
    }

    /// CSV header for pairwise distance data
    pub fn csv_header() -> &'static str {
        "node1,node2,distance"
    }
}

/// Precomputed structure for O(1) Lowest Common Ancestor (LCA) queries.
///
/// Uses an Euler tour of the tree combined with a sparse table for
/// range minimum queries (RMQ). Preprocessing takes O(n log n) time and space,
/// and each LCA query takes O(1) time.
///
/// This is much faster than the naive `find_lca` method for bulk queries
/// (e.g., `precompute_lca_depths` which performs O(n^2) LCA queries).
pub struct LcaTable {
    /// Euler tour: sequence of node indices visited during DFS
    euler: Vec<usize>,
    /// Tree-level depth of each position in the Euler tour (for RMQ comparison)
    euler_depth: Vec<usize>,
    /// First occurrence of each node index in the Euler tour
    first: Vec<usize>,
    /// Sparse table for range minimum queries on euler_depth.
    /// `sparse[k][i]` = index in euler with minimum depth in `[i, i + 2^k - 1]`
    sparse: Vec<Vec<usize>>,
}

impl LcaTable {
    /// Build an LCA lookup table for the given tree.
    ///
    /// Preprocessing is O(n log n) in time and space where n is the number of nodes.
    ///
    /// # Panics
    ///
    /// Panics if the tree has no nodes.
    pub fn new(tree: &FlatTree) -> Self {
        let n = tree.nodes.len();
        assert!(n > 0, "Cannot build LCA table for empty tree");

        let (euler, euler_depth, first) = Self::euler_tour(tree);
        let sparse = Self::build_sparse_table(&euler_depth);

        LcaTable { euler, euler_depth, first, sparse }
    }

    /// Find the lowest common ancestor of two nodes in O(1).
    ///
    /// # Panics
    ///
    /// Panics if `u` or `v` are not valid node indices for the tree this table was built from.
    pub fn lca(&self, u: usize, v: usize) -> usize {
        if u == v {
            return u;
        }
        let l = self.first[u].min(self.first[v]);
        let r = self.first[u].max(self.first[v]);
        let min_idx = self.rmq(l, r);
        self.euler[min_idx]
    }

    /// Compute the Euler tour of the tree using iterative DFS.
    ///
    /// Returns `(euler_tour, euler_depths, first_occurrence)` where:
    /// - `euler_tour[i]` is the node index at position i in the tour
    /// - `euler_depths[i]` is the tree level (edges from root) at position i
    /// - `first_occurrence[node]` is the first position of `node` in the tour
    ///
    /// The tour has length `2n - 1` for a tree with `n` nodes.
    fn euler_tour(tree: &FlatTree) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let n = tree.nodes.len();
        let tour_len = 2 * n; // 2n-1 entries, allocate 2n for safety

        let mut euler = Vec::with_capacity(tour_len);
        let mut euler_depth = Vec::with_capacity(tour_len);
        let mut first = vec![0usize; n];
        let mut seen = vec![false; n];

        // Stack of (node_idx, tree_level, phase)
        // phase 0: entering node for the first time
        // phase 1: returned from left subtree
        // phase 2: returned from right subtree
        let mut stack: Vec<(usize, usize, u8)> = Vec::with_capacity(tour_len);
        stack.push((tree.root, 0, 0));

        while let Some((node_idx, level, phase)) = stack.pop() {
            let node = &tree.nodes[node_idx];

            match phase {
                0 => {
                    // First visit: record in tour and first-occurrence
                    if !seen[node_idx] {
                        first[node_idx] = euler.len();
                        seen[node_idx] = true;
                    }
                    euler.push(node_idx);
                    euler_depth.push(level);

                    // Schedule after-left continuation, then push left child
                    stack.push((node_idx, level, 1));
                    if let Some(left) = node.left_child {
                        stack.push((left, level + 1, 0));
                    }
                }
                1 => {
                    // Returned from left child: re-add to tour if left child existed
                    if node.left_child.is_some() {
                        euler.push(node_idx);
                        euler_depth.push(level);
                    }

                    // Schedule after-right continuation, then push right child
                    stack.push((node_idx, level, 2));
                    if let Some(right) = node.right_child {
                        stack.push((right, level + 1, 0));
                    }
                }
                2 => {
                    // Returned from right child: re-add to tour if right child existed
                    if node.right_child.is_some() {
                        euler.push(node_idx);
                        euler_depth.push(level);
                    }
                    // Done with this node
                }
                _ => unreachable!()
            }
        }

        (euler, euler_depth, first)
    }

    /// Build a sparse table for range minimum queries on the depth array.
    ///
    /// `sparse[k][i]` = index in `depths` with minimum value in `[i, i + 2^k - 1]`.
    fn build_sparse_table(depths: &[usize]) -> Vec<Vec<usize>> {
        let m = depths.len();
        if m == 0 {
            return vec![];
        }
        if m == 1 {
            return vec![vec![0]];
        }

        let max_k = Self::floor_log2(m) + 1;
        let mut sparse = vec![vec![0usize; m]; max_k];

        // Base case: each element is its own minimum
        for (i, slot) in sparse[0].iter_mut().enumerate() {
            *slot = i;
        }

        // Fill larger ranges by combining two halves
        for k in 1..max_k {
            let half = 1 << (k - 1);
            let range_size = 1usize << k;
            if range_size > m {
                break;
            }
            for i in 0..=(m - range_size) {
                let left = sparse[k - 1][i];
                let right = sparse[k - 1][i + half];
                sparse[k][i] = if depths[left] <= depths[right] { left } else { right };
            }
        }

        sparse
    }

    /// Range minimum query: returns the index with minimum depth in `[l, r]`.
    fn rmq(&self, l: usize, r: usize) -> usize {
        if l == r {
            return l;
        }
        let len = r - l + 1;
        let k = Self::floor_log2(len);
        let left = self.sparse[k][l];
        let right = self.sparse[k][r + 1 - (1 << k)];
        if self.euler_depth[left] <= self.euler_depth[right] { left } else { right }
    }

    /// Compute floor(log2(n)) using bit operations. Panics if n == 0.
    fn floor_log2(n: usize) -> usize {
        assert!(n > 0);
        (usize::BITS - 1 - n.leading_zeros()) as usize
    }
}

impl Node {
    pub fn total_length(&self) -> f64 {
        let mut total_length = 0.0;
        // if add_root_length is False, we want to exclude the first node so we make this False temporarily
        for node in self.iter(TraversalOrder::PreOrder) {
            total_length += node.length;
        }
        total_length
    }
    pub fn zero_root_length(&mut self) {
        self.length = 0.0;
    }
    /// Assigns depths to each node in the tree starting from the current node.
    /// Each node's depth includes its own stem length plus all ancestral stem lengths.
    ///
    /// # Arguments
    /// * `current_depth` - The depth at the parent node (usually 0.0 at the root's parent).
    pub fn assign_depths(&mut self, current_depth: f64) {
        // Include this node's stem length in its depth
        let node_depth = current_depth + self.length;
        self.depth = Some(node_depth);

        if let Some(ref mut left_child) = self.left_child {
            left_child.assign_depths(node_depth);
        }

        if let Some(ref mut right_child) = self.right_child {
            right_child.assign_depths(node_depth);
        }
    }
    /// Updates the lengths of the nodes in a tree based on their depths.
    ///
    /// # Arguments
    /// * `node` - The root node of the tree.
    /// * `parent_depth` - The depth of the parent node.
    ///
    /// # Panics
    ///
    /// Panics if depths have not been assigned (via [`assign_depths`](Self::assign_depths)).
    pub fn depths_to_lengths(&mut self, parent_depth: f64) {
        let depth = self.depth.expect("depths must be assigned before calling depths_to_lengths() - call assign_depths() first");
        self.length = depth - parent_depth;

        if let Some(left_child) = &mut self.left_child {
            left_child.depths_to_lengths(depth);
        }
        if let Some(right_child) = &mut self.right_child {
            right_child.depths_to_lengths(depth);
        }
    }
}

impl FlatTree {
    pub fn total_length(&self) -> f64 {
        let mut total_length = 0.0;
        // if add_root_length is False, we want to exclude the first node so we make this False temporarily
        for node in &self.nodes {
            total_length += node.length;
        }
        total_length
    }
    pub fn zero_root_length(&mut self) {
        self.nodes[self.root].length = 0.0;
    }
    /// Assigns depths to each node in the tree.
    /// Each node's depth includes its own stem length plus all ancestral stem lengths.
    pub fn assign_depths(&mut self) {
        let root_index = self.root;
        // Root depth includes its own stem length
        let root_length = self.nodes[root_index].length;
        self.nodes[root_index].depth = Some(root_length);

        let mut stack = vec![root_index];

        while let Some(node_index) = stack.pop() {
            let current_depth = self.nodes[node_index]
                .depth
                .expect("depths must be assigned before calling this function - call assign_depths() first");

            // Process left child
            if let Some(left_index) = self.nodes[node_index].left_child {
                let left_length = self.nodes[left_index].length;
                let left_depth = current_depth + left_length;
                self.nodes[left_index].depth = Some(left_depth);
                stack.push(left_index);
            }

            // Process right child
            if let Some(right_index) = self.nodes[node_index].right_child {
                let right_length = self.nodes[right_index].length;
                let right_depth = current_depth + right_length;
                self.nodes[right_index].depth = Some(right_depth);
                stack.push(right_index);
            }
        }
    }
    /// Computes the sorted, deduplicated vector of all node depths in the tree.
    ///
    /// This forms the time subdivision used by [`find_contemporaneity`](Self::find_contemporaneity):
    /// each unique depth value becomes a boundary point between consecutive time intervals.
    ///
    /// # Prerequisites
    ///
    /// Depths must be assigned before calling this method (via [`assign_depths`](Self::assign_depths)).
    ///
    /// # Returns
    ///
    /// A sorted `Vec<f64>` of unique depth values. The first element is always `0.0`
    /// (the root depth), and subsequent elements correspond to speciation, extinction, or
    /// leaf events in increasing order of depth.
    ///
    /// # Example
    ///
    /// For the Newick tree `((A:1,B:2)C:1,D:5)R:0`, after assigning depths the node
    /// depths are `{R:0, C:1, A:2, B:3, D:5}`, so this returns `[0, 1, 2, 3, 5]`.
    pub fn make_subdivision(&self) -> Vec<f64> {

        let mut depths: Vec<f64> = self.nodes.iter().filter_map(|node| node.depth).collect();
        depths.sort_by(|a, b| a.total_cmp(b));
        depths.dedup();
        depths
    }
    /// Computes the vector of time interval durations from the tree's depth subdivision.
    ///
    /// Each entry `intervals[i]` is the duration of the `i`-th time interval, defined as
    /// `depths[i] - depths[i-1]`. A leading zero is prepended so that `intervals` has
    /// the same length as the subdivision vector returned by
    /// [`make_subdivision`](Self::make_subdivision).
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` where:
    /// - `intervals[0]` is always `0.0` (there is no interval before the first depth point).
    /// - `intervals[i]` for `i >= 1` equals `depths[i] - depths[i-1]`.
    ///
    /// # Panics
    ///
    /// Panics (unsigned underflow) if the tree has no assigned depths, because the
    /// subdivision will be empty. Call [`assign_depths`](Self::assign_depths) first.
    ///
    /// # Example
    ///
    /// If the depths are `[0, 1, 2, 3, 5]`, the returned intervals are `[0, 1, 1, 1, 2]`.
    pub fn make_intervals(&self) -> Vec<f64> {
        let depths = self.make_subdivision();
        let mut intervals: Vec<f64> = Vec::with_capacity(depths.len());
        intervals.push(0.0);
        for i in 0..depths.len() - 1 {
            intervals.push(depths[i + 1] - depths[i]);
        }
        intervals
    }
    /// Finds the index in `depths` whose value is closest to `value`.
    ///
    /// Uses binary search for efficiency. When `value` falls exactly between two
    /// depth entries, the earlier (lower) index is preferred. If `value` is outside
    /// the range of `depths`, the nearest boundary index is returned.
    ///
    /// # Arguments
    ///
    /// * `depths` - A sorted slice of depth values (as returned by
    ///   [`make_subdivision`](Self::make_subdivision)).
    /// * `value`  - The depth value to look up.
    ///
    /// # Returns
    ///
    /// The index `i` in `depths` such that `depths[i]` is closest to `value`.
    fn find_closest_index(&self, depths: &[f64], value: f64) -> usize {
        match depths.binary_search_by(|probe| probe.total_cmp(&value)) {
            Ok(idx) => idx,
            Err(idx) => {
                if idx == 0 {
                    0
                } else if idx == depths.len() {
                    depths.len() - 1
                } else {
                    // Determine which of depths[idx - 1] or depths[idx] is closer to the value
                    if (value - depths[idx - 1]).abs() < (value - depths[idx]).abs() {
                        idx - 1
                    } else {
                        idx
                    }
                }
            }
        }
    }

    /// Builds a vector of species lists per time interval, describing which lineages
    /// (branches) are alive during each interval of the tree's time subdivision.
    ///
    /// The time subdivision is the sorted, deduplicated vector of node depths returned
    /// by [`make_subdivision`](Self::make_subdivision). Each consecutive pair of depths
    /// defines an interval: `contemporaneity[j]` contains the node indices of branches
    /// alive during the half-open interval `(depths[j-1], depths[j]]`.
    ///
    /// # Interval convention
    ///
    /// A branch (node) with parent depth `p` and own depth `d` is considered alive in
    /// the slots `(start_index + 1)..=end_index`, where `start_index` and `end_index`
    /// are the closest subdivision indices to `p` and `d` respectively. In other words,
    /// the branch is alive **strictly after** its parent's speciation event up to and
    /// **including** its own event time.
    ///
    /// # Important edge cases
    ///
    /// - **`contemporaneity[0]` is always empty**: there are no events before the root
    ///   depth, so no branch can be alive in the first slot.
    /// - **The root node produces an empty range**: since the root's branch length is
    ///   typically `0.0`, its start and end indices coincide, making
    ///   `(start_index + 1)..=end_index` empty. This is intentional --- the root is a
    ///   speciation point, not a lineage that persists over time.
    /// - **Very short branches**: if a branch is so short that its start and end depths
    ///   round to the same subdivision slot, the range `(start_index + 1)..=end_index`
    ///   is empty and that branch appears in zero intervals.
    ///
    /// # Prerequisites
    ///
    /// Depths must be assigned before calling this method (via
    /// [`assign_depths`](Self::assign_depths)).
    ///
    /// # Arguments
    ///
    /// * `depths` - A sorted slice of time points (depths) representing the subdivision,
    ///   as returned by [`make_subdivision`](Self::make_subdivision).
    ///
    /// # Panics
    ///
    /// Panics if depths have not been assigned (via [`assign_depths`](Self::assign_depths)).
    ///
    /// # Returns
    ///
    /// A `Vec<Vec<usize>>` of the same length as `depths`, where `contemporaneity[j]`
    /// contains the indices of nodes whose branches span the interval ending at
    /// `depths[j]`.
    pub fn find_contemporaneity(&self, depths: &[f64]) -> Vec<Vec<usize>> {
        let mut contemporaneity: Vec<Vec<usize>> = vec![Vec::new(); depths.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            let node_depth = node.depth.expect("depths must be assigned before calling find_contemporaneity() - call assign_depths() first");
            let start_time = node_depth - node.length;
            let end_time = node_depth;

            // Find the indices in the subdivision closest to the node's start and end times.
            let start_index = self.find_closest_index(depths, start_time);
            let end_index = self.find_closest_index(depths, end_time);

            // We don't count the start index because the node is not alive on the interval that ends at the start index.
            for slot in &mut contemporaneity[(start_index + 1)..=end_index] {
                slot.push(i);
            }
        }
        contemporaneity
    }

    /// Computes the number of species over each time interval from the contemporaneity vector.
    ///
    /// # Arguments
    /// * `contemporaneity` - A vector of vectors containing indices of contemporaneous species over intervals.
    ///
    /// # Returns
    /// A vector containing the number of species over each interval.
    pub fn number_of_species(&self, contemporaneity: &[Vec<usize>]) -> Vec<f64> {
        contemporaneity.iter().map(|species| species.len() as f64).collect()
    }

    /// Finds the lowest common ancestor (LCA) of two nodes.
    ///
    /// # Arguments
    /// * `node_a` - Index of the first node.
    /// * `node_b` - Index of the second node.
    ///
    /// # Returns
    /// The index of the lowest common ancestor.
    pub fn find_lca(&self, node_a: usize, node_b: usize) -> Result<usize, String> {
        let n = self.nodes.len();
        if node_a >= n {
            return Err(format!(
                "node_a index {} is out of bounds (tree has {} nodes)",
                node_a, n
            ));
        }
        if node_b >= n {
            return Err(format!(
                "node_b index {} is out of bounds (tree has {} nodes)",
                node_b, n
            ));
        }

        // Collect ancestors of node_a (including itself)
        let mut ancestors_a = std::collections::HashSet::new();
        let mut current = Some(node_a);
        while let Some(idx) = current {
            ancestors_a.insert(idx);
            current = self.nodes[idx].parent;
        }

        // Walk up from node_b until we find a common ancestor
        let mut current = Some(node_b);
        while let Some(idx) = current {
            if ancestors_a.contains(&idx) {
                return Ok(idx);
            }
            current = self.nodes[idx].parent;
        }

        Err(format!("No common ancestor found for nodes {} and {}", node_a, node_b))
    }

    /// Precomputes a matrix of LCA depths for all node pairs.
    ///
    /// Uses an Euler tour + sparse table for O(n log n) preprocessing and O(1) per
    /// LCA query, giving O(n^2) total instead of the naive O(n^2 * h) approach.
    ///
    /// This is used for efficient distance computation during assortative transfer selection.
    /// The distance between two nodes A and B at time t is: 2 * (t - lca_depth[A][B])
    ///
    /// # Panics
    ///
    /// Panics if the tree has no nodes (via [`LcaTable::new`]).
    ///
    /// # Returns
    /// A symmetric matrix where `result[i][j]` = depth of LCA(i, j).
    pub fn precompute_lca_depths(&self) -> Result<Vec<Vec<f64>>, String> {
        let n = self.nodes.len();
        let lca_table = LcaTable::new(self);
        let mut lca_depths = vec![vec![0.0; n]; n];

        for (i, row) in lca_depths.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate().skip(i) {
                let lca = lca_table.lca(i, j);
                let depth = self.nodes[lca].depth.ok_or_else(|| {
                    format!(
                        "Node {} has no assigned depth. Call assign_depths() before computing distances.",
                        lca
                    )
                })?;
                *cell = depth;
            }
        }
        // Fill symmetric lower triangle
        #[allow(clippy::needless_range_loop)]
        for i in 1..n {
            for j in 0..i {
                let val = lca_depths[j][i];
                lca_depths[i][j] = val;
            }
        }

        Ok(lca_depths)
    }

    /// Computes the path from a node to its ancestor.
    ///
    /// # Arguments
    /// * `node_idx` - Index of the starting node
    /// * `ancestor_idx` - Index of the ancestor (must be an ancestor of node_idx)
    ///
    /// # Returns
    /// A vector of node indices representing the path from node to ancestor (inclusive)
    fn path_to_ancestor(&self, node_idx: usize, ancestor_idx: usize) -> Vec<usize> {
        let mut path = vec![node_idx];
        let mut current = node_idx;
        while current != ancestor_idx {
            current = self.nodes[current].parent.expect("Node should have parent on path to ancestor");
            path.push(current);
        }
        path
    }

    /// Computes the distance between two nodes.
    ///
    /// # Arguments
    /// * `node_a` - Index of the first node
    /// * `node_b` - Index of the second node
    /// * `distance_type` - Type of distance to compute (Topological or Metric)
    ///
    /// # Panics
    ///
    /// Panics if the parent chain from either node to their LCA is broken (i.e., a node
    /// on the path has no parent). This should not happen on a well-formed tree.
    ///
    /// # Returns
    /// The distance between the two nodes.
    pub fn distance_between(&self, node_a: usize, node_b: usize, distance_type: DistanceType) -> Result<f64, String> {
        if node_a == node_b {
            return Ok(0.0);
        }

        let lca = self.find_lca(node_a, node_b)?;

        match distance_type {
            DistanceType::Topological => {
                // Count edges from node_a to LCA and from node_b to LCA
                let path_a = self.path_to_ancestor(node_a, lca);
                let path_b = self.path_to_ancestor(node_b, lca);
                // Subtract 1 from each because path includes the endpoint
                // Total edges = (nodes_a - 1) + (nodes_b - 1)
                Ok(((path_a.len() - 1) + (path_b.len() - 1)) as f64)
            }
            DistanceType::Metric => {
                // Sum branch lengths from node_a to LCA and from node_b to LCA
                let mut distance = 0.0;

                // Path from node_a to LCA (don't include LCA's branch)
                let mut current = node_a;
                while current != lca {
                    distance += self.nodes[current].length;
                    current = self.nodes[current].parent.expect("Node should have parent");
                }

                // Path from node_b to LCA (don't include LCA's branch)
                current = node_b;
                while current != lca {
                    distance += self.nodes[current].length;
                    current = self.nodes[current].parent.expect("Node should have parent");
                }

                Ok(distance)
            }
        }
    }

    /// Computes all pairwise distances between nodes in the tree.
    ///
    /// # Arguments
    /// * `distance_type` - Type of distance to compute (Topological or Metric)
    /// * `leaves_only` - If true, only compute distances between leaf nodes
    ///
    /// # Panics
    ///
    /// Panics if the tree has a broken parent chain (see [`distance_between`](Self::distance_between)).
    ///
    /// # Returns
    /// A vector of PairwiseDistance entries containing all pairs (including symmetric
    /// pairs only, excluding self-distances).
    ///
    /// # Example
    /// ```ignore
    /// let tree = parse_newick("((A:1,B:1):1,C:2):0;").unwrap().to_flat_tree();
    /// let distances = tree.pairwise_distances(DistanceType::Metric, true);
    /// // Returns distances for: (A,B), (A,C), (B,C)  — upper triangle only
    /// ```
    pub fn pairwise_distances(&self, distance_type: DistanceType, leaves_only: bool) -> Result<Vec<PairwiseDistance<'_>>, String> {
        let indices: Vec<usize> = if leaves_only {
            // Get only leaf node indices
            self.nodes.iter()
                .enumerate()
                .filter(|(_, node)| node.left_child.is_none() && node.right_child.is_none())
                .map(|(idx, _)| idx)
                .collect()
        } else {
            // All node indices
            (0..self.nodes.len()).collect()
        };

        let n = indices.len();
        let mut distances = Vec::with_capacity(n * (n - 1) / 2);

        for (pos_i, &i) in indices.iter().enumerate() {
            for &j in &indices[pos_i + 1..] {
                let dist = self.distance_between(i, j, distance_type)?;
                distances.push(PairwiseDistance {
                    node1: &self.nodes[i].name,
                    node2: &self.nodes[j].name,
                    distance: dist,
                });
            }
        }

        Ok(distances)
    }

    /// Computes the pairwise distance matrix as a 2D vector.
    ///
    /// # Arguments
    /// * `distance_type` - Type of distance to compute (Topological or Metric)
    ///
    /// # Panics
    ///
    /// Panics if the tree has a broken parent chain (see [`distance_between`](Self::distance_between)).
    ///
    /// # Returns
    /// A symmetric matrix where `result[i][j]` = distance between nodes i and j.
    pub fn pairwise_distance_matrix(&self, distance_type: DistanceType) -> Result<Vec<Vec<f64>>, String> {
        let n = self.nodes.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for (i, row) in matrix.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate().skip(i) {
                *cell = self.distance_between(i, j, distance_type)?;
            }
        }
        // Fill symmetric lower triangle
        #[allow(clippy::needless_range_loop)]
        for i in 1..n {
            for j in 0..i {
                let val = matrix[j][i];
                matrix[i][j] = val;
            }
        }

        Ok(matrix)
    }
}

#[cfg(test)]
mod tests {
    use crate::newick::parse_newick;
    use crate::node::FlatTree;

    /// Helper: parse a Newick string and return a FlatTree with depths assigned.
    fn make_tree(newick: &str) -> FlatTree {
        let mut nodes = parse_newick(newick).unwrap();
        let mut tree = nodes.pop().unwrap().to_flat_tree();
        tree.assign_depths();
        tree
    }

    // ----------------------------------------------------------------
    // Test 1: 3-leaf ultrametric tree
    //
    //         R (depth=0, len=0)       index 0
    //        / \
    //       I   C (depth=2, len=2)     I = index 1, C = index 4
    //      / \
    //     A   B (depth=2, len=1)       A = index 2, B = index 3
    //
    // Subdivision (unique depths): [0, 1, 2]
    //
    // Expected contemporaneity:
    //   slot 0 -> []          (no branch is alive at time 0)
    //   slot 1 -> [1, 4]     (I and C alive in (0, 1])
    //   slot 2 -> [2, 3, 4]  (A, B, and C alive in (1, 2])
    // ----------------------------------------------------------------
    #[test]
    fn test_contemporaneity_ultrametric() {
        let tree = make_tree("((A:1,B:1):1,C:2):0;");

        let depths = tree.make_subdivision();
        assert_eq!(depths, vec![0.0, 1.0, 2.0]);

        let contemp = tree.find_contemporaneity(&depths);
        assert_eq!(contemp.len(), 3);

        // Slot 0 is always empty
        assert!(contemp[0].is_empty(), "contemporaneity[0] should be empty");

        // Slot 1: internal node I (index 1) and C (index 4)
        assert_eq!(contemp[1], vec![1, 4]);

        // Slot 2: leaves A (2), B (3), and C (4)
        assert_eq!(contemp[2], vec![2, 3, 4]);
    }

    // ----------------------------------------------------------------
    // Test 2: 3-leaf non-ultrametric tree
    //
    //         R (depth=0, len=0)       index 0
    //        / \
    //       I   C (depth=5, len=5)     I = index 1 (depth=1, len=1), C = index 4
    //      / \
    //     A   B                        A = index 2 (depth=2, len=1)
    //                                  B = index 3 (depth=3, len=2)
    //
    // Subdivision: [0, 1, 2, 3, 5]
    //
    // Expected contemporaneity:
    //   slot 0 -> []              (empty, always)
    //   slot 1 -> [1, 4]         (I and C alive in (0, 1])
    //   slot 2 -> [2, 3, 4]      (A, B, C alive in (1, 2])
    //   slot 3 -> [3, 4]         (B and C alive in (2, 3])
    //   slot 4 -> [4]            (only C alive in (3, 5])
    // ----------------------------------------------------------------
    #[test]
    fn test_contemporaneity_non_ultrametric() {
        let tree = make_tree("((A:1,B:2):1,C:5):0;");

        let depths = tree.make_subdivision();
        assert_eq!(depths, vec![0.0, 1.0, 2.0, 3.0, 5.0]);

        let contemp = tree.find_contemporaneity(&depths);
        assert_eq!(contemp.len(), 5);

        assert!(contemp[0].is_empty(), "contemporaneity[0] should be empty");
        assert_eq!(contemp[1], vec![1, 4]);
        assert_eq!(contemp[2], vec![2, 3, 4]);
        assert_eq!(contemp[3], vec![3, 4]);
        assert_eq!(contemp[4], vec![4]);
    }

    // ----------------------------------------------------------------
    // Test 3: Edge cases — root in no intervals, contemporaneity[0] empty
    //
    // Uses a minimal 2-leaf tree: (A:3,B:3):0;
    //
    //         R (depth=0, len=0)       index 0
    //        / \
    //       A   B (depth=3, len=3)     A = index 1, B = index 2
    //
    // Subdivision: [0, 3]
    //
    // Expected contemporaneity:
    //   slot 0 -> []       (always empty)
    //   slot 1 -> [1, 2]   (A and B alive in (0, 3])
    //
    // Root (index 0) must NOT appear in any slot because its branch
    // length is 0, producing an empty range.
    // ----------------------------------------------------------------
    #[test]
    fn test_contemporaneity_edge_cases() {
        let tree = make_tree("(A:3,B:3):0;");

        let depths = tree.make_subdivision();
        assert_eq!(depths, vec![0.0, 3.0]);

        let contemp = tree.find_contemporaneity(&depths);
        assert_eq!(contemp.len(), 2);

        // contemporaneity[0] is always empty
        assert!(contemp[0].is_empty(), "contemporaneity[0] should always be empty");

        // Root (index 0) should not appear anywhere
        for (j, slot) in contemp.iter().enumerate() {
            assert!(
                !slot.contains(&tree.root),
                "root should not appear in any interval, but found in slot {}",
                j,
            );
        }

        // The two leaves should be alive in slot 1
        assert_eq!(contemp[1], vec![1, 2]);
    }

    // ----------------------------------------------------------------
    // Supplementary: verify make_intervals consistency
    // ----------------------------------------------------------------
    #[test]
    fn test_make_intervals() {
        let tree = make_tree("((A:1,B:2):1,C:5):0;");
        let intervals = tree.make_intervals();
        assert_eq!(intervals, vec![0.0, 1.0, 1.0, 1.0, 2.0]);
    }

    // ================================================================
    // LcaTable parity tests: compare O(1) LCA vs naive find_lca
    // ================================================================

    use super::LcaTable;

    /// Helper: assert that LcaTable.lca(i, j) == find_lca(i, j) for all node pairs.
    fn assert_lca_parity(tree: &FlatTree) {
        let lca_table = LcaTable::new(tree);
        let n = tree.nodes.len();

        for i in 0..n {
            for j in 0..n {
                let old_lca = tree.find_lca(i, j).unwrap();
                let new_lca = lca_table.lca(i, j);
                assert_eq!(
                    old_lca, new_lca,
                    "LCA mismatch for nodes ({}, {}): naive={}, sparse_table={}",
                    i, j, old_lca, new_lca
                );
            }
        }
    }

    #[test]
    fn test_lca_table_parity_ultrametric() {
        let tree = make_tree("((A:1,B:1):1,C:2):0;");
        assert_lca_parity(&tree);
    }

    #[test]
    fn test_lca_table_parity_non_ultrametric() {
        let tree = make_tree("((A:1,B:2):1,C:5):0;");
        assert_lca_parity(&tree);
    }

    #[test]
    fn test_lca_table_parity_caterpillar() {
        // Highly unbalanced (caterpillar) tree
        let tree = make_tree("(A:1,(B:1,(C:1,D:1):1):1):0;");
        assert_lca_parity(&tree);
    }

    #[test]
    fn test_lca_table_parity_large_balanced() {
        // 8-leaf balanced tree
        let tree = make_tree("(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1):0;");
        assert_lca_parity(&tree);
    }

    #[test]
    fn test_lca_table_single_node() {
        let tree = make_tree("A:0;");
        let lca_table = LcaTable::new(&tree);
        assert_eq!(lca_table.lca(0, 0), 0);
    }

    #[test]
    fn test_lca_table_two_leaves() {
        let tree = make_tree("(A:1,B:1):0;");
        assert_lca_parity(&tree);
    }

    #[test]
    fn test_lca_table_deep_caterpillar() {
        // Deeply unbalanced: ((((A:1,B:1):1,C:1):1,D:1):1,E:1):0;
        let tree = make_tree("((((A:1,B:1):1,C:1):1,D:1):1,E:1):0;");
        assert_lca_parity(&tree);
    }

    #[test]
    fn test_lca_table_varied_branch_lengths() {
        // Non-uniform branch lengths
        let tree = make_tree("((A:0.5,B:3.7):2.1,(C:0.1,D:10.0):0.3):0;");
        assert_lca_parity(&tree);
    }

    #[test]
    fn test_precompute_lca_depths_parity() {
        // Verify that precompute_lca_depths (now using LcaTable) matches
        // individual find_lca calls on a non-trivial tree
        let tree = make_tree("(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1):0;");

        let lca_depths = tree.precompute_lca_depths().unwrap();

        for i in 0..tree.nodes.len() {
            for j in 0..tree.nodes.len() {
                let lca = tree.find_lca(i, j).unwrap();
                let expected_depth = tree.nodes[lca].depth.unwrap();
                assert!(
                    (lca_depths[i][j] - expected_depth).abs() < 1e-10,
                    "LCA depth mismatch for ({}, {}): matrix={}, find_lca={}",
                    i, j, lca_depths[i][j], expected_depth
                );
            }
        }
    }

    #[test]
    fn test_precompute_lca_depths_parity_non_ultrametric() {
        let tree = make_tree("((A:0.5,B:3.7):2.1,(C:0.1,D:10.0):0.3):0;");

        let lca_depths = tree.precompute_lca_depths().unwrap();

        for i in 0..tree.nodes.len() {
            for j in 0..tree.nodes.len() {
                let lca = tree.find_lca(i, j).unwrap();
                let expected_depth = tree.nodes[lca].depth.unwrap();
                assert!(
                    (lca_depths[i][j] - expected_depth).abs() < 1e-10,
                    "LCA depth mismatch for ({}, {}): matrix={}, find_lca={}",
                    i, j, lca_depths[i][j], expected_depth
                );
            }
        }
    }

    #[test]
    fn test_euler_tour_length() {
        // For n nodes, Euler tour should have exactly 2n-1 entries
        let tree = make_tree("(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1):0;");
        let lca_table = LcaTable::new(&tree);
        let n = tree.nodes.len();
        assert_eq!(lca_table.euler.len(), 2 * n - 1,
            "Euler tour length should be 2n-1 = {}, got {}", 2 * n - 1, lca_table.euler.len());
    }

    #[test]
    fn test_euler_tour_first_occurrence() {
        // Every node should appear in the tour at least once,
        // and first[node] should point to that node in the tour
        let tree = make_tree("((A:1,B:1):1,C:2):0;");
        let lca_table = LcaTable::new(&tree);

        for node_idx in 0..tree.nodes.len() {
            assert_eq!(
                lca_table.euler[lca_table.first[node_idx]], node_idx,
                "first[{}] should point to {} in euler tour", node_idx, node_idx
            );
        }
    }
}
