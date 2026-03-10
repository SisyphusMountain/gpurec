import math

import torch
from torch.utils.data import Dataset
from .extract_parameters import extract_parameters
from .likelihood import E_fixed_point, Pi_fixed_point, Pi_wave_forward, compute_log_likelihood
from .batching import collate_gene_families, collate_wave
from .scheduling import compute_clade_waves
from .preprocess_cpp import _load_extension as _load_species_gene_ext


class GeneDataset(Dataset):
    def __init__(
        self,
        species_tree_path,
        gene_tree_paths,
        genewise, # whether to have a different theta for each gene family
        specieswise, # no need for genewise: it only appear when collating or gradient steps
        pairwise, # changes the size of theta
        dtype=torch.float32,
        device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.genewise = genewise
        self.specieswise = specieswise
        self.pairwise = pairwise     
        self.device = device
        self.dtype = dtype
        ext = _load_species_gene_ext()

        self.species_helpers = ext.preprocess_species(species_tree_path)
        self.tr_mat_unnormalized = torch.log2(self.species_helpers["Recipients_mat"])
        self.S = int(self.species_helpers['S'])
        
        # creating an initial theta
        if specieswise:
            if pairwise:
                theta = -100*torch.ones(self.S, 2, dtype=dtype, device=device)
                # If using pairwise coefficients, the transfer rate is implicitly contained in this matrix.
                self.tr_mat_unnormalized = self.tr_mat_unnormalized - 10.0
            else:
                theta = -10*torch.ones(self.S, 1, dtype=dtype, device=device)  
        else:
            theta = -10000*torch.ones(3, dtype=dtype, device=device)  

        _INV_LN2 = 1.0 / math.log(2.0)
        families = []
        for gpath in gene_tree_paths:
            gene_data = ext.preprocess_gene_with_species(self.species_helpers, gpath)
            ccp = gene_data['ccp']
            # Convert log_split_probs from ln (C++ output) to log2
            ccp['log_split_probs_sorted'] = ccp['log_split_probs_sorted'] * _INV_LN2
            families.append({
                'ccp_helpers': ccp,
                'root_clade_id': int(gene_data['root_clade_id']),
                'leaf_row_index': gene_data['leaf_row_index'],
                'leaf_col_index': gene_data['leaf_col_index'],
                'C': int(ccp['C']),
                'N_splits': int(ccp['N_splits']),
                'theta': theta.clone(), # same theta for all families at the beginning. Will be optimized later.
                'transfer_mat_unnormalized': self.tr_mat_unnormalized,
                'log_split_probs': ccp['log_split_probs_sorted'],
            })
        # stored on CPU. Only move when computing likelihood and optimizing.
        self.families = families
        self.gene_tree_paths = list(gene_tree_paths)
        self.species_tree_path = species_tree_path

        self.num_families = len(families)
    
    def __len__(self):
        return len(self.families)
    
    def __getitem__(self, idx):
        # Will work for a single sample, but will need
        # custom collate_fn to work with batches
        return self.families[idx]
    
    def change_dtype(self, dtype):
        """We may want to change dtype at the very end of optimization."""
        self.dtype = dtype
        for fam in self.families:
            fam['ccp_helpers'] = {k: v.to(dtype=dtype) if torch.is_tensor(v) else v for k, v in fam['ccp_helpers'].items()}
            fam['theta'] = fam['theta'].to(dtype=dtype)
        self.tr_mat_unnormalized = self.tr_mat_unnormalized.to(dtype=dtype)
        self.species_helpers = {k: v.to(dtype=dtype) if torch.is_tensor(v) else v for k, v in self.species_helpers.items()}

    def set_params(self, idx, D, T, L):
        # only if genewise
        theta = torch.log(torch.tensor([D, L, T], device=self.device, dtype=self.dtype))
        self.families[idx]['theta'] = theta.to(device=self.device, dtype=self.dtype)


    @torch.no_grad()
    def compute_likelihood(
        self,
        idx: int = 0,
        *,
        max_iters_E: int = 2000,
        tol_E: float = 1e-6,
        max_iters_Pi: int = 2000,
        tol_Pi: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict:
        """Compute log-likelihood for a dataset element at `idx`.

        Uses stored preprocessing (including leaf indices) to avoid recomputation.
        Returns a dict with Pi, E, components, and log_likelihood.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        fam = self.families[idx]

        def _move_tensor(t: torch.Tensor) -> torch.Tensor:
            if t.dtype.is_floating_point:
                return t.to(device=device, dtype=dtype)
            else:
                return t.to(device=device)

        species_helpers = {k: (_move_tensor(v) if torch.is_tensor(v) else v)
                           for k, v in self.species_helpers.items()}
        ccp_helpers = {k: (_move_tensor(v) if torch.is_tensor(v) else v)
                       for k, v in fam['ccp_helpers'].items()}

        leaf_row_index = _move_tensor(fam['leaf_row_index']).to(torch.long)
        leaf_col_index = _move_tensor(fam['leaf_col_index']).to(torch.long)

        theta = fam['theta'].to(device=device, dtype=dtype)
        transfer_mat_unnorm = self.tr_mat_unnormalized.to(device=device, dtype=dtype)

        log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
            theta,
            transfer_mat_unnorm,
            genewise=self.genewise,
            specieswise=self.specieswise,
            pairwise=self.pairwise,
        )

        # Ensure vector shape for baseline where needed
        if max_transfer_mat.ndim == 2 and max_transfer_mat.shape[-1] == 1:
            max_transfer_vec = max_transfer_mat.squeeze(-1)
        else:
            max_transfer_vec = max_transfer_mat

        # E fixed point
        E_out = E_fixed_point(
            species_helpers=species_helpers,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_mat=max_transfer_vec,
            max_iters=max_iters_E,
            tolerance=tol_E,
            warm_start_E=None,
            dtype=dtype,
            device=device,
        )
        E = E_out['E']
        E_s1 = E_out['E_s1']
        E_s2 = E_out['E_s2']
        Ebar = E_out['E_bar']
        # Pi fixed point
        Pi_out = Pi_fixed_point(
            ccp_helpers=ccp_helpers,
            species_helpers=species_helpers,
            leaf_row_index=leaf_row_index,
            leaf_col_index=leaf_col_index,
            E=E,
            Ebar=Ebar,
            E_s1=E_s1,
            E_s2=E_s2,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            transfer_mat_T=transfer_mat.transpose(-1, -2),
            max_transfer_mat=max_transfer_vec,
            max_iters=max_iters_Pi,
            tolerance=tol_Pi,
            warm_start_Pi=None,
            device=device,
            dtype=dtype,
            genewise=self.genewise,
            specieswise=self.specieswise,
            pairwise=self.pairwise,
            clades_per_gene=None,
            batch_info=None,
        )
        Pi = Pi_out['Pi']

        root_clade_id = int(fam['root_clade_id'])
        logL = compute_log_likelihood(Pi, E, root_clade_id)

        return {
            'log_likelihood': float(logL),
            'Pi': Pi,
            'E': E,
            'Ebar': Ebar,
            'E_s1': E_s1,
            'E_s2': E_s2,
        }

    @torch.no_grad()
    def compute_likelihood_batch(
        self,
        indices: list[int] | None = None,
        *,
        max_iters_E: int = 2000,
        tol_E: float = 1e-12,
        max_iters_Pi: int = 2000,
        tol_Pi: float = 1e-12,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_wave: bool = False,
        chunk_size: int | None = None,
    ) -> list[float]:
        """Compute log-likelihoods for a batch of gene families.

        Returns a list of per-family log-likelihoods, in the same order as `indices`.

        Args:
            chunk_size: If set, process families in chunks of this size to avoid OOM.
                Only applies to the wave path. Recommended: 20 for S~2000.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        if indices is None:
            indices = list(range(len(self.families)))
        if len(indices) == 0:
            return []

        # Handle chunking: split large batches to avoid OOM
        if chunk_size is not None and use_wave and len(indices) > chunk_size:
            all_logLs = []
            for start in range(0, len(indices), chunk_size):
                chunk_indices = indices[start:start + chunk_size]
                all_logLs.extend(self.compute_likelihood_batch(
                    chunk_indices,
                    max_iters_E=max_iters_E, tol_E=tol_E,
                    max_iters_Pi=max_iters_Pi, tol_Pi=tol_Pi,
                    device=device, dtype=dtype,
                    use_wave=use_wave,
                ))
            return all_logLs

        # Build batch items compatible with collate_gene_families
        batch_items = []
        for idx in indices:
            fam = self.families[idx]
            batch_items.append({
                'ccp': fam['ccp_helpers'],
                'leaf_row_index': fam['leaf_row_index'],
                'leaf_col_index': fam['leaf_col_index'],
                'root_clade_id': int(fam['root_clade_id']),
            })

        batched = collate_gene_families(batch_items, dtype=dtype, device=device)
        ccp_helpers = batched['ccp']
        leaf_row_index = batched['leaf_row_index']
        leaf_col_index = batched['leaf_col_index']
        root_clade_ids = batched['root_clade_ids']  # Long[F]
        clades_per_gene = torch.tensor([m['C'] for m in batched['family_meta']], dtype=torch.long, device=device)

        transfer_mat_unnorm = self.tr_mat_unnormalized.to(device=device, dtype=dtype)

        if self.genewise:
            # Stack per-family theta and extract per-gene parameters
            theta_stack = torch.stack([
                self.families[i]['theta'].to(device=device, dtype=dtype) for i in indices
            ], dim=0)
            log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
                theta_stack, transfer_mat_unnorm,
                genewise=True, specieswise=self.specieswise, pairwise=self.pairwise,
            )
            # Ensure max_transfer has shape [N_genes, S]
            if max_transfer_mat.ndim == 3 and max_transfer_mat.shape[-1] == 1:
                max_transfer_vec = max_transfer_mat.squeeze(-1)
            else:
                max_transfer_vec = max_transfer_mat
        else:
            # Shared parameters across families
            theta0 = self.families[indices[0]]['theta'].to(device=device, dtype=dtype)
            log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
                theta0, transfer_mat_unnorm,
                genewise=False, specieswise=self.specieswise, pairwise=self.pairwise,
            )
            if max_transfer_mat.ndim == 2 and max_transfer_mat.shape[-1] == 1:
                max_transfer_vec = max_transfer_mat.squeeze(-1)
            else:
                max_transfer_vec = max_transfer_mat

        # Move species helpers to device/dtype
        def _move_tensor(t: torch.Tensor) -> torch.Tensor:
            if t.dtype.is_floating_point:
                return t.to(device=device, dtype=dtype)
            else:
                return t.to(device=device)

        species_helpers = {k: (_move_tensor(v) if torch.is_tensor(v) else v)
                           for k, v in self.species_helpers.items()}

        # E fixed point (vectorized across genes when parameters are batched)
        E_out = E_fixed_point(
            species_helpers=species_helpers,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_mat=max_transfer_vec,
            max_iters=max_iters_E,
            tolerance=tol_E,
            warm_start_E=None,
            dtype=dtype,
            device=device,
        )
        E = E_out['E']
        E_s1 = E_out['E_s1']
        E_s2 = E_out['E_s2']
        Ebar = E_out['E_bar']

        if use_wave and not self.genewise:
            # Wave-based forward pass with cross-family batching
            families_waves = []
            families_phases = []
            for idx in indices:
                fam = self.families[idx]
                ch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                          for k, v in fam['ccp_helpers'].items()}
                waves_i, phases_i = compute_clade_waves(ch_dev)
                families_waves.append(waves_i)
                families_phases.append(phases_i)

            offsets = [m['clade_offset'] for m in batched['family_meta']]
            cross_waves = collate_wave(families_waves, offsets)

            max_n_waves = max(len(p) for p in families_phases)
            cross_phases = []
            for k in range(max_n_waves):
                phase_k = 1
                for fp in families_phases:
                    if k < len(fp):
                        phase_k = max(phase_k, fp[k])
                cross_phases.append(phase_k)

            Pi_out = Pi_wave_forward(
                waves=cross_waves,
                ccp_helpers=ccp_helpers,
                species_helpers=species_helpers,
                leaf_row_index=leaf_row_index,
                leaf_col_index=leaf_col_index,
                E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
                log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                transfer_mat=transfer_mat,
                max_transfer_mat=max_transfer_vec,
                device=device, dtype=dtype,
                phases=cross_phases,
                local_iters=max_iters_Pi,
                local_tolerance=tol_Pi,
            )
        else:
            # Batched Pi fixed point over all clades
            Pi_out = Pi_fixed_point(
                ccp_helpers=ccp_helpers,
                species_helpers=species_helpers,
                leaf_row_index=leaf_row_index,
                leaf_col_index=leaf_col_index,
                E=E,
                Ebar=Ebar,
                E_s1=E_s1,
                E_s2=E_s2,
                log_pS=log_pS,
                log_pD=log_pD,
                log_pL=log_pL,
                transfer_mat_T=transfer_mat.transpose(-1, -2),
                max_transfer_mat=max_transfer_vec,
                max_iters=max_iters_Pi,
                tolerance=tol_Pi,
                warm_start_Pi=None,
                device=device,
                dtype=dtype,
                genewise=self.genewise,
                specieswise=self.specieswise,
                pairwise=self.pairwise,
                clades_per_gene=(clades_per_gene if self.genewise else None),
                batch_info=(
                    {'seg_ptr': torch.nn.functional.pad(torch.cumsum(clades_per_gene, dim=0), (1, 0))}
                    if self.genewise else None
                ),
            )
        Pi = Pi_out['Pi']

        # Vector of per-family log-likelihoods
        logL_vec = compute_log_likelihood(Pi, E, root_clade_ids)
        return [float(x) for x in logL_vec.detach().cpu().tolist()]
