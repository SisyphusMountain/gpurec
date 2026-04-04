import math
from typing import Any

import torch
from torch.utils.data import Dataset
from .extract_parameters import extract_parameters, extract_parameters_uniform
from .likelihood import E_fixed_point, Pi_fixed_point, Pi_wave_forward, compute_log_likelihood
from .batching import collate_gene_families, collate_wave, build_wave_layout
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

        # Preprocess first family to get species helpers
        first_raw = ext.preprocess(species_tree_path, [str(gene_tree_paths[0])])
        self.species_helpers = first_raw['species']
        self.tr_mat_unnormalized = torch.log2(self.species_helpers["Recipients_mat"])
        self.unnorm_row_max = self.tr_mat_unnormalized.max(dim=-1).values  # [S], precomputed
        self.S = int(self.species_helpers['S'])

        # creating an initial theta (log2-space: rates = 2^theta)
        _THETA_INIT = math.log2(1e-10)
        if pairwise:
            if specieswise:
                raise ValueError("specieswise and pairwise are mutually exclusive")
            # Pairwise: theta has D and L only; T is implicit in the transfer matrix
            theta = _THETA_INIT * torch.ones(2, dtype=dtype, device=device)
            self.tr_mat_unnormalized = self.tr_mat_unnormalized - 10.0
        elif specieswise:
            theta = _THETA_INIT * torch.ones(self.S, 3, dtype=dtype, device=device)
        else:
            theta = _THETA_INIT * torch.ones(3, dtype=dtype, device=device)

        _INV_LN2 = 1.0 / math.log(2.0)
        families = []
        for i, gpath in enumerate(gene_tree_paths):
            if i == 0:
                raw = first_raw
            else:
                raw = ext.preprocess(species_tree_path, [str(gpath)])
            ccp = raw['ccp']
            # Convert log_split_probs from ln (C++ output) to log2
            ccp['log_split_probs_sorted'] = ccp['log_split_probs_sorted'] * _INV_LN2
            families.append({
                'ccp_helpers': ccp,
                'root_clade_id': int(ccp['root_clade_id']),
                'leaf_row_index': raw['leaf_row_index'],
                'leaf_col_index': raw['leaf_col_index'],
                'C': int(ccp['C']),
                'N_splits': int(ccp['N_splits']),
                'theta': theta.clone(),
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
        theta = torch.log2(torch.tensor([D, L, T], device=self.device, dtype=self.dtype))
        self.families[idx]['theta'] = theta.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _normalize_max_transfer(max_transfer_mat: torch.Tensor) -> torch.Tensor:
        if max_transfer_mat.ndim >= 2 and max_transfer_mat.shape[-1] == 1:
            return max_transfer_mat.squeeze(-1)
        return max_transfer_mat

    def _resolve_device_dtype(
        self,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> tuple[torch.device, torch.dtype]:
        return (self.device if device is None else device,
                self.dtype if dtype is None else dtype)

    @staticmethod
    def _move_tensor(t: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if t.dtype.is_floating_point:
            return t.to(device=device, dtype=dtype)
        return t.to(device=device)

    def _move_mapping(self, mapping: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        return {
            k: (self._move_tensor(v, device=device, dtype=dtype) if torch.is_tensor(v) else v)
            for k, v in mapping.items()
        }

    def _species_helpers_for_mode(
        self,
        *,
        pibar_mode: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[dict[str, Any], torch.Tensor | None]:
        skip_keys = set()
        if pibar_mode == 'uniform_approx':
            skip_keys = {'ancestors_dense', 'Recipients_mat'}
        elif pibar_mode == 'uniform':
            skip_keys = {'Recipients_mat'}

        species_helpers = {
            k: (self._move_tensor(v, device=device, dtype=dtype) if torch.is_tensor(v) and k not in skip_keys else v)
            for k, v in self.species_helpers.items()
        }

        ancestors_T = None
        if pibar_mode == 'uniform':
            ancestors_T = species_helpers['ancestors_dense'].T.to_sparse_coo()
        return species_helpers, ancestors_T

    def _extract_single_params(
        self,
        fam: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        theta = fam['theta'].to(device=device, dtype=dtype)
        transfer_mat_unnorm = self.tr_mat_unnormalized.to(device=device, dtype=dtype)
        log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
            theta,
            transfer_mat_unnorm,
            genewise=self.genewise,
            specieswise=self.specieswise,
            pairwise=self.pairwise,
        )
        return log_pS, log_pD, log_pL, transfer_mat, self._normalize_max_transfer(max_transfer_mat)

    def _extract_batch_params(
        self,
        indices: list[int],
        *,
        pibar_mode: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        use_uniform_extract = (pibar_mode in ('uniform_approx', 'uniform')) and not self.pairwise

        if use_uniform_extract and not self.genewise:
            unnorm_row_max = self.unnorm_row_max.to(device=device, dtype=dtype)
            theta0 = self.families[indices[0]]['theta'].to(device=device, dtype=dtype)
            log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters_uniform(
                theta0, unnorm_row_max, specieswise=self.specieswise,
            )
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat

        if use_uniform_extract and self.genewise:
            unnorm_row_max = self.unnorm_row_max.to(device=device, dtype=dtype)
            theta_stack = torch.stack([
                self.families[i]['theta'].to(device=device, dtype=dtype) for i in indices
            ], dim=0)
            log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters_uniform(
                theta_stack, unnorm_row_max, specieswise=self.specieswise, genewise=True,
            )
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat

        transfer_mat_unnorm = self.tr_mat_unnormalized.to(device=device, dtype=dtype)
        if self.genewise:
            theta_stack = torch.stack([
                self.families[i]['theta'].to(device=device, dtype=dtype) for i in indices
            ], dim=0)
            log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
                theta_stack, transfer_mat_unnorm,
                genewise=True, specieswise=self.specieswise, pairwise=self.pairwise,
            )
        else:
            theta0 = self.families[indices[0]]['theta'].to(device=device, dtype=dtype)
            log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
                theta0, transfer_mat_unnorm,
                genewise=False, specieswise=self.specieswise, pairwise=self.pairwise,
            )

        return log_pS, log_pD, log_pL, transfer_mat, self._normalize_max_transfer(max_transfer_mat)

    def _solve_e_fixed_point(
        self,
        *,
        species_helpers: dict[str, Any],
        log_pS: torch.Tensor,
        log_pD: torch.Tensor,
        log_pL: torch.Tensor,
        transfer_mat: torch.Tensor | None,
        max_transfer_vec: torch.Tensor,
        max_iters_E: int,
        tol_E: float,
        device: torch.device,
        dtype: torch.dtype,
        pibar_mode: str = 'dense',
        ancestors_T: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | int]:
        return E_fixed_point(
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
            pibar_mode=pibar_mode,
            ancestors_T=ancestors_T,
        )

    def _compute_batch_non_genewise(
        self,
        *,
        indices: list[int],
        batched: dict[str, Any],
        ccp_helpers: dict[str, Any],
        leaf_row_index: torch.Tensor,
        leaf_col_index: torch.Tensor,
        root_clade_ids: torch.Tensor,
        species_helpers: dict[str, Any],
        E: torch.Tensor,
        Ebar: torch.Tensor,
        E_s1: torch.Tensor,
        E_s2: torch.Tensor,
        log_pS: torch.Tensor,
        log_pD: torch.Tensor,
        log_pL: torch.Tensor,
        transfer_mat: torch.Tensor | None,
        max_transfer_vec: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        max_iters_Pi: int,
        tol_Pi: float,
        pibar_mode: str,
    ) -> list[float]:
        families_waves = []
        families_phases = []
        for idx in indices:
            fam = self.families[idx]
            waves_i, phases_i = compute_clade_waves(fam['ccp_helpers'])
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

        wave_layout = build_wave_layout(
            waves=cross_waves,
            phases=cross_phases,
            ccp_helpers=ccp_helpers,
            leaf_row_index=leaf_row_index,
            leaf_col_index=leaf_col_index,
            root_clade_ids=root_clade_ids,
            device=device,
            dtype=dtype,
        )

        Pi_out = Pi_wave_forward(
            wave_layout=wave_layout,
            species_helpers=species_helpers,
            E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_mat=max_transfer_vec,
            device=device, dtype=dtype,
            local_iters=max_iters_Pi,
            local_tolerance=tol_Pi,
            pibar_mode=pibar_mode,
        )

        logL_vec = compute_log_likelihood(Pi_out['Pi'], E, root_clade_ids)
        return [float(x) for x in logL_vec.detach().cpu().tolist()]

    def _compute_batch_genewise(
        self,
        *,
        indices: list[int],
        species_helpers: dict[str, Any],
        E: torch.Tensor,
        Ebar: torch.Tensor,
        E_s1: torch.Tensor,
        E_s2: torch.Tensor,
        log_pS: torch.Tensor,
        log_pD: torch.Tensor,
        log_pL: torch.Tensor,
        transfer_mat: torch.Tensor | None,
        max_transfer_vec: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        max_iters_Pi: int,
        tol_Pi: float,
        pibar_mode: str,
    ) -> list[float]:
        logL_list = []
        for fam_idx, global_idx in enumerate(indices):
            fam = self.families[global_idx]
            single_item = {
                'ccp': fam['ccp_helpers'],
                'leaf_row_index': fam['leaf_row_index'],
                'leaf_col_index': fam['leaf_col_index'],
                'root_clade_id': int(fam['root_clade_id']),
            }
            single_batched = collate_gene_families([single_item], dtype=dtype, device=device)
            single_ccp = single_batched['ccp']
            single_li = single_batched['leaf_row_index']
            single_lc = single_batched['leaf_col_index']
            single_root = single_batched['root_clade_ids']

            E_i = E[fam_idx]
            E_s1_i = E_s1[fam_idx]
            E_s2_i = E_s2[fam_idx]
            Ebar_i = Ebar[fam_idx]

            pS_i = log_pS[fam_idx]
            pD_i = log_pD[fam_idx]
            pL_i = log_pL[fam_idx]
            mt_i = max_transfer_vec[fam_idx]

            waves_i, phases_i = compute_clade_waves(fam['ccp_helpers'])
            wave_layout_i = build_wave_layout(
                waves=waves_i,
                phases=phases_i,
                ccp_helpers=single_ccp,
                leaf_row_index=single_li,
                leaf_col_index=single_lc,
                root_clade_ids=single_root,
                device=device,
                dtype=dtype,
            )

            tm_i = transfer_mat[fam_idx] if transfer_mat is not None else None

            Pi_out_i = Pi_wave_forward(
                wave_layout=wave_layout_i,
                species_helpers=species_helpers,
                E=E_i, Ebar=Ebar_i, E_s1=E_s1_i, E_s2=E_s2_i,
                log_pS=pS_i, log_pD=pD_i, log_pL=pL_i,
                transfer_mat=tm_i,
                max_transfer_mat=mt_i,
                device=device, dtype=dtype,
                local_iters=max_iters_Pi,
                local_tolerance=tol_Pi,
                pibar_mode=pibar_mode,
            )
            logL_i = compute_log_likelihood(Pi_out_i['Pi'], E_i, single_root)
            logL_list.append(float(logL_i.item()))

        return logL_list


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
        use_wave: bool = False,
        pibar_mode: str = 'dense',
    ) -> dict:
        """Compute log-likelihood for a dataset element at `idx`.

        Uses stored preprocessing (including leaf indices) to avoid recomputation.
        Returns a dict with Pi, E, components, and log_likelihood.

        Args:
            use_wave: If True, use wave instead of fixed-point.
            pibar_mode: Pibar strategy for wave path ('dense', 'uniform_approx', etc.).
        """
        device, dtype = self._resolve_device_dtype(device, dtype)

        fam = self.families[idx]
        species_helpers = self._move_mapping(self.species_helpers, device=device, dtype=dtype)
        ccp_helpers = self._move_mapping(fam['ccp_helpers'], device=device, dtype=dtype)

        leaf_row_index = self._move_tensor(fam['leaf_row_index'], device=device, dtype=dtype).to(torch.long)
        leaf_col_index = self._move_tensor(fam['leaf_col_index'], device=device, dtype=dtype).to(torch.long)

        log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = self._extract_single_params(
            fam, device=device, dtype=dtype,
        )

        E_out = self._solve_e_fixed_point(
            species_helpers=species_helpers,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_vec=max_transfer_vec,
            max_iters_E=max_iters_E,
            tol_E=tol_E,
            device=device,
            dtype=dtype,
        )
        E = E_out['E']
        E_s1 = E_out['E_s1']
        E_s2 = E_out['E_s2']
        Ebar = E_out['E_bar']
        root_clade_id = int(fam['root_clade_id'])
        # Pi computation: wave-based or fixed-point
        if use_wave:
            waves, phases = compute_clade_waves(ccp_helpers)
            wave_layout = build_wave_layout(
                waves=waves, phases=phases,
                ccp_helpers=ccp_helpers,
                leaf_row_index=leaf_row_index,
                leaf_col_index=leaf_col_index,
                root_clade_ids=torch.tensor([root_clade_id], dtype=torch.long, device=device),
                device=device, dtype=dtype,
            )
            Pi_out = Pi_wave_forward(
                wave_layout=wave_layout,
                species_helpers=species_helpers,
                E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
                log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                transfer_mat=transfer_mat,
                max_transfer_mat=max_transfer_vec,
                device=device, dtype=dtype,
                local_iters=max_iters_Pi,
                local_tolerance=tol_Pi,
                pibar_mode=pibar_mode,
            )
        else:
            Pi_out = Pi_fixed_point(
                ccp_helpers=ccp_helpers,
                species_helpers=species_helpers,
                leaf_row_index=leaf_row_index,
                leaf_col_index=leaf_col_index,
                E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
                log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                transfer_mat_T=transfer_mat.transpose(-1, -2),
                max_transfer_mat=max_transfer_vec,
                max_iters=max_iters_Pi, tolerance=tol_Pi,
                warm_start_Pi=None, device=device, dtype=dtype,
            )
        Pi = Pi_out['Pi']
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
        chunk_size: int | None = None,
        max_wave_size: int = 4096,
        pibar_mode: str = 'dense',
    ) -> list[float]:
        """Compute log-likelihoods for a batch of gene families.

        Returns a list of per-family log-likelihoods, in the same order as `indices`.

        Uses wave-ordered forward pass for non-genewise families,
        falls back to fixed-point for genewise.

        Args:
            chunk_size: If set, process families in chunks of this size to avoid OOM.
                Recommended: 20 for S~2000.
            max_wave_size: Max clades per wave for wave scheduling. Default 4096.
            pibar_mode: 'dense' (cuBLAS matmul) or 'uniform_approx' (O(W*S) approximation).
                Use 'uniform_approx' for large S where the transfer matrix is nearly uniform.
        """
        device, dtype = self._resolve_device_dtype(device, dtype)

        if indices is None:
            indices = list(range(len(self.families)))
        if len(indices) == 0:
            return []

        # Handle chunking: split large batches to avoid OOM
        if chunk_size is not None and len(indices) > chunk_size:
            all_logLs = []
            for start in range(0, len(indices), chunk_size):
                chunk_indices = indices[start:start + chunk_size]
                all_logLs.extend(self.compute_likelihood_batch(
                    chunk_indices,
                    max_iters_E=max_iters_E, tol_E=tol_E,
                    max_iters_Pi=max_iters_Pi, tol_Pi=tol_Pi,
                    device=device, dtype=dtype,
                    chunk_size=chunk_size,
                    max_wave_size=max_wave_size,
                    pibar_mode=pibar_mode,
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

        log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = self._extract_batch_params(
            indices,
            pibar_mode=pibar_mode,
            device=device,
            dtype=dtype,
        )

        species_helpers, ancestors_T = self._species_helpers_for_mode(
            pibar_mode=pibar_mode,
            device=device,
            dtype=dtype,
        )

        # E fixed point (vectorized across genes when parameters are batched)
        E_out = self._solve_e_fixed_point(
            species_helpers=species_helpers,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_vec=max_transfer_vec,
            max_iters_E=max_iters_E,
            tol_E=tol_E,
            device=device,
            dtype=dtype,
            pibar_mode=pibar_mode,
            ancestors_T=ancestors_T,
        )
        E = E_out['E']
        E_s1 = E_out['E_s1']
        E_s2 = E_out['E_s2']
        Ebar = E_out['E_bar']

        # Free large [S,S] tensors after E_step when using uniform modes
        if pibar_mode in ('uniform_approx', 'uniform'):
            transfer_mat = None

        if not self.genewise:
            return self._compute_batch_non_genewise(
                indices=indices,
                batched=batched,
                ccp_helpers=ccp_helpers,
                leaf_row_index=leaf_row_index,
                leaf_col_index=leaf_col_index,
                root_clade_ids=root_clade_ids,
                species_helpers=species_helpers,
                E=E,
                Ebar=Ebar,
                E_s1=E_s1,
                E_s2=E_s2,
                log_pS=log_pS,
                log_pD=log_pD,
                log_pL=log_pL,
                transfer_mat=transfer_mat,
                max_transfer_vec=max_transfer_vec,
                device=device,
                dtype=dtype,
                max_iters_Pi=max_iters_Pi,
                tol_Pi=tol_Pi,
                pibar_mode=pibar_mode,
            )

        return self._compute_batch_genewise(
            indices=indices,
            species_helpers=species_helpers,
            E=E,
            Ebar=Ebar,
            E_s1=E_s1,
            E_s2=E_s2,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_vec=max_transfer_vec,
            device=device,
            dtype=dtype,
            max_iters_Pi=max_iters_Pi,
            tol_Pi=tol_Pi,
            pibar_mode=pibar_mode,
        )
