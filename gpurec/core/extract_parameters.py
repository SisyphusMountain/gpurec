import torch
from .log2_utils import log2_softmax


def extract_parameters(theta, transfer_mat_unnormalized, genewise, specieswise, pairwise):
    if genewise and specieswise:
        N_genes, N_sp, _ = theta.shape
        zeros_tensor = theta.new_zeros((N_genes, N_sp, 1))
        if pairwise:
            complete_theta = torch.cat((zeros_tensor, theta, transfer_mat_unnormalized.unsqueeze(0).expand(N_genes, -1, -1)), dim=-1)
            result = log2_softmax(complete_theta, dim=-1)
            log_pS = result[...,0]
            log_pD = result[...,1]
            log_pL = result[...,2]
            max_transfer_mat = torch.max(transfer_mat_unnormalized, dim=-1, keepdim=True).values
            transfer_mat = torch.exp2(result[...,3:] - max_transfer_mat)
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat
        else:
            complete_theta = torch.cat((zeros_tensor, theta), dim=-1)
            result = log2_softmax(complete_theta, dim=-1)
            log_pS = result[...,0]
            log_pD = result[...,1]
            log_pL = result[...,2]
            log_pT = result[...,3] # All have shape [N_genes, N_sp]
            # transfer_mat has shape [N_sp, N_sp]. We need one per gene, so we add a broadcasting dimension
            log_transfer_mat = log_pT.unsqueeze(-1) + transfer_mat_unnormalized.unsqueeze(0) # shape [N_genes, N_sp, N_sp]
            max_transfer_mat = torch.max(log_transfer_mat, dim=-1, keepdim=True).values
            transfer_mat = torch.exp2(log_transfer_mat - max_transfer_mat)
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat
    elif genewise:
        N_genes, _ = theta.shape
        zeros_tensor = theta.new_zeros((N_genes, 1))
        if pairwise:
            raise NotImplementedError("We don't implement pairwise transfer coefficients if we don't even use specieswise parameters.")
        else:
            complete_theta = torch.cat((zeros_tensor, theta), dim=-1)
            result = log2_softmax(complete_theta, dim=-1)
            log_pS = result[...,0]
            log_pD = result[...,1]
            log_pL = result[...,2]
            log_pT = result[...,3] # All have shape [N_genes]
            # transfer_mat has shape [N_sp, N_sp]. We need one per gene, so we add a broadcasting dimension
            log_transfer_mat = log_pT.view(N_genes, 1, 1) + transfer_mat_unnormalized.unsqueeze(0) # shape [N_genes, N_sp, N_sp]
            max_transfer_mat = torch.max(log_transfer_mat, dim=-1, keepdim=True).values
            transfer_mat = torch.exp2(log_transfer_mat - max_transfer_mat)
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat
    elif specieswise:
        N_sp, _ = theta.shape
        zeros_tensor = theta.new_zeros((N_sp, 1))
        if pairwise:
            complete_theta = torch.cat((zeros_tensor, theta, transfer_mat_unnormalized), dim=-1)
            result = log2_softmax(complete_theta, dim=-1)
            log_pS = result[...,0]
            log_pD = result[...,1]
            log_pL = result[...,2]
            max_transfer_mat = torch.max(transfer_mat_unnormalized, dim=-1, keepdim=True).values
            transfer_mat = torch.exp2(result[...,3:] - max_transfer_mat)
            # s = log_pS[0]
            # d = log_pD[0]
            # l = log_pL[0]
            # t = transfer_mat[0]
            # m = max_transfer_mat[0]
            # We have (t*torch.exp2(m)).sum() + torch.exp2(s) + torch.exp2(d) + torch.exp2(l) = 1.0
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat
        else:
            complete_theta = torch.cat((zeros_tensor, theta), dim=-1)
            result = log2_softmax(complete_theta, dim=-1)
            log_pS = result[...,0]
            log_pD = result[...,1]
            log_pL = result[...,2]
            log_pT = result[...,3] # All have shape [N_sp]
            # transfer_mat has shape [N_sp, N_sp]
            # Each donor species i has row logits log_pT[i] + transfer_mat_unnormalized[i, :]
            log_transfer_mat = log_pT.unsqueeze(-1) + transfer_mat_unnormalized  # shape [N_sp, N_sp]
            max_transfer_mat = torch.max(log_transfer_mat, dim=-1, keepdim=True).values
            transfer_mat = torch.exp2(log_transfer_mat - max_transfer_mat)
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat
    else:
        zeros_tensor = theta.new_zeros((1,))
        if pairwise:
            N_sp, _ = transfer_mat_unnormalized.shape
            complete_theta = torch.cat((zeros_tensor.unsqueeze(0).expand(N_sp, -1), theta.unsqueeze(0).expand(N_sp, -1), transfer_mat_unnormalized), dim=-1)
            result = log2_softmax(complete_theta, dim=-1)
            log_pS = result[...,0].squeeze(0)
            log_pD = result[...,1].squeeze(0)
            log_pL = result[...,2].squeeze(0)
            max_transfer_mat = torch.max(transfer_mat_unnormalized, dim=-1, keepdim=True).values
            transfer_mat = torch.exp2(result[...,3:] - max_transfer_mat)
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat
        else:
            complete_theta = torch.cat((zeros_tensor, theta), dim=-1)
            result = log2_softmax(complete_theta, dim=-1)
            log_pS = result[...,0]
            log_pD = result[...,1]
            log_pL = result[...,2]
            log_pT = result[...,3] # shape []
            # transfer_mat has shape [N_sp, N_sp]
            # Scalar T distributed using provided unnormalized recipient logits per donor row
            log_transfer_mat = log_pT + transfer_mat_unnormalized  # shape [N_sp, N_sp]
            max_transfer_mat = torch.max(log_transfer_mat, dim=-1, keepdim=True).values
            transfer_mat = torch.exp2(log_transfer_mat - max_transfer_mat)
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat


def extract_parameters_uniform(theta, unnorm_row_max, specieswise, genewise=False):
    """Extract SDTL parameters without materializing the [S,S] transfer matrix.

    For the uniform pibar approximation, we only need max_transfer_mat (per-row
    maxima of the log transfer matrix).  Since log_transfer_mat[i,j] =
    log_pT[i] + unnorm[i,j], the row max is log_pT[i] + max_j(unnorm[i,j]).
    This avoids the O(S^2) exp2 and eliminates the PCIe transfer of the full
    [S,S] matrix.

    Parameters
    ----------
    theta : Tensor
        Rate parameters (same as extract_parameters).
        When genewise=True: [G, 3] (non-specieswise) or [G, S, 3] (specieswise).
    unnorm_row_max : Tensor [S]
        Precomputed row maxima of transfer_mat_unnormalized: max_j(unnorm[i,j]).
    specieswise : bool
        Whether rates are per-species.
    genewise : bool
        Whether rates are per-gene.

    Returns
    -------
    log_pS, log_pD, log_pL, transfer_mat (None), max_transfer_mat
    """
    if genewise:
        if specieswise:
            # theta [G, S, 3]
            G, S, _ = theta.shape
            zeros = theta.new_zeros((G, S, 1))
            result = log2_softmax(torch.cat((zeros, theta), dim=-1), dim=-1)
            log_pS = result[..., 0]   # [G, S]
            log_pD = result[..., 1]   # [G, S]
            log_pL = result[..., 2]   # [G, S]
            log_pT = result[..., 3]   # [G, S]
            # mt[g, s] = log_pT[g, s] + unnorm_row_max[s]
            max_transfer_mat = log_pT + unnorm_row_max  # [G, S]
        else:
            # theta [G, 3]
            zeros = theta.new_zeros((theta.shape[0], 1))
            result = log2_softmax(torch.cat((zeros, theta), dim=-1), dim=-1)
            log_pS = result[:, 0]     # [G]
            log_pD = result[:, 1]     # [G]
            log_pL = result[:, 2]     # [G]
            log_pT = result[:, 3]     # [G]
            # mt[g, s] = log_pT[g] + unnorm_row_max[s]
            max_transfer_mat = log_pT.unsqueeze(-1) + unnorm_row_max  # [G, S]
        return log_pS, log_pD, log_pL, None, max_transfer_mat
    if specieswise:
        N_sp, _ = theta.shape
        zeros_tensor = theta.new_zeros((N_sp, 1))
        complete_theta = torch.cat((zeros_tensor, theta), dim=-1)
        result = log2_softmax(complete_theta, dim=-1)
        log_pS = result[..., 0]
        log_pD = result[..., 1]
        log_pL = result[..., 2]
        log_pT = result[..., 3]  # [S]
        max_transfer_mat = log_pT + unnorm_row_max  # [S]
    else:
        zeros_tensor = theta.new_zeros((1,))
        complete_theta = torch.cat((zeros_tensor, theta), dim=-1)
        result = log2_softmax(complete_theta, dim=-1)
        log_pS = result[0]
        log_pD = result[1]
        log_pL = result[2]
        log_pT = result[3]  # scalar
        max_transfer_mat = log_pT + unnorm_row_max  # [S]
    return log_pS, log_pD, log_pL, None, max_transfer_mat


if __name__ == "__main__":
    def make_inputs(N_sp=10, N_genes=12, seed=0):
        g = torch.Generator().manual_seed(seed)
        theta1 = torch.randn((3,), generator=g)
        theta2 = torch.randn((N_sp, 3), generator=g)
        theta3 = torch.randn((N_genes, 3), generator=g)
        theta4 = torch.randn((N_genes, N_sp, 3), generator=g)
        theta5 = torch.randn((N_sp, 2), generator=g)
        tr_mat = torch.randn((N_sp, N_sp), generator=g)
        return theta1, theta2, theta3, theta4, theta5, tr_mat


    def test_shapes_neither_nonpairwise():
        N_sp, N_genes = 10, 12
        theta1, _, _, _, _, tr_mat = make_inputs(N_sp, N_genes)
        log_pS, log_pD, log_pL, T, T_max = extract_parameters(theta1, tr_mat, genewise=False, specieswise=False, pairwise=False)
        assert log_pS.shape == torch.Size([])
        assert log_pD.shape == torch.Size([])
        assert log_pL.shape == torch.Size([])
        assert T.shape == (N_sp, N_sp)
        assert T_max.shape == (N_sp, 1)
        print("test_shapes_neither_nonpairwise PASSED")

    def test_shapes_neither_pairwise():
        N_sp, N_genes = 10, 12
        _, _, _, _, _, tr_mat = make_inputs(N_sp, N_genes)
        theta = torch.randn((2,))  # [D, L]
        log_pS, log_pD, log_pL, T, T_max = extract_parameters(theta, tr_mat, genewise=False, specieswise=False, pairwise=True)
        assert log_pS.shape == (N_sp,)
        assert log_pD.shape == (N_sp,)
        assert log_pL.shape == (N_sp,)
        assert T.shape == (N_sp, N_sp)
        assert T_max.shape == (N_sp, 1)
        print("test_shapes_neither_pairwise PASSED")

    def test_shapes_specieswise_nonpairwise():
        N_sp, N_genes = 10, 12
        _, theta2, _, _, _, tr_mat = make_inputs(N_sp, N_genes)
        log_pS, log_pD, log_pL, T, T_max = extract_parameters(theta2, tr_mat, genewise=False, specieswise=True, pairwise=False)
        assert log_pS.shape == (N_sp,)
        assert log_pD.shape == (N_sp,)
        assert log_pL.shape == (N_sp,)
        assert T.shape == (N_sp, N_sp)
        assert T_max.shape == (N_sp, 1)
        print("test_shapes_specieswise_nonpairwise PASSED")

    def test_shapes_specieswise_pairwise():
        N_sp, N_genes = 10, 12
        _, _, _, _, theta5, tr_mat = make_inputs(N_sp, N_genes)
        log_pS, log_pD, log_pL, T, T_max = extract_parameters(theta5, tr_mat, genewise=False, specieswise=True, pairwise=True)
        assert log_pS.shape == (N_sp,)
        assert log_pD.shape == (N_sp,)
        assert log_pL.shape == (N_sp,)
        assert T.shape == (N_sp, N_sp)
        assert T_max.shape == (N_sp, 1)
        print("test_shapes_specieswise_pairwise PASSED")

    def test_shapes_genewise_nonpairwise():
        N_sp, N_genes = 10, 12
        _, _, theta3, _, _, tr_mat = make_inputs(N_sp, N_genes)
        log_pS, log_pD, log_pL, T, T_max = extract_parameters(theta3, tr_mat, genewise=True, specieswise=False, pairwise=False)
        assert log_pS.shape == (N_genes,)
        assert log_pD.shape == (N_genes,)
        assert log_pL.shape == (N_genes,)
        assert T.shape == (N_genes, N_sp, N_sp)
        assert T_max.shape == (N_genes, N_sp, 1)
        print("test_shapes_genewise_nonpairwise PASSED")


    def test_shapes_genewise_specieswise_nonpairwise():
        N_sp, N_genes = 10, 12
        _, _, _, theta4, _, tr_mat = make_inputs(N_sp, N_genes)
        log_pS, log_pD, log_pL, T, T_max = extract_parameters(theta4, tr_mat, genewise=True, specieswise=True, pairwise=False)
        assert log_pS.shape == (N_genes, N_sp)
        assert log_pD.shape == (N_genes, N_sp)
        assert log_pL.shape == (N_genes, N_sp)
        assert T.shape == (N_genes, N_sp, N_sp)
        assert T_max.shape == (N_genes, N_sp, 1)
        print("test_shapes_genewise_specieswise_nonpairwise PASSED")

    def test_shapes_genewise_specieswise_pairwise():
        N_sp, N_genes = 10, 12
        _, _, _, _, _, tr_mat = make_inputs(N_sp, N_genes)
        theta = torch.randn((N_genes, N_sp, 2))  # [D, L]
        log_pS, log_pD, log_pL, T, T_max = extract_parameters(theta, tr_mat, genewise=True, specieswise=True, pairwise=True)
        assert log_pS.shape == (N_genes, N_sp)
        assert log_pD.shape == (N_genes, N_sp)
        assert log_pL.shape == (N_genes, N_sp)
        assert T.shape == (N_genes, N_sp, N_sp)
        assert T_max.shape == (N_sp, 1)
        print("test_shapes_genewise_specieswise_pairwise PASSED")


    test_shapes_neither_nonpairwise()
    test_shapes_neither_pairwise()
    test_shapes_specieswise_nonpairwise()
    test_shapes_specieswise_pairwise()
    test_shapes_genewise_nonpairwise()
    test_shapes_genewise_specieswise_nonpairwise()
    test_shapes_genewise_specieswise_pairwise()