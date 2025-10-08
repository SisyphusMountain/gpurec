import torch


def extract_parameters(theta, transfer_mat_unnormalized, genewise, specieswise, pairwise):
    if genewise and specieswise:
        N_genes, N_sp, _ = theta.shape
        zeros_tensor = theta.new_zeros((N_genes, N_sp, 1))
        if pairwise:
            complete_theta = torch.cat((zeros_tensor, theta, transfer_mat_unnormalized.unsqueeze(0).expand(N_genes, -1, -1)), dim=-1)
            result = torch.log_softmax(complete_theta, dim=-1)
            log_pS = result[...,0]
            log_pD = result[...,1]
            log_pL = result[...,2]
            max_transfer_mat = torch.max(transfer_mat_unnormalized, dim=-1, keepdim=True).values
            transfer_mat = torch.exp(result[...,3:] - max_transfer_mat)
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat
        else:
            complete_theta = torch.cat((zeros_tensor, theta), dim=-1)
            result = torch.log_softmax(complete_theta, dim=-1)
            log_pS = result[...,0]
            log_pD = result[...,1]
            log_pL = result[...,2]
            log_pT = result[...,3] # All have shape [N_genes, N_sp]
            # transfer_mat has shape [N_sp, N_sp]. We need one per gene, so we add a broadcasting dimension
            log_transfer_mat = log_pT.unsqueeze(-1) + transfer_mat_unnormalized.unsqueeze(0) # shape [N_genes, N_sp, N_sp]
            max_transfer_mat = torch.max(log_transfer_mat, dim=-1, keepdim=True).values
            transfer_mat = torch.exp(log_transfer_mat - max_transfer_mat)
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat
    elif genewise:
        N_genes, _ = theta.shape
        zeros_tensor = theta.new_zeros((N_genes, 1))
        if pairwise:
            raise NotImplementedError("We don't implement pairwise transfer coefficients if we don't even use specieswise parameters.")
        else:
            complete_theta = torch.cat((zeros_tensor, theta), dim=-1)
            result = torch.log_softmax(complete_theta, dim=-1)
            log_pS = result[...,0]
            log_pD = result[...,1]
            log_pL = result[...,2]
            log_pT = result[...,3] # All have shape [N_genes]
            # transfer_mat has shape [N_sp, N_sp]. We need one per gene, so we add a broadcasting dimension
            log_transfer_mat = log_pT.view(N_genes, 1, 1) + transfer_mat_unnormalized.unsqueeze(0) # shape [N_genes, N_sp, N_sp]
            max_transfer_mat = torch.max(log_transfer_mat, dim=-1, keepdim=True).values
            transfer_mat = torch.exp(log_transfer_mat - max_transfer_mat)
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat
    elif specieswise:
        N_sp, _ = theta.shape
        zeros_tensor = theta.new_zeros((N_sp, 1))
        if pairwise:
            complete_theta = torch.cat((zeros_tensor, theta, transfer_mat_unnormalized), dim=-1)
            result = torch.log_softmax(complete_theta, dim=-1)
            log_pS = result[...,0]
            log_pD = result[...,1]
            log_pL = result[...,2]
            max_transfer_mat = torch.max(transfer_mat_unnormalized, dim=-1, keepdim=True).values
            transfer_mat = torch.exp(result[...,3:] - max_transfer_mat)
            # s = log_pS[0]
            # d = log_pD[0]
            # l = log_pL[0]
            # t = transfer_mat[0]
            # m = max_transfer_mat[0]
            # We have (t*torch.exp(m)).sum() + torch.exp(s) + torch.exp(d) + torch.exp(l) = 1.0
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat
        else:
            complete_theta = torch.cat((zeros_tensor, theta), dim=-1)
            result = torch.log_softmax(complete_theta, dim=-1)
            log_pS = result[...,0]
            log_pD = result[...,1]
            log_pL = result[...,2]
            log_pT = result[...,3] # All have shape [N_sp]
            # transfer_mat has shape [N_sp, N_sp]
            # Each donor species i has row logits log_pT[i] + transfer_mat_unnormalized[i, :]
            log_transfer_mat = log_pT.unsqueeze(-1) + transfer_mat_unnormalized  # shape [N_sp, N_sp]
            max_transfer_mat = torch.max(log_transfer_mat, dim=-1, keepdim=True).values
            transfer_mat = torch.exp(log_transfer_mat - max_transfer_mat)
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat
    else:
        zeros_tensor = theta.new_zeros((1,))
        if pairwise:
            N_sp, _ = transfer_mat_unnormalized.shape
            complete_theta = torch.cat((zeros_tensor.unsqueeze(0).expand(N_sp, -1), theta.unsqueeze(0).expand(N_sp, -1), transfer_mat_unnormalized), dim=-1)
            result = torch.log_softmax(complete_theta, dim=-1)
            log_pS = result[...,0].squeeze(0)
            log_pD = result[...,1].squeeze(0)
            log_pL = result[...,2].squeeze(0)
            max_transfer_mat = torch.max(transfer_mat_unnormalized, dim=-1, keepdim=True).values
            transfer_mat = torch.exp(result[...,3:] - max_transfer_mat)
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat
        else:
            complete_theta = torch.cat((zeros_tensor, theta), dim=-1)
            result = torch.log_softmax(complete_theta, dim=-1)
            log_pS = result[...,0]
            log_pD = result[...,1]
            log_pL = result[...,2]
            log_pT = result[...,3] # shape []
            # transfer_mat has shape [N_sp, N_sp]
            # Scalar T distributed using provided unnormalized recipient logits per donor row
            log_transfer_mat = log_pT + transfer_mat_unnormalized  # shape [N_sp, N_sp]
            max_transfer_mat = torch.max(log_transfer_mat, dim=-1, keepdim=True).values
            transfer_mat = torch.exp(log_transfer_mat - max_transfer_mat)
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat


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