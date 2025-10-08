from more_itertools import pairwise
import torch
from ..core.likelihood import compute_log_likelihood, E_step 
from ..core.extract_parameters import extract_parameters


"""
We write the fixed point equation as:
(Pi, E) = H(Pi, E, theta) = (F(Pi, E, theta), G(E, theta))
By differentiating we get
dE* = dG/dE * E* + dG/dtheta * dtheta
so that 
dE* / dtheta = (I - dG/dE)^(-1) * dG/dtheta

Then for Pi we have
dPi* = dF/dPi * dPi* + dF/dE * dE* + dF/dtheta * dtheta
so that
dPi* / dtheta = (I - dF/dPi)^(-1) * (dF/dE * dE* + dF/dtheta)

We can obtain the inverses using Neumann series, because
F and G are contractions so their spectral radius is < 1.

We first obtain dE*/dtheta, then dPi*/dtheta, and finally
dL/dtheta = dL/dPi * dPi*/dtheta + dL/dE * dE*/dtheta

We compute the product from left to right (vector-Jacobian product) which never requires
materializing the Jacobian.
"""


# first obtain dL/dPi and dL/dE
def compute_dL_dPi_E(Pi, E, root_clade_idx):
    Pi_cp = Pi.clone().detach().requires_grad_(True)
    E_cp = E.clone().detach().requires_grad_(True)
    nlll = compute_log_likelihood(Pi_cp, E_cp, root_clade_idx)
    mean_nlll = nlll.mean()
    mean_nlll.backward()
    return Pi_cp.grad, E_cp.grad


def apply_G_to_E(E, sp_P_idx, sp_child12_idx, log_pS, log_pD, log_pL, transfer_mat):
    function = lambda E: E_step(E, sp_P_idx, sp_child12_idx, log_pS, log_pD, log_pL, transfer_mat)
    return E, function(E)

def compute_dG_dE(v, E, GE):
    # output should have been computed as output = G(E)
    # and input vectors should have gradients enabled
    E_grad = torch.autograd.grad(outputs=GE,
                                    inputs=(E,),
                                    grad_outputs=v,
                                    retain_graph=True,
                                    materialize_grads=True)
    return E_grad

def compute_dG_dtheta(v, log_pS, log_pD, log_pL, transfer_mat, Gtheta):
    grads = torch.autograd.grad(outputs=Gtheta,
                                    inputs=(log_pS, log_pD, log_pL, transfer_mat),
                                    grad_outputs=v,
                                    materialize_grads=True)
    return grads

# now we can obtain dE*/dtheta
def compute_dEstar_dtheta(E, log_pS, log_pD, log_pL, transfer_mat, dG_dtheta, max_iters=10):
    pass

# Let's obtain dPi*/dtheta

def compute_dF_dPi(v, Pi, FPi):
    grads = torch.autograd.grad(outputs=FPi,
                                    inputs=(Pi,),
                                    grad_outputs=v,
                                    retain_graph=True,
                                    materialize_grads=True)
    return grads

def compute_dF_dE(v, E, FE):
    grads = torch.autograd.grad(outputs=FE,
                                    inputs=(E,),
                                    grad_outputs=v,
                                    materialize_grads=True)
    
    return grads

def compute_dF_dtheta(v, theta, transfer_mat, Ftheta, genewise, specieswise, pairwise):
    """dF/dtheta when the transfer matrix has trainable parameters"""
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(theta=theta,
                                                                            transfer_mat_unnormalized=transfer_mat,
                                                                            genewise=genewise,
                                                                            specieswise=specieswise,
                                                                            pairwise=pairwise,)
    grads = torch.autograd.grad(outputs=Ftheta,
                                    inputs=(log_pS, log_pD, log_pL, transfer_mat),
                                    grad_outputs=v,
                                    materialize_grads=True)
    return grads


def compute_dL_dtheta():
    pass



