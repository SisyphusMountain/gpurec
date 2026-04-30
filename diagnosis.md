Now I have enough context to give precise answers to both questions.                                        
                                                                                                              
  ---                                                                                                         
  1. Why AleRax hits 1e-10 but we don't                                                                       
                                                                                                              
  Different parameterizations. AleRax optimizes D, L, T directly as rates, with a hard lower bound of 1e-10   
  enforced by L-BFGS-B. When the gradient says "decrease D", the optimizer literally walks to 1e-10 and stops 
  there.                                                                                                      
                                                                                                              
  We use a softmax parameterization: complete_theta = [0, theta_D, theta_L, theta_T], then log2_softmax over  
  it. So theta_D = log2(D_rate_unnormalized), and the floor _THETA_MIN = log2(1e-10) ≈ -33.2. The NLL is      
  essentially flat between theta_D = -25 (rate ≈ 3e-8) and theta_D = -33.2 (rate = 1e-10) — the difference is 
  well below float32 precision (7 significant digits). So the optimizer stops at whatever tiny value the      
  gradient first rounds to zero, which might be -20 or -25 rather than -33.2. The absolute rate difference is 
  negligible (both are "effectively zero"), but the relative difference (3e-8 vs 1e-10) is 300x = 32,000%.    
                                                                                                              
  This is not a correctness problem — it's a float32 precision floor effect on degenerate parameters. The     
  actual NLL difference is unrepresentable in float32.                                                        
                                                                                                              
  ---                                                                                                         
  2. Where NaN comes from
                         
  Two distinct sources:
                                                                                                              
  Source A — specieswise NaN at eval ~26: _Log2Softmax itself has the comment "Custom backward avoids NaN for 
  very negative inputs (e.g. theta = -33 in fp32)". The custom backward is protected, but the NaN is coming   
  from the forward pass at those parameter values. Looking at extract_parameters_uniform:                     
                                                                                                            
  max_transfer_mat = log_pT + unnorm_row_max  # [S]

  unnorm_row_max is computed once at dataset initialization and never updated. As optimization progresses and 
  some species' T rates drop toward 0, log_pT[s] → -33. The max_transfer_mat[s] becomes very negative. Inside
  E_step or Pi_wave_forward, there are logsumexp operations that add and subtract these, and for certain      
  combinations of species (particularly when one species is simultaneously a donor and recipient with       
  near-zero rates), the computation produces (-inf) + (+inf) = NaN.

  Source B — genewise NaN in batched run: Family 2's theta converges to T≈0. In the batched L-BFGS, the       
  Hessian approximation is built from gradient history across all steps. Once family 2's T gradient rounds to
  0 (at the boundary), the L-BFGS two-loop recursion uses stale curvature information and proposes a step that
   is valid for families 0,1,3,4 but overshoots for family 2 — landing theta at a point where the forward pass
   (for that specific family's extreme parameter values) produces NaN. In isolation, the single-family L-BFGS
  builds its own Hessian and handles the boundary correctly.

  The common underlying cause: float32 gradients become zero at the parameter floor before the optimizer      
  formally satisfies its convergence criterion, which confuses the Hessian approximation and triggers
  pathological steps.             