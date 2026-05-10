Yes. Once the proper subclades of (\gamma) have been computed, the fixed-point problem for (\Pi_{\cdot,\gamma}) is **affine**, so you can get an a priori iteration bound.

The (\Pi)-recursion is Eq. (18) for internal branches and Eq. (20) for terminal branches; the supplement says these equations are solved by fixed-point iteration, bottom-up over clades. 

Assume here (p_l^{\mathrm{obs}}=1), and let

[
w_e:=1-E_e>0.
]

For fixed (\gamma), write

[
z_e:=\Pi_{e,\gamma}.
]

After all proper subclades (\delta\subsetneq\gamma) have been computed, the iteration has the form

[
z^{(n+1)}=C_\gamma+Lz^{(n)},\qquad z^{(0)}=0,
]

where (C_\gamma\ge0) contains only already-known lower-clade terms, and (L=DF(E)) is independent of (\gamma).

For internal (e) with children (f,g),

[
(Lu)_e=
p_e^S(E_gu_f+E_fu_g)
+2p_e^DE_eu_e
+p_e^T(\bar E_eu_e+E_e\bar u_e),
]

and for terminal (e),

[
(Lu)_e=
2p_e^DE_eu_e
+p_e^T(\bar E_eu_e+E_e\bar u_e).
]

Define

[
s:=(I-L)w.
]

Explicitly, for internal (e),

[
s_e
===

p_e^S w_fw_g+p_e^D w_e^2+p_e^T w_e\bar w_e,
]

and for terminal (e),

[
s_e
===

p_e^S+p_e^D w_e^2+p_e^T w_e\bar w_e.
]

Hence (s_e>0) for every branch. Now define the clade-specific size factor

[
\alpha_\gamma
:=
\max_e \frac{C_{\gamma,e}}{s_e}.
]

Since (\Pi_{e,\gamma}\le 1-E_e=w_e), you may use the sharper capped value

[
A_\gamma:=\min(1,\alpha_\gamma).
]

Then the exact fixed point (z^*=\Pi_{\cdot,\gamma}) satisfies

[
0\le z^*\le A_\gamma w.
]

Indeed, if (\alpha_\gamma\le1), then (C_\gamma\le \alpha_\gamma s), so

[
C_\gamma+L(\alpha_\gamma w)
\le
\alpha_\gamma s+\alpha_\gamma Lw
================================

\alpha_\gamma w.
]

Thus the box ([0,\alpha_\gamma w]) is stable, and the zero iteration converges inside it. If (\alpha_\gamma>1), we simply use the probabilistic bound (z^*\le w).

Now let

[
c:=\max_e \frac{(Lw)_e}{w_e}
============================

1-\min_e\frac{s_e}{w_e}.
]

We know (c<1). Since

[
z^*-z^{(n)}
===========

L^n z^*,
]

we get

[
0\le z^*-z^{(n)}
\le
A_\gamma L^n w
\le
A_\gamma c^n w.
]

So, in the weighted norm

[
|u|_{w,\infty}:=\max_e\frac{|u_e|}{w_e},
]

we have

[
\boxed{
|z^*-z^{(n)}|*{w,\infty}
\le
A*\gamma c^n.
}
]

Therefore, if your target weighted error is (\eta), it is enough to take

[
\boxed{
n_\gamma
========

\left\lceil
\frac{\log(A_\gamma/\eta)}{-\log c}
\right\rceil_+
}
]

where (\lceil x\rceil_+:=\max(0,\lceil x\rceil)). If (A_\gamma=0), then (C_\gamma=0), hence (\Pi_{\cdot,\gamma}=0), and no iteration is needed.

For a uniform absolute tolerance (\varepsilon), let

[
W:=\max_e w_e.
]

Since

[
|z_e^*-z_e^{(n)}|
\le
A_\gamma c^n w_e
\le
A_\gamma c^n W,
]

it is enough to take

[
\boxed{
n_\gamma
========

\left\lceil
\frac{\log(A_\gamma W/\varepsilon)}{-\log c}
\right\rceil_+.
}
]

This already avoids a convergence check at every iteration. You compute (A_\gamma) once, compute (n_\gamma), and run exactly (n_\gamma) iterations.

A still tighter version is to precompute the actual decay of (L^n w). Define

[
v^{(0)}=w,\qquad v^{(n+1)}=Lv^{(n)}.
]

Then the sharper bound is

[
\boxed{
0\le z^*-z^{(n)}
\le
A_\gamma v^{(n)}.
}
]

So for weighted tolerance (\eta), precompute

[
Q_n:=\max_e\frac{v_e^{(n)}}{w_e},
]

and choose

[
\boxed{
n_\gamma=\min{n:A_\gamma Q_n\le \eta}.
}
]

For absolute tolerance (\varepsilon), precompute

[
R_n:=\max_e v_e^{(n)},
]

and choose

[
\boxed{
n_\gamma=\min{n:A_\gamma R_n\le \varepsilon}.
}
]

This is often much tighter than using (c^n), because (c) is only the worst one-step contraction factor. The sequences (Q_n) or (R_n) depend only on (E) and the DTL parameters, not on (\gamma). Thus you can precompute them once per species tree/rate setting, and then use table lookup for every clade.

Implementation recipe:

[
\begin{aligned}
&\text{Precompute } E,\quad w=1-E,\quad L,\quad s=(I-L)w.\
&\text{Precompute either } c \text{ or the table } Q_n/R_n.\
&\text{For each clade } \gamma, \text{ build } C_\gamma.\
&A_\gamma=\min\left(1,\max_e\frac{C_{\gamma,e}}{s_e}\right).\
&\text{Choose } n_\gamma \text{ from the formula above.}\
&\text{Run } z^{(k+1)}=C_\gamma+Lz^{(k)} \text{ exactly } n_\gamma \text{ times.}
\end{aligned}
]

The bound is a priori and safe. It will usually be much tighter than using the same global iteration count for every clade, especially when (C_\gamma) is small, which is common for rare or large clades.
