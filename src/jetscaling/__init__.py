"""jetscaling -- replication of jet-tagging neural scaling laws.

Module map (build order matches docs/orientation.md section 6):

    data.py       HDF5 -> features + mask; the D-subsampling machinery
    model.py      the one set-transformer recipe, parameterised by size only
    train.py      one config in -> one immutable results JSON out
    sweep.py      the (N, D) grid from configs/grid.yaml
    fit.py        L(N,D) = L_inf + A/N^a + B/D^b, Huber on log residuals + bootstrap
    costmodel.py  C = 6ND + kD + mN analytics (no GPU needed)

Nothing in this package reads a hyperparameter from anywhere but configs/.
"""

__version__ = "0.1.0"
