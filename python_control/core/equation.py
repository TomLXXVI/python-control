import sympy as sp


def solve(
    eq: sp.Expr,
    unknown: sp.Symbol,
    domain: sp.Set = sp.S.Complexes
) -> sp.Expr:
    """
    Solve an equation *f(x) = 0* algebraically for the unknown variable *x* in
    the given domain.
    """
    sol = sp.solveset(eq, unknown, domain=domain)
    if isinstance(sol, sp.FiniteSet):
        sol = list(sol)
        try:
            return sol[0]
        except IndexError:
            raise ValueError('no solution')
    else:
        raise ValueError(f'the equation has no single solution for {unknown}')
