"""
Defines the symbols s, t and K, which are used and shared throughout the
package.
"""
import sympy as sp

s = sp.Symbol('s', complex=True)
t = sp.Symbol('t', real=True, positive=True)
K = sp.Symbol('K', real=True, positive=True)
