import copy as cp
import expr_extend as exprext
import grapl.eqn as eqn
import grapl.expr as expr

def remove_outgoing(G, node):
    for child in G.ch({node}):
        G.vars[child].parents = G.vars[child].parents.difference({node})
    G.vars[node].children = set()
    for bidirect in G.bi({node}):
        G.vars[bidirect].bidirects = G.vars[bidirect].bidirects.difference({node})
    G.vars[node].bidirects = set()
    return G

def remove_incoming(G, node):
    for parent in G.pa({node}):
        G.vars[parent].children = G.vars[parent].children.difference({node})
    G.vars[node].parents = set()
    for bidirect in G.bi({node}):
        G.vars[bidirect].bidirects = G.vars[bidirect].bidirects.difference({node})
    G.vars[node].bidirects = set()
    return G

def ismsep_set(G, source: set, target: set, cond = set()):
    
    for s in source:
        for t in target:
            if not G.ismsep(s, t, cond):
                return False
    return True

def isdsep_set(G, source: set, target: set, cond = set()):
    for s in source:
        for t in target:
            if not G.isdsep(s, t, cond):
                return False
    return True

def sid(Y, X, S, D):
    Z = set()
    P_star = exprext.LocalDOExpr()  
    P_star.addvars(num = D.nodes().difference(S))
    D_original = cp.deepcopy(D)
    
    def sid_recur(Y, X, Z, S, P_star, D, D_original):
        
        V = D.nodes()
        if S.issubset(V):
            V = V.difference(S)
        G = cp.deepcopy(D)
        G = G.sub(V)
        
        if len(X) == 0:
            VdY = V.difference(Y)
            P_star_copy = cp.deepcopy(P_star)
            P_star_copy.addvars(mrg = VdY.union(P_star_copy.mrg))
            P_star_copy.simplify()
            return P_star_copy, True

        Y_an_G = G.an(Y)
        VdY_an_G = V.difference(Y_an_G)
        if len(VdY_an_G) > 0:
            P_star_copy = cp.deepcopy(P_star)
            P_star_copy.addvars(mrg = VdY_an_G.union(P_star_copy.mrg))
            G_sub_Y_an_G = cp.deepcopy(G)
            G_sub_Y_an_G = G_sub_Y_an_G.sub(Y_an_G)
            return sid_recur(Y, X.intersection(Y_an_G), Z, S, P_star_copy, G_sub_Y_an_G, D_original)
        
        VdX = V.difference(X)
        G_doX = cp.deepcopy(G)
        for x in X:
            G_doX = remove_incoming(G_doX, x)
        Y_an_G_doX = G_doX.an(Y)
        W = VdX.difference(Y_an_G_doX)
        if len(W) > 0:
            return sid_recur(Y, X.union(W), Z, S, P_star, G, D_original)
        
        G_sub_dX = cp.deepcopy(G)
        G_sub_dX = G_sub_dX.sub(VdX)
        C = G_sub_dX.districts()
            
        if len(C) > 1:
            I_star_exprs = []
            for c in C:
                P_star_copy = cp.deepcopy(P_star)
                I_star_expr_c, c_sidfixable = sid_recur(c, V.difference(c), Z, S, P_star_copy, G, D_original)
                if c_sidfixable:
                    I_star_mrg = list(I_star_expr_c.mrg)
                    for mrg_var in I_star_mrg:
                        new_var = mrg_var + chr(39)
                        I_star_expr_c.subsvar(mrg_var, new_var)
                    I_star_expr_c.simplify()
                    I_star_exprs.append(I_star_expr_c)
                else:
                    return None, False
            I_star_expr = exprext.LocalDOExpr()
            I_star_expr.addvars(mrg = V.difference(Y.union(X)))
            I_star_expr.combine(tuple(I_star_exprs))
            I_star_expr.simplify()
            return I_star_expr, True

        if len(C) == 1:
            
            D_doX = cp.deepcopy(D_original)
            for x in X:
                D_doX = remove_incoming(D_doX, x)

            S_ind_Y_on_X = ismsep_set(D_doX, Y, S, X)
            S_ind_Y_on_XandZ = ismsep_set(D_doX, Y, S, X.union(Z))
            if S_ind_Y_on_X:
                py_dox = exprext.LocalDOExpr()
                py_dox.addvars(num = Y, dov = X)
                return py_dox, True
            elif S_ind_Y_on_XandZ:
                py_dox = exprext.LocalDOExpr()
                py_dox.addvars(num = Y.union(Z), den = Z, dov = X)
                return py_dox, True
            
            if len(G.districts()) <= 1:
                return None, False
            
            if C[0] in G.districts():
                mrg_vars = C[0].difference(Y)
                expr_c0 = exprext.LocalDOExpr() 
                expr_c0.addvars(mrg = mrg_vars)
                topo_ordering = G.topsort()
                for v_i in C[0]:
                    vi_topo_index = topo_ordering.index(v_i)
                    vi_topo_procd = set(topo_ordering[: vi_topo_index])
                    den_term = vi_topo_procd
                    num_term = {v_i}.union(den_term)
                    expr_c0.addvars(num = num_term, den = den_term)
                    expr_c0.simplify()
                expr_c = exprext.LocalDOExpr()
                expr_c.addvars(mrg = mrg_vars)
                for num in expr_c0.num:
                    mrg_vars_num = V.difference(num)
                    P_star_copy = cp.deepcopy(P_star)
                    P_star_copy.addvars(mrg = mrg_vars_num)
                    P_star_copy.simplify()
                    expr_c.combine((P_star_copy,))
                for den in expr_c0.den:
                    mrg_vars_den = V.difference(den)
                    P_star_copy = cp.deepcopy(P_star)
                    P_star_copy.addvars(mrg = mrg_vars_den)
                    P_star_copy.simplify()
                    P_star_copy_inv = exprext.LocalDOExpr(num=P_star_copy.den, den=P_star_copy.num, mrg=P_star_copy.mrg)
                    P_star_copy_inv.simplify()
                    expr_c.combine((P_star_copy_inv,))
                expr_c.simplify()
                return expr_c, True
            
            c0_in_c_flag = False
            for c in G.districts():
                if C[0].issubset(c):
                    c0_in_c_flag = True
                    c_prime = c
                    break
            if c0_in_c_flag:
                c_expr = exprext.LocalDOExpr()
                topo_ordering = G.topsort()
                for v_i in c_prime:
                    vi_topo_index = topo_ordering.index(v_i)
                    vi_topo_procd = set(topo_ordering[: vi_topo_index])
                    vi_topo_procd_and_c_prime = vi_topo_procd.intersection(c_prime)
                    vidc_prime = vi_topo_procd.difference(c_prime)
                    den_term = vi_topo_procd_and_c_prime.union(vidc_prime)
                    num_term = {v_i}.union(den_term)
                    c_expr.addvars(num = num_term, den = den_term)
                G_sub_c_prime = cp.deepcopy(G)
                G_sub_c_prime = G_sub_c_prime.sub(c_prime)
                return sid_recur(Y, X.intersection(c_prime), X.difference(c_prime), S, c_expr, G_sub_c_prime, D_original)  
    
    rhs, transportable = sid_recur(Y, X, Z, S, P_star, D, D_original)
    if not transportable:
        print("Not transportable")
        return None
    lhs = expr.Expr()
    lhs.addvars(num=Y, dov=X)
    transport_formula = eqn.Eqn(lhs, rhs)
    return transport_formula
