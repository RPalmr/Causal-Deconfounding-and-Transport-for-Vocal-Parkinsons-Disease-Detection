import sID_expr_moreDo as sid_alg_keepDo
import sID_expr as sid_alg
import grapl.algorithms as algs
import grapl.dsl as dsl

test_graph =   'Y; D; M; X; H; S; Sel;\
                Y -> M; \
                M -> X; \
                D -> Y; \
                D -> X; \
                Y -> H; \
                H -> M; \
                M -> S; \
                S -> X; \
                Sel -> X; \
                X <-> Y; ' 
Y = {"X"}
X = {"X"}
S = {"Sel"}

grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)

I_star = sid_alg.sid(Y = Y, X = X, S = S, D = test_graph)
I_star
