import sID_expr_moreDo as sid_alg_keepDo
import sID_expr as sid_alg
import grapl.algorithms as algs
import grapl.dsl as dsl
import expr_extend

test_graph = '"example 1(a)"; \
    X; Y; Z; S;\
    S -> Z; \
    Z -> Y; \
    Z -> X; \
    X -> Y; \
    X <-> Y; '
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = {"S"}, D = test_graph)
I_star

test_graph = '"example 1(c)"; \
              X; Y; Z; S; \
              S -> Z; \
              X -> Z; \
              Z -> Y; \
              X -> Y; \
              X <-> Y; '     
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = {"S"}, D = test_graph)
I_star
test_graph = '  Y; X; V; Z; W; S;\
                X -> Z; \
                Z -> W; \
                W -> V; \
                V -> Y; \
                S -> V; \
                X <-> Y; \
                V <-> Y; \
                X <-> Z; '              
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)

Y = {"V", "Y"}
X = {"X", "Z", "W"}
S = {"S"}

I_star = sid_alg.sid(Y = Y, X = X, S = S, D = test_graph)
I_star

id_str, id_eqn, identifiable = algs.idfixall(G = test_graph, Y = Y, X = X, mode = "all")
id_eqn

Y = {"W"}
X = {"X", "Z", "V", "Y"}
S = {"S"}

I_star = sid_alg.sid(Y = Y, X = X, S = S, D = test_graph)
I_star

id_str, id_eqn, identifiable = algs.idfixall(G = test_graph, Y = Y, X = X, mode = "all")
id_eqn

Y = {"Z"}
X = {"X", "W", "V", "Y"}
S = {"S"}

I_star = sid_alg.sid(Y = Y, X = X, S = S, D = test_graph)
I_star

test_graph = '"frontdoor example"; \
             S; Z; X; Y; \
                S -> Z; \
                Z -> X; \
                Z -> Y; \
                X -> Y; \
                X <-> Y; \
                Z <-> X;'              
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)

I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = {"S"}, D = test_graph)
I_star
test_graph = '"test sel diagram 4(e)"; \
             S; Z; X; Y; \
                S -> X; \
                Z -> X; \
                Y -> Z; \
                Z <-> Y;'     
                
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)

I_star = sid_alg.sid(Y = {"X"}, X = {"Y"}, S = {"S"}, D = test_graph)
I_star   

test_graph = '"test admg sel"; \
            X; S; W; Z; Y;\
            Y -> Z; \
            Z -> W; \
            S -> X; \
            W -> X; \
            Y <-> Z; \
            X <-> Y;'      
           
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)

test_graph.ismsep("W", "Y", {"Z"})

I_star = sid_alg_keepDo.sid(Y = {"X"}, X = {"Y"}, S = {"S"}, D = test_graph)
I_star
test_graph = '"frontdoor example"; \
             S; Z; X; Y; \
                S -> X; \
                Z -> X; \
                Y -> Z; \
                X <-> Y;'              
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
test_graph.ismsep("S", "Y")

test_graph = '"eicu_expr"; \
    Demog; Sep; preSigns; Measure; ObsIn; S1; S2;\
    Demog -> Measure; \
    Demog -> preSigns; \
    Sep -> preSigns; \
    Sep -> ObsIn; \
    Sep -> Measure; \
    ObsIn -> Measure; \
    preSigns -> Measure; \
    Sep <-> preSigns; \
    S1 -> Measure; \
    S2 -> Demog;'

grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Measure"}, X = {"Sep"}, S = {"S1", "S2"}, D = test_graph)
I_star
test_graph = '"mimic-iv"; \
    Demog; Sep; preSigns; Measure; ObsIn; Adm; \
    Demog -> Measure; \
    Demog -> preSigns; \
    Adm -> Sep; \
    Adm -> preSigns; \
    Sep -> preSigns; \
    Sep -> ObsIn; \
    Sep -> Measure; \
    ObsIn -> Measure; \
    preSigns -> Measure; \
    '
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
X = {"Sep"}
Y = {"preSigns", "Demog"}
id_eqn, identifiable = sid_alg.sid(D = test_graph, Y = Y, X = X, S =set())
id_eqn
X = {"Sep"}
Y = {"Demog"}
id_str, id_eqn, identifiable = algs.idfixall(G = test_graph, Y = Y, X = X, mode = "shortest")
id_eqn
X = {"Sep"}
Y = {"Measure"}
id_eqn, identifiable = sid_alg.sid(D = test_graph, Y = Y, X = X, S =set())
id_eqn
test_graph = '"eicu_expr"; \
    Demog; Sep; Measure; ObsIn; Sofa; S2; S1;\
    Demog -> Measure; \
    Demog -> Sep; \
    Demog -> ObsIn; \
    Sep -> ObsIn; \
    Sep -> Sofa; \
    ObsIn -> Measure; \
    Sofa -> Measure; \
    Sofa <-> Sep; \
    S2 -> Demog; \
    S1 -> Measure; '

grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star= sid_alg.sid(Y = {"Measure"}, X = {"Sep"}, S = {"S2", "S1"}, D = test_graph)
I_star
test_graph = '"eicu_expr"; \
    Demog; Sep; Measure; ObsIn; S2; S1;\
    Demog -> Measure; \
    Demog -> Sep; \
    Demog -> ObsIn; \
    Sep -> ObsIn; \
    ObsIn -> Measure; \
    Measure <-> Sep; \
    Sep <-> ObsIn; \
    S2 -> Demog; \
    S1 -> Measure; '

grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg_keepDo.sid(Y = {"Measure"}, X = {"Sep"}, S = {"S2", "S1"}, D = test_graph)
I_star

test_graph = 'X; U; Y; Z; V; W; \
              X->U;  \
              W->X;  \
              W->U; \
              U->Z;  \
              Z->Y;  \
              V->U;  \
              V<->Z; \
              X<->Y; '
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
test_graph.an("Z")
test_graph = '"frontdoor example"; \
                Z; X; Y; \
                Z -> Y; \
                X -> Z; \
                X <-> Y;'              
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star

test_graph = '"frontdoor example"; \
                Z; X; Y; W; \
                W -> Y; \
                X -> Z; \
                Z -> W; \
                X <-> Z; \
                X <-> Y;'              
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X", "Z"}, S = set(), D = test_graph)
I_star
test_graph = '"frontdoor example"; \
                Z; X; Y; W; M; V; U; \
                X -> M; \
                M -> Y; \
                Y -> Z; \
                Y -> W; \
                Z -> W; \
                Z -> V; \
                Z -> U; \
                U <-> V; \
                X <-> Z; \
                X <-> Y;'              
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star
test_graph = '"frontdoor example"; \
                Z; X; Y; W; M; \
                X -> Z; \
                Z -> M; \
                M -> W; \
                W -> Y; \
                X <-> M; \
                X <-> Y;'              
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X", "M", "Z"}, S = set(), D = test_graph)
I_star
test_graph = '"frontdoor example"; \
                Z; X; Y; W; V; M; \
                X -> Z; \
                Z -> Y; \
                V -> X; \
                V -> W; \
                X -> W; \
                Y -> M; \
                X <-> M; \
                Y <-> V; \
                X <-> Y;'              
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
test_graph.de("V")
test_graph = '"frontdoor example"; \
                Z; X; Y; W; V; \
                X -> Z; \
                Z -> Y; \
                V -> W; \
                W -> Y; \
                V <-> X; \
                X <-> Y; '              
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star

test_graph = 'Z; X; Y; W; \
               Z -> Y; \
                X -> Z; \
                W -> Z; \
                X <-> Y;' 

grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star
test_graph = 'Z; X; Y; V; U;\
               X -> U; \
               U -> Z; \
               Z -> Y; \
               V -> U; \
               V <-> Z; \
               X <-> Y;' 
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star

test_graph = 'Z; X; Y; V; U;\
               X -> U; \
               U -> Z; \
               Z -> Y; \
               V -> Z; \
               V -> Y; \
               X <-> Y;' 
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star

test_graph = 'Z; X; Y; V; U; W;\
               X -> U; \
               U -> Z; \
               Z -> Y; \
               W -> U; \
               W -> V; \
               V -> Y; \
               W <-> X; \
               Z <-> Y; \
               X <-> Y;' 
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)               
I_star = sid_alg.sid(Y = {"Y"}, X = {"X", "W"}, S = set(), D = test_graph)
I_star

test_graph = 'Z; X; Y; V; U; W;\
               X -> U; \
               U -> Z; \
               Z -> Y; \
               X -> V; \
               V -> U; \
               V -> Y; \
               W -> Z; \
               W <-> Y; \
               X <-> Y;' 
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)               
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star
test_graph = 'Z; X; Y; V; U;\
               X -> U; \
               U -> V; \
               V -> Z; \
               Z -> Y; \
               Z <-> Y; \
               X <-> U; \
               X <-> Y;' 
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X", "U"}, S = set(), D = test_graph)
I_star
test_graph = 'Z; X; Y; W; \
               Z -> Y; \
                X -> Z; \
                X -> W; \
                Y -> W; \
                X <-> Y;' 

grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star
test_graph = 'Z; X; Y; W; V; \
               Z -> Y; \
                X -> Z; \
                X -> W; \
                V -> Z; \
                W -> V; \
                X <-> Y;' 

grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star
test_graph = 'Z; X; Y; W; V; M; \
               Z -> Y; \
               X -> Z; \
               X -> W; \
               V -> Z; \
               W -> V; \
               M -> W; \
               M <-> Y; \
               X <-> Y;' 

grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star
test_graph = 'Z; X; Y; W; V; U; \
               X->U; \
               U->Z; \
               Z->Y; \
               X->V; \
               V->W; \
               Y->W; \
               U<->V; \
               X<->Y;'

grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star

test_graph = 'Z; X; Y; V; U; \
               X->U; \
               U->Z; \
               Z->Y; \
               V->Z; \
               V->Y; \
               X<->Z;'

grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star

test_graph = 'M; X; Y; V; U;\
               X -> U; \
               U -> M; \
               M -> Y; \
               V -> U; \
               V <-> M; \
               X <-> Y;' 
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star

test_graph = 'M; X; Y; V; Z; \
               X->M; \
               M->Y; \
               V->Z; \
               Z->Y; \
               X<->V; \
               X<->Y;'

grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X", "V"}, S = set(), D = test_graph)
I_star

test_graph = 'M; X; Y; V; U; \
              X->M; \
              M->Y; \
              U->V; \
              Y->U; \
              X<->V; \
              X<->Y; \
              Y<->U;'

grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X", "V"}, S = set(), D = test_graph)
I_star

test_graph = 'M; X; Y; V; U;\
               X -> U; \
               X -> V; \
               U -> M; \
               M -> Y; \
               V -> U; \
               V -> M; \
               X <-> Y;' 
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star

test_graph = 'M; X; Y; Z; U;\
               X -> M; \
               M -> Y; \
               Z -> M; \
               Z -> U; \
               X <-> Y; \
               U <-> Y;' 
grapl_obj = dsl.GraplDSL()
test_graph = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"Y"}, X = {"X"}, S = set(), D = test_graph)
I_star
                              
graph ='Y; H; G; S; X; D; Sel1; Sel2;\
        Y -> H; \
        H -> G; \
        G -> X; \
        H -> S; \
        Y -> S; \
        S -> G; \
        D -> Y; \
        D -> H; \
        D -> G; \
        D -> X; \
        D -> S; \
        Sel1 -> X; \
        Sel2 -> G; \
        X <-> Y; ' 
grapl_obj = dsl.GraplDSL()
G = grapl_obj.readgrapl(graph)
I_star = sid_alg.sid(Y = {"X"}, X = {"Y"}, S = {"Sel1", "Sel2"}, D = G)
I_star
test_graph = '  M; X; Y; D; H; S;\
                Y -> M; \
                M -> X; \
                Y -> H; \
                H -> M; \
                M -> S; \
                S -> X; \
                D -> Y; \
                D -> X; \
                X <-> Y;'
grapl_obj = dsl.GraplDSL()
G = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"X"}, X = {"Y"}, S = set(), D = G)
I_star
test_graph = '  Y; Z; W; X; S1; S2; \
                Y -> Z; \
                Z -> W; \
                W -> X; \
                S1 -> X; \
                S2 -> W; \
                Y <-> X; \
                Y <-> Z; '
grapl_obj = dsl.GraplDSL()
G = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"X"}, X = {"Y"}, S = {"S1", "S2"}, D = G)
I_star

test_graph = '  Y; Z; W; X; S; \
                Y -> Z; \
                Z -> W; \
                W -> X; \
                S -> X; \
                Y <-> X; \
                Y <-> Z; '
grapl_obj = dsl.GraplDSL()
G = grapl_obj.readgrapl(test_graph)
I_star = sid_alg.sid(Y = {"X"}, X = {"Y"}, S = {"S"}, D = G)
I_star
