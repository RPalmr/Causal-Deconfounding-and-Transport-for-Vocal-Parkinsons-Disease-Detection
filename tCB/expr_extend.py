import copy
import grapl.expr as expr
import grapl.util as util

class LocalDOExpr(expr.Expr):
    def __init__(self, num=None, den=None, mrg=None, dov=None):
        super().__init__(num or [], den or [], mrg or set(), set())
        self._num_dovs = [set() for _ in self.num]
        self._den_dovs = [set() for _ in self.den]
        if dov:
            for dset in (self._num_dovs + self._den_dovs):
                dset.update(dov)

    def addvars(self, num=None, den=None, mrg=None, dov=None):
        if num is None: num = []
        if den is None: den = []
        if mrg is None: mrg = set()
        if dov is None: dov = set()

        n_num_before = len(self.num)
        n_den_before = len(self.den)

        super().addvars(num=num, den=den, mrg=mrg)

        n_num_added = len(self.num) - n_num_before
        n_den_added = len(self.den) - n_den_before

        self._num_dovs.extend([set(dov) for _ in range(n_num_added)])
        self._den_dovs.extend([set(dov) for _ in range(n_den_added)])

        assert len(self.num) == len(self._num_dovs)
        assert len(self.den) == len(self._den_dovs)

    def combine(self, exprs):
        for ex in exprs:
            if not isinstance(ex, LocalDOExpr):
                raise ValueError("Can only combine LocalDOExpr instances")
            ex_copy = copy.deepcopy(ex)
            self.num.extend(ex_copy.num)
            self._num_dovs.extend(ex_copy._num_dovs)
            self.den.extend(ex_copy.den)
            self._den_dovs.extend(ex_copy._den_dovs)
            self.mrg |= ex_copy.mrg
        return self

    def cancel(self):
        new_num, new_num_dov = [], []
        new_den, new_den_dov = [], []
        changed = False

        for term, dov in zip(self.den, self._den_dovs):
            new_den.append(term)
            new_den_dov.append(dov)

        for term, dov in zip(self.num, self._num_dovs):
            matched = False
            for i, (dterm, ddov) in enumerate(zip(new_den, new_den_dov)):
                if term == dterm and dov == ddov:
                    new_den.pop(i)
                    new_den_dov.pop(i)
                    matched = True
                    changed = True
                    break
            if not matched:
                new_num.append(term)
                new_num_dov.append(dov)

        if changed:
            self.num = new_num
            self._num_dovs = new_num_dov
            self.den = new_den
            self._den_dovs = new_den_dov

        return changed

    def marginal(self):
        can_marg = set()
        for node in list(self.mrg):
            if any(node in term for term in self.den):
                continue

            occ_idx = [i for i, term in enumerate(self.num)
                    if (node in term) and (node not in self._num_dovs[i])]

            if len(occ_idx) == 1:
                can_marg.add(node)

        if not can_marg:
            return False

        new_num, new_num_dov = [], []
        for term, dov in zip(self.num, self._num_dovs):
            stripped = set(term) - can_marg
            new_num.append(stripped)
            new_num_dov.append(dov)

        self.num = new_num
        self._num_dovs = new_num_dov
        self.mrg -= can_marg
        return True

    def simplify(self):
        changed = True
        any_change = False
        while changed:
            c1 = self.cancel()
            c2 = self.marginal()
            changed = c1 or c2
            any_change |= changed
        return any_change

    def tostr(self):
        expr_str = ''
        if self.mrg:
            expr_str += util.mrgstr(self.mrg)
        if len(self.num) > 1 and self.mrg:
            expr_str += '['
        for term, dov in zip(self.num, self._num_dovs):
            expr_str += util.probstr(term, do_nodes=dov)
        if self.den:
            expr_str += '/'
            if len(self.den) > 1:
                expr_str += '{'
            for term, dov in zip(self.den, self._den_dovs):
                expr_str += util.probstr(term, do_nodes=dov)
            if len(self.den) > 1:
                expr_str += '}'
        if len(self.num) > 1 and self.mrg:
            expr_str += ']'
        return expr_str

    def tocondstr(self):
        expr_str = ''
        if self.mrg:
            expr_str += util.mrgstr(self.mrg)
        if len(self.num) > 1 and self.mrg:
            expr_str += '['
        num_ord = list(zip(self.num, self._num_dovs))
        num_ord.sort(key=lambda x: len(x[0]), reverse=True)
        den_ord = list(zip(self.den, self._den_dovs))
        den_ord.sort(key=lambda x: len(x[0]), reverse=True)
        while num_ord:
            term, dov = num_ord.pop(0)
            assigned = False
            for j, (dterm, ddov) in enumerate(den_ord):
                if dterm.issubset(term) and dov == ddov:
                    den_ord.pop(j)
                    expr_str += util.probcndstr(term - dterm, cnd_nodes=dterm, do_nodes=dov)
                    assigned = True
                    break
            if not assigned:
                expr_str += util.probstr(term, do_nodes=dov)
        if den_ord:
            expr_str += '/'
            for term, dov in den_ord:
                expr_str += util.probstr(term, do_nodes=dov)
        if len(self.num) > 1 and self.mrg:
            expr_str += ']'
        return expr_str
