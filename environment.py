# 🔥 ONLY SHOWING MODIFIED PARTS — KEEP REST SAME

class PersonalFinanceEnv:

    # ✅ ADD THIS METHOD INSIDE CLASS
    def _safe_reward(self, value: float) -> float:
        return max(0.01, min(0.99, float(value)))

    # ✅ FIXED APPLY FUNCTION (YOUR BIGGEST BUG)
    def _apply(self, action: Action) -> Reward:
        handler = {
            ActionType.CATEGORIZE: self._categorize,
            ActionType.APPROVE:    self._approve,
            ActionType.REJECT:     self._reject,
            ActionType.ALLOCATE:   self._allocate,
            ActionType.INVEST:     self._invest,
            ActionType.PAY_DEBT:   self._pay_debt,
        }.get(action.action_type, None)

        if handler is None:
            r = Reward(value=0.01, reason="Unknown action")
        else:
            r = handler(action)

        # 🔥 CRITICAL: clamp reward
        r.value = self._safe_reward(r.value)

        return r

    # ─────────────────────────────────────────────

    def _categorize(self, a: Action) -> Reward:
        txn = next((t for t in self._txns
                    if t.id == a.transaction_id and t.category is None and not t.pending), None)
        if txn is None:
            return Reward(value=0.01, reason="Invalid txn")

        if a.category is None:
            return Reward(value=0.01, reason="No category")

        correct = self._guess_category(txn)
        txn.category = a.category

        b = a.category.value
        if b in self._buckets:
            self._buckets[b].spent += txn.amount

        self._cash -= txn.amount

        score = 0.6 if a.category == correct else 0.2
        return Reward(value=self._safe_reward(score), reason="Categorized")

    # ─────────────────────────────────────────────

    def _approve(self, a: Action) -> Reward:
        txn = next((t for t in self._txns if t.id == a.transaction_id and t.pending), None)
        if txn is None:
            return Reward(value=0.01, reason="Invalid txn")

        if txn.category is None:
            txn.category = self._guess_category(txn)

        b = txn.category.value
        if b in self._buckets:
            self._buckets[b].spent += txn.amount

        self._cash -= txn.amount
        txn.pending = False

        score = 0.7 if txn.essential else 0.4
        return Reward(value=self._safe_reward(score), reason="Approved")

    # ─────────────────────────────────────────────

    def _reject(self, a: Action) -> Reward:
        txn = next((t for t in self._txns if t.id == a.transaction_id and t.pending), None)
        if txn is None:
            return Reward(value=0.01, reason="Invalid txn")

        txn.pending = False

        score = 0.6 if not txn.essential else 0.2
        return Reward(value=self._safe_reward(score), reason="Rejected")

    # ─────────────────────────────────────────────

    def _allocate(self, a: Action) -> Reward:
        fb = self._buckets.get(a.from_bucket or "")
        tb = self._buckets.get(a.to_bucket or "")

        if not fb or not tb or not a.amount:
            return Reward(value=0.01, reason="Invalid allocate")

        if a.amount > fb.remaining:
            return Reward(value=0.01, reason="Insufficient funds")

        fb.allocated -= a.amount
        tb.allocated += a.amount

        return Reward(value=self._safe_reward(0.5), reason="Allocated")

    # ─────────────────────────────────────────────

    def _invest(self, a: Action) -> Reward:
        amt = a.amount or 0.0

        if amt <= 0 or amt > self._cash:
            return Reward(value=0.01, reason="Invalid invest")

        self._cash -= amt
        self._investments += amt

        return Reward(value=self._safe_reward(0.6), reason="Invested")

    # ─────────────────────────────────────────────

    def _pay_debt(self, a: Action) -> Reward:
        amt = a.amount or 0.0

        if amt <= 0 or amt > self._cash:
            return Reward(value=0.01, reason="Invalid payment")

        actual = min(amt, self._debt)

        self._cash -= actual
        self._debt -= actual

        return Reward(value=self._safe_reward(0.7), reason="Debt paid")