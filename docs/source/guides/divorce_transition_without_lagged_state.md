# Implementing a divorce/marriage transition without a lagged partner state

This note documents a working recipe, verified against an independent
hand-rolled EGM solver for both a two-period and a four-period model. The
full working code (model functions, reference solver, and the tests that
verify the equivalence claimed below) lives in
`tests/resources/divorce_model/` and `tests/test_divorce_toy_model.py`.

## The problem

You want: if the agent divorces between period `t` and `t+1`, they keep
half of what the household saved; if they marry, their new partner matches
their wealth and it doubles. That is a **transition-based** rule — it
depends on *both* `partner_state` at `t` and `partner_state` at `t+1`.

dcegm evaluates `budget_constraint` for the *child* state. Its
`partner_state` is the realization at `t+1`, after the partner transition, and
the only information about the parent is `lagged_choice` and the saved assets.
The law of motion can therefore condition on the new partner state but not on
the old one -- and halving or doubling needs both. Everything below works with
that child `partner_state` alone: when this note says "the current period's
`partner_state`", it means the state the budget is evaluated in, i.e. the
child's.

There are two ways to get this into dcegm:

1. **Add a `lagged_partner_state` to the state space** (a deterministic
   state carrying last period's `partner_state` forward, alongside the
   `lagged_choice` dcegm already tracks). With both `partner_state`
   (current) and `lagged_partner_state` (previous) visible inside
   `budget_constraint`, you can adjust the incoming asset directly and
   literally, on the transition. This works, and is the more obvious route
   -- at the cost of a genuine extra state dimension (bigger state space,
   more batches, more memory). This note does not walk through that model.
2. **Keep the state space as-is and solve it via internal bookkeeping on
   the individual level.** No extra state, no lagged variable. This is
   what's implemented and verified in `tests/resources/divorce_model/`.

Either way, the actual thing you need to get right is *how to correctly
rescale the asset given the transition*. Approach 2 answers that without
paying for a second state dimension, and that's what the rest of this note
covers.

## The Euler equation with partner transitions

**General case.** Let $W_t$ be wealth at the beginning of period $t$, $C_t$
consumption, $A_t = W_t - C_t$ saved assets and $p_t$ the partner state. Next
period's wealth follows a law of motion $W_{t+1} = g(A_t, p_t, p_{t+1}) +
y_{t+1}$. With

$$
V_t(W_t, p_t) = \max_{C_t} \; u(C_t) + \beta \, E_t\left[V_{t+1}(W_{t+1}, p_{t+1})\right],
$$

the first-order condition and the envelope condition $V'_{t+1}(W_{t+1}) =
u'(C_{t+1})$ give the general Euler equation

$$
u'(C_t) = \beta \, E_t\left[\frac{\partial W_{t+1}}{\partial A_t} \, u'(C_{t+1})\right].
$$

The return on saving is the derivative of the law of motion with respect to
saved assets. With discrete choices and taste shocks, $u'(C_{t+1})$ is the
expected marginal utility over next period's choice probabilities; nothing
below changes.

**Without partner transitions**, $g = A_t (1 + r)$, so $\partial W_{t+1} /
\partial A_t = 1 + r$ and

$$
u'(C_t) = \beta (1 + r) \, E_t\left[u'(C_{t+1})\right].
$$

This is the return dcegm uses.

**With partner transitions**, saved assets are halved on divorce and doubled
on marriage: $g = \kappa(p_t, p_{t+1}) \, A_t (1 + r)$ with
$\kappa(0, 1) = 2$ (marriage), $\kappa(1, 0) = \tfrac12$ (divorce) and
$\kappa = 1$ otherwise. Now $\partial W_{t+1} / \partial A_t = \kappa(p_t,
p_{t+1}) (1 + r)$ and

$$
u'(C_t) = \beta (1 + r) \sum_{p'} P(p' \mid p_t) \, \kappa(p_t, p') \,
E_t\left[u'(C_{t+1}) \mid p'\right].
$$

Spelled out, with marriage probability $\pi$ for a single agent and divorce
probability $\delta$ for a married one:

$$
\begin{aligned}
\text{single today:} \quad u'(C_t) &= \beta (1 + r) \left[(1 - \pi) \, E_t[u'(C_{t+1}) \mid \text{single}] + 2 \pi \, E_t[u'(C_{t+1}) \mid \text{married}]\right], \\
\text{married today:} \quad u'(C_t) &= \beta (1 + r) \left[(1 - \delta) \, E_t[u'(C_{t+1}) \mid \text{married}] + \tfrac12 \delta \, E_t[u'(C_{t+1}) \mid \text{single}]\right].
\end{aligned}
$$

A single agent saves more than without transitions, because every saved unit
doubles if they marry; a married agent saves less, because half of it is lost
on divorce. If the partner state cannot change ($\pi = 0$, $\delta = 0$), both
lines reduce to the equation without transitions.

**In individual units.** dcegm's recipe below tracks $w = W / s$, $c = C / s$
and $a = A / s$ with $s(p) = 2$ if partnered and $1$ otherwise. Then
$\kappa(p_t, p_{t+1}) = s_{t+1} / s_t$, and multiplying the Euler equation by
$s_t$ gives

$$
s_t \, u'(s_t c_t) = \beta (1 + r) \, E_t\left[s_{t+1} \, u'(s_{t+1} c_{t+1})\right].
$$

This is dcegm's Euler equation with return $1 + r$ and marginal utility
$m(c, p) = s(p) \, u'(s(p) \, c)$ -- the chain rule of the utility $u(s(p) \,
c)$. The budget below delivers the $1 + r$; the marginal utility delivers
$\kappa$.

## Why the naive version of approach 2 breaks

The first thing everyone tries: multiply the incoming asset by a
partner-status multiplier in `budget_constraint`, keyed off the *current*
`partner_state` (since that's all you have without the lagged state):

```python
def budget_constraint(period, lagged_choice, partner_state,
                       asset_end_of_previous_period, income_shock,
                       params, model_specs):
    multiplier = 2.0 if partner_state == 1 else 1.0
    return multiplier * asset_end_of_previous_period * (1 + params["interest_rate"]) + income
```

This computes the **wealth level** correctly. It silently breaks the
**Euler equation**. dcegm's Euler-equation solver hardcodes the marginal
return on savings as `1 + params["interest_rate"]`
(`src/dcegm/egm/solve_euler_equation.py:166`,
`rhs_euler = marginal_utility_next * (1 + interest_rate) * discount_factor`).
It never differentiates `budget_constraint` — it just assumes
`d(wealth)/d(asset_end_of_previous_period) = 1 + interest_rate`, always. If
your `budget_constraint` actually implies `d(wealth)/d(a) = multiplier *
(1+r)`, dcegm computes the consumption-savings tradeoff as if a dollar
saved always earns `1+r`, when for a partnered agent it actually earns
`2*(1+r)`. The wealth level is right; the price of saving used in the
policy function is wrong. This is invisible unless you check against an
independent solve — the model still runs, converges, produces a
"reasonable-looking" policy, just the wrong one.

## The recipe that works

**Two changes, applied together, both keyed off the *current* period's own
`partner_state` only:**

### 1. `budget_constraint`: double, then divide the whole thing by the same multiplier again

```python
def budget_constraint(period, lagged_choice, partner_state,
                       asset_end_of_previous_period, income_shock,
                       params, model_specs):
    multiplier = jnp.where(partner_state == 1, 2.0, 1.0)
    own_income = params["y_work"] * (lagged_choice == 0)
    partner_income = params["y_partner"] * (partner_state == 1)
    wealth = (
        multiplier * asset_end_of_previous_period * (1 + params["interest_rate"])
        + own_income
        + partner_income
    )
    return jnp.maximum(wealth, params["consumption_floor"]) / multiplier
```

For the asset term, `multiplier` cancels exactly:
`multiplier * a * (1+r) / multiplier = a * (1+r)`. So
`d(wealth)/d(a) = 1+r` regardless of `partner_state` — dcegm's hardcoded
assumption is now *actually true*, not just assumed. The income terms
still get divided by `multiplier`, i.e. partner income effectively gets
shared 50/50 through this division.

### 2. Utility: scale the consumption argument by the same multiplier

```python
def utility_func(consumption, choice, partner_state, params):
    scale = jnp.where(partner_state == 1, 2.0, 1.0)
    x = scale * consumption
    felicity = ((x ** (1 - params["rho"]) - 1) / (1 - params["rho"]))
    return felicity - (1 - choice) * params["delta"]
```

Consumption is tracked in *individual* terms (dcegm's own choice
variable), but if it's drawn from a jointly funded (pooled) account when
partnered, a dollar of individual consumption corresponds to two dollars of
joint spending — hence `scale * consumption` inside felicity.

`marginal_utility_func` and `inverse_marginal_utility_func` are the plain chain
rule of this utility: an individual dollar buys `scale` dollars of joint
consumption, so its marginal utility is `scale * felicity'(scale * c)`.

```python
def marginal_utility_func(consumption, partner_state, params):
    scale = jnp.where(partner_state == 1, 2.0, 1.0)
    x = scale * consumption
    return scale * x ** (-params["rho"])

def inverse_marginal_utility_func(marginal_utility, partner_state, params):
    scale = jnp.where(partner_state == 1, 2.0, 1.0)
    return (marginal_utility / scale) ** (-1 / params["rho"]) / scale
```

Do not drop the outer `scale`. The reason is the derivative of the
transition-based law of motion with respect to saved assets (see
`resources_after_transition` below):

```
d resources_{t+1} / d a_t = rescaling(p_t, p_{t+1}) * (1 + r)
rescaling = 2 on marriage, 1/2 on divorce, 1 otherwise
```

This is the return on saving, so the Euler equation of the transition-based
model is

```
u'(C_t) = beta * E[ d resources_{t+1} / d a_t * u'(C_{t+1}) ]
        = beta * (1 + r) * E[ rescaling(p_t, p_{t+1}) * u'(C_{t+1}) ]
```

(`d_resources_d_assets` in `tests/resources/divorce_model/reference.py`). In the
individual convention the budget's derivative is `1 + r` in every branch -- the
recipe's budget divides the rescaling back out -- so the rescaling has to come
from somewhere else: from marginal utility. With `C = scale * c`, dcegm's Euler
equation with the chain-rule marginal utility reads

```
scale_t * u'(C_t) = beta * (1 + r) * E[ scale_{t+1} * u'(C_{t+1}) ]
<=> u'(C_t)       = beta * (1 + r) * E[ (scale_{t+1} / scale_t) * u'(C_{t+1}) ]
```

and `scale_{t+1} / scale_t` is exactly `rescaling(p_t, p_{t+1})`. Without the
outer `scale` the derivative is lost: singles ignore that marriage doubles their
savings and save too little, married agents ignore that divorce halves them and
save too much. Whenever partner status cannot change, `rescaling = 1` and the
two versions coincide, which is why the error is invisible in models without
transitions.

## Why this reproduces the transition-based rule exactly

The natural worry: doesn't rescaling every period based on current status,
instead of only at the transition, change the economics? It doesn't. Write
out the transition-based resource function directly (this is the thing
approach 1, with the lagged state, would implement):

```python
def resources_after_transition(a, partner_state_0, partner_state_1, income, r):
    if partner_state_0 == 1 and partner_state_1 == 0:
        a = a / 2   # divorce
    elif partner_state_0 == 0 and partner_state_1 == 1:
        a = a * 2   # marriage
    return a * (1 + r) + income + partner_income(partner_state_1)
```

The following identity holds **exactly**, for every combination of
`partner_state_0`, `partner_state_1`:

```
resources_after_transition(scale(partner_state_0) * a, partner_state_0, partner_state_1, income)
    == scale(partner_state_1) * a * (1 + r) + income + partner_income(partner_state_1)
```

The right-hand side depends only on `partner_state_1` (and the raw `a`) —
`partner_state_0` drops out entirely. In words: feeding the transition-based
formula an asset pre-scaled by *today's* multiplier makes it collapse to
exactly the per-period, current-state-only formula. These are not two
different economic models; they are the same model with two different unit
conventions for what the state variable "individual wealth" means:

- **The per-period convention (recipe above):** `a` is always denominated
  such that its real-dollar value is `scale(current partner_state) * a`. A
  married person's `a=100` and a single person's `a=100` represent
  different real dollar amounts.
- **The transition-based convention (what a `lagged_partner_state` model
  would implement directly):** `a` is real dollars, always, and gets
  explicitly converted (halved/doubled) exactly at the moment
  `partner_state` changes.

Converting between the two conventions is bookkeeping (multiply/divide by
`scale`), not a different model. `tests/test_divorce_toy_model.py`'s
`test_dcegm_policy_matches_hand_solved_reference` verifies this to
floating-point precision for a two-period model by querying the
transition-based reference at `scale(partner_state_0) * a0_end` and
dividing the resulting policy back by `scale` — exactly the conversion
above.

**Practical upshot:** the internal-bookkeeping recipe is not a compromise
relative to the lagged-state model; it's an exact reformulation of the same
economics, without the extra state dimension.

## Extending to more than two periods

Two additional, genuine (not conceptual) numerical issues show up once
intermediate periods are no longer analytic (see `reference.py`'s
`solve_reference` / `continuation_value_and_marg_util` for the fixes, and
`test_dcegm_policy_matches_hand_solved_reference_n_periods` for the
verification, ~2e-3 max and ~5e-6 median relative error for a 4-period model):

1. **Borrowing constraint, bottom of the grid.** A solved period's
   endogenous grid only starts at its own `a_end=0` point's implied wealth
   — it does not cover `[0, endog_grid[0])`. A continuation-wealth query
   below that is not an edge case to `np.interp`-clamp away: the true
   optimal policy there is "consume everything, save nothing"
   (`policy = wealth`, exactly). dcegm handles this natively (its own
   solved arrays carry a duplicate natural-borrowing-constraint point at
   the bottom); if you hand-roll a reference solver for verification, you
   have to add the same handling explicitly or your reference will be
   silently wrong for a wide low-wealth range, not just approximately off.

2. **Extrapolation, top of the grid.** A marriage transition can double
   continuation wealth, which can push it *above* the range the
   continuation period's own grid was solved over (built for un-doubled,
   individual-scale wealth). There's no exact closed form here (unlike the
   borrowing constraint); linear extrapolation from the top two grid points
   is the standard, accurate-enough fix (`c(wealth)` is close to linear for
   large wealth under CRRA). Alternatively, simply build the exogenous
   asset grid wide enough that this rarely binds for your actual estimation
   sample — but don't rely on that alone without checking, since the
   multiplier can compound across consecutive partnered periods.

Both issues are symptoms of the exact same root cause: interpolating a
solved policy/value function outside the domain it was actually solved
over. They're not specific to the divorce mechanism, but the `2x` (or
`0.5x`) rescaling here makes them bite much sooner than in a standard
model, so they need explicit handling rather than assuming the default grid
is "wide enough."

## Checklist

1. Add `partner_state`-conditional `scale = 2 if partnered else 1` (or the
   real model's actual pooling assumption, if not a simple doubling) to
   `budget_constraint`'s asset term, and divide the *whole* wealth
   expression by the same `scale` again.
2. Use the plain chain rule for `marginal_utility_func` /
   `inverse_marginal_utility_func`, keeping the outer `scale` factor.
3. Decide deliberately between this approach and adding
   `lagged_partner_state` to the state space -- both are valid; this note's
   recipe avoids the extra state dimension, at the cost of the
   less-obvious derivation above.
4. If extending beyond two periods, check the asset grid's range against
   the maximum multiplier compounding you expect given the estimated
   marriage/divorce transition probabilities, and add explicit
   borrowing-constraint / extrapolation handling to any independent
   verification solver you write (dcegm handles both natively already).
5. Verify against an independent solve before trusting the result — this
   class of bug (wealth level right, Euler equation silently wrong)
   produces a model that runs and looks plausible without ever raising an
   error.
