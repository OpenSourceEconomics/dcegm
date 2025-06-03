import numpy as np

# --------------------------
# Setup: Grid & Interpolation Targets
# --------------------------

x0 = 7.17094712
x1 = 7.30803500
x_test = np.array([7.1925089, 7.26826418])  # interpolation points


# Standard linear interpolation
def linear_interp(x, x0, x1, v0, v1):
    return v0 + (x - x0) / (x1 - x0) * (v1 - v0)


# --------------------------
# Version "fues" (value_calc == value_expec)
# --------------------------

v0 = 2.50593035
v1_fues = 2.51011269  # same in both value_calc and value_expec

interp_fues = linear_interp(x_test, x0, x1, v0, v1_fues)

# Actual interpolated result from log:
interp_fues_logged = np.array([2.50658817, 2.50889934])  # from output
assert np.allclose(interp_fues, interp_fues_logged), "Mismatch in fues interpolation!"

# --------------------------
# Version "ue fedor" (value_calc differs slightly)
# --------------------------

v1_uefedor = 2.51034068  # changed value

interp_uefedor = linear_interp(x_test, x0, x1, v0, v1_uefedor)

# Actual interpolated result from log:
interp_uefedor_logged = np.array([2.50662403, 2.50906119])  # from output
assert np.allclose(
    interp_uefedor, interp_uefedor_logged
), "Mismatch in ue fedor interpolation!"

# --------------------------
# Difference Analysis
# --------------------------

delta_v1 = v1_uefedor - v1_fues
interp_weight = (x_test - x0) / (x1 - x0)
delta_interp = interp_uefedor - interp_fues

# --------------------------
# Print Results
# --------------------------

print("== Wealth Grid ==")
print(f"Grid:               [{x0:.8f}, {x1:.8f}]")
print(f"Test points:        {x_test}")

print("\n== Value Inputs ==")
print(f"v0:                 {v0}")
print(f"v1 (fues):          {v1_fues}")
print(f"v1 (ue fedor):      {v1_uefedor}")
print(f"Δv1:                {delta_v1:.8e}")

print("\n== Interpolation Results ==")
print(f"fues (manual):      {interp_fues}")
print(f"fues (logged):      {interp_fues_logged}")
print(f"ue fedor (manual):  {interp_uefedor}")
print(f"ue fedor (logged):  {interp_uefedor_logged}")
print(f"Δinterp (manual):   {delta_interp}")

print("\n== Interpolation Explanation ==")
print(
    f"""
The difference in interpolated values between 'ue fedor' and 'fues' comes from the upstream change in `value[1]`:

  Δv1 = {delta_v1:.8e}

This is propagated to the interpolated values using weights:
  weights = {interp_weight}

So we expect:
  Δinterp ≈ weights × Δv1:
    {interp_weight[0]:.4f} × {delta_v1:.8e} ≈ {delta_interp[0]:.8e}
    {interp_weight[1]:.4f} × {delta_v1:.8e} ≈ {delta_interp[1]:.8e}

✅ Therefore, this is **not** an interpolation or floating-point issue —
   it's caused by the slight change in the value input `v1`.
"""
)
