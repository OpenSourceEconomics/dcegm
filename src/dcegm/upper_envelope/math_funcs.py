from typing import Tuple

import jax.numpy as jnp


def calc_intersection_and_extrapolate_policy(
    wealth_1_lower_curve: float | jnp.ndarray,
    value_1_lower_curve: float | jnp.ndarray,
    policy_1_lower_curve: float | jnp.ndarray,
    wealth_2_lower_curve: float | jnp.ndarray,
    value_2_lower_curve: float | jnp.ndarray,
    policy_2_lower_curve: float | jnp.ndarray,
    wealth_1_upper_curve: float | jnp.ndarray,
    value_1_upper_curve: float | jnp.ndarray,
    policy_1_upper_curve: float | jnp.ndarray,
    wealth_2_upper_curve: float | jnp.ndarray,
    value_2_upper_curve: float | jnp.ndarray,
    policy_2_upper_curve: float | jnp.ndarray,
):
    """Return the intersection points on endog grid, policy, and value functions.

    Calculates the intersection points on the wealth grid, the left and right policy
    as well as the value function.

    Even though the function is described for scalars it also works for arrays.

    We introduce here the left and right policy function to avoid inserting two wealth
    points next to each other in comparison to the original upper_envelope. The left and
    right policy arrays coincide on all entries except the entry corresponding to the
    intersection point on the wealth grid. We use left and right as lower and higher
    on the line of all real numbers. The left policy function holds the extrapolated
    policy function value from the two points left to the intersection. We use this
    value if we interpolate the policy function for a wealth point between the wealth
    point to the left of the intersection point and the intersection point. vice versa,
    the right policy function value holds the extrapolated policy function value from
    the two points right to the intersection point, and we use it to interpolate the
    policy if a wealth point is to the right of the intersection.

    Our interpolation function incorporates this structure and benefits from the fact
    that value and policy functions use the same wealth grid. More information is
    provided in the interpolation module.

    Args:
        wealth_1_lower_curve (float): The first wealth point on the lower curve.
        value_1_lower_curve (float): The value function at the first wealth point on the
            lower curve.
        policy_1_lower_curve (float): The policy function at the first wealth point on
            the lower curve.
        wealth_2_lower_curve (float): The second wealth point on the lower curve.
        value_2_lower_curve (float): The value function at the second wealth point on
            the lower curve.
        policy_2_lower_curve (float): The policy function at the second wealth point on
            the lower curve.
        wealth_1_upper_curve (float): The first wealth point on the upper curve.
        value_1_upper_curve (float): The value function at the first wealth point on
            the upper curve.
        policy_1_upper_curve (float): The policy function at the first wealth point on
            the upper curve.
        wealth_2_upper_curve (float): The second wealth point on the upper curve.
        value_2_upper_curve (float): The value function at the second wealth point on
            the     upper curve.
        policy_2_upper_curve (float): The policy function at the second wealth point on
            the upper curve.

    Returns:
        tuple:

        - float: intersection point on wealth grid
        - float: value of the value function at intersection
        - float: value of the lower (left) policy function at the intersection
        - float: value of the upper (right) policy function at the intersection.

    """
    # Calculate intersection of two lines
    intersect_grid, intersect_value = _find_linear_intersection(
        x1=wealth_1_lower_curve,
        y1=value_1_lower_curve,
        x2=wealth_2_lower_curve,
        y2=value_2_lower_curve,
        x3=wealth_1_upper_curve,
        y3=value_1_upper_curve,
        x4=wealth_2_upper_curve,
        y4=value_2_upper_curve,
    )

    # Extrapolate policy
    policy_left = _evaluate_point_on_line(
        x1=wealth_1_lower_curve,
        y1=policy_1_lower_curve,
        x2=wealth_2_lower_curve,
        y2=policy_2_lower_curve,
        point_to_evaluate=intersect_grid,
    )

    policy_right = _evaluate_point_on_line(
        x1=wealth_1_upper_curve,
        y1=policy_1_upper_curve,
        x2=wealth_2_upper_curve,
        y2=policy_2_upper_curve,
        point_to_evaluate=intersect_grid,
    )

    return intersect_grid, intersect_value, policy_left, policy_right


def calc_gradient(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
):
    """Calculate the gradient between two points.

    Returns 0 if the points are the same.

    Args:
        x1 (float): x-coordinate of the first point.
        y1 (float): y-coordinate of the first point.
        x2 (float): x-coordinate of the second point.
        y2 (float): y-coordinate of the second point.

    Returns:
        float: The gradient between the two points.

    """
    return (y1 - y2) / (x1 - x2 + 1e-16)


def _evaluate_point_on_line(
    x1: float, y1: float, x2: float, y2: float, point_to_evaluate: float
) -> float:
    """Evaluate a point on a line defined by (x1, y1) and (x2, y2).

    Args:
        x1 (float): x-coordinate of the first point.
        y1 (float): y-coordinate of the first point.
        x2 (float): x-coordinate of the second point.
        y2 (float): y-coordinate of the second point.
        point_to_evaluate (float): The point to evaluate.

    Returns:
        float: The value of the point on the line.

    """
    return (y2 - y1) / ((x2 - x1) + 1e-16) * (point_to_evaluate - x1) + y1


def _find_linear_intersection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> Tuple[float, float]:
    """Find the linear intersection of two lines.

    Args:

        x1 (float): x-coordinate of the first point of the first line.
        y1 (float): y-coordinate of the first point of the first line.
        x2 (float): x-coordinate of the second point of the first line.
        y2 (float): y-coordinate of the second point of the first line.
        x3 (float): x-coordinate of the first point of the second line.
        y3 (float): y-coordinate of the first point of the second line.
        x4 (float): x-coordinate of the second point of the second line.
        y4 (float): y-coordinate of the second point of the second line.

    Returns:
        tuple: x and y coordinates of the intersection point.

    """

    slope1 = calc_gradient(x1, y1, x2, y2)
    slope2 = calc_gradient(x3, y3, x4, y4)

    x_intersection = (slope1 * x1 - slope2 * x3 + y3 - y1) / ((slope1 - slope2) + 1e-16)
    y_intersection = slope1 * (x_intersection - x1) + y1

    return x_intersection, y_intersection
