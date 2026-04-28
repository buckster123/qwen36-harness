"""Calculator tool — exact-precision math via sympy. No ``eval``."""

from __future__ import annotations

from . import Registry, ToolError, default_registry


def register(registry: Registry = default_registry) -> None:
    try:
        import sympy
    except ImportError:  # pragma: no cover - sympy is a hard dep
        raise RuntimeError("sympy required for calc tool") from None

    @registry.tool(
        name="calc.eval",
        description=(
            "Evaluate a mathematical expression using exact-precision arithmetic. "
            "Supports arithmetic, sqrt, log, exp, trig, and symbolic algebra. "
            "Use this whenever you need a numeric answer — never compute multi-step "
            "math by hand. Examples: '17*91', 'sqrt(2)+pi', 'sin(pi/4)', "
            "'integrate(x**2, x)', 'solve(x**2-4, x)'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A sympy-parseable expression or function call.",
                },
                "as_float": {
                    "type": "boolean",
                    "description": "If true, evaluate to a numerical float (vs. exact form).",
                    "default": False,
                },
            },
            "required": ["expression"],
        },
    )
    def calc_eval(expression: str, as_float: bool = False) -> dict:
        # Reject obviously non-mathematical input — sympy is permissive and
        # will parse "this is not math" as a boolean expression. We require
        # at least one digit, math operator, or known math function name.
        normalised = expression.strip()
        if not normalised:
            raise ToolError("empty expression")
        looks_mathy = (
            any(c.isdigit() for c in normalised)
            or any(op in normalised for op in "+-*/^=<>")
            or any(
                fn in normalised
                for fn in (
                    "sqrt", "log", "exp", "sin", "cos", "tan", "pi", "E",
                    "integrate", "diff", "solve", "simplify", "factor",
                    "expand", "limit", "Sum", "Product", "oo",
                )
            )
        )
        if not looks_mathy:
            raise ToolError(
                f"'{expression}' does not look like a mathematical expression"
            )
        try:
            result = sympy.sympify(expression)
        except (sympy.SympifyError, SyntaxError, TypeError) as e:
            raise ToolError(f"could not parse '{expression}': {e}") from None
        if as_float:
            try:
                evaluated = float(result.evalf())  # type: ignore[no-untyped-call]
                return {"expression": expression, "result": evaluated, "form": "float"}
            except (TypeError, ValueError) as e:
                raise ToolError(f"could not coerce to float: {e}") from None
        return {
            "expression": expression,
            "result": str(result),
            "result_simplified": str(sympy.simplify(result)),
            "form": "exact",
        }


__all__ = ["register"]
