"""What the switch would save, given how many calls the local model takes over."""


def economics(coverage, price_per_1k_calls=.50, monthly_calls=3000000, serving_cost=20):
    """Calls the local model answers stop costing LLM money; the rest still go to the LLM.

    `serving_cost` is the fixed monthly cost of running the local model.
    """
    baseline = monthly_calls * price_per_1k_calls / 1000
    saving = coverage * baseline - serving_cost
    saved_per_call = coverage * price_per_1k_calls / 1000
    return {
        'baseline': baseline,
        'after': baseline - saving,
        'monthly_saving': saving,
        'percent_saving': 100 * saving / baseline if baseline else 0,
        'calls_saved_per_1000': 1000 * coverage,
        'monthly_calls_saved': coverage * monthly_calls,
        'break_even_monthly_calls': serving_cost / saved_per_call if saved_per_call > 0 else None,
        'assumptions': {'price_per_1k_calls': price_per_1k_calls, 'monthly_calls': monthly_calls,
                        'serving_cost': serving_cost},
    }
