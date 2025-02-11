def adjust_predictions(predictions: float) -> float:
    """
    Adjust predictions total amount of taxi charge by adding 2$.
    """
    extra_charge = 2
    return predictions + extra_charge