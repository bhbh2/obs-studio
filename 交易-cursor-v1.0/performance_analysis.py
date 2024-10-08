import numpy as np

class PerformanceAnalysis:
    @staticmethod
    def calculate_drawdowns(portfolio_values):
        hwm = np.maximum.accumulate(portfolio_values)
        epsilon = 1e-10
        drawdowns = (hwm - portfolio_values) / (hwm + epsilon)
        return drawdowns

    @staticmethod
    def calculate_max_drawdown(portfolio_values):
        drawdowns = PerformanceAnalysis.calculate_drawdowns(portfolio_values)
        return drawdowns.max()

    @staticmethod
    def calculate_cagr(initial_value, final_value, years):
        epsilon = 1e-10
        if initial_value <= epsilon or final_value <= epsilon or years <= epsilon:
            return 0
        return (final_value / initial_value) ** (1 / years) - 1
