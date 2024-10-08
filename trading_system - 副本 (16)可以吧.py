import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import matplotlib.pyplot as plt
import json
import talib
import os
import pyautogui
import keyboard
import threading
import queue
import requests
from abc import ABC, abstractmethod

# 设置日志
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 禁用 pyautogui 的 FAILSAFE 模式
pyautogui.FAILSAFE = False

# 策略基类
class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, data):
        pass

# 趋势跟踪策略
class TrendFollowingStrategy(Strategy):
    def __init__(self):
        self.short_window = 10
        self.long_window = 30
        self.rsi_window = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

    def generate_signal(self, data):
        data['MA_short'] = data['close'].rolling(window=self.short_window).mean()
        data['MA_long'] = data['close'].rolling(window=self.long_window).mean()
        data['RSI'] = talib.RSI(data['close'], timeperiod=self.rsi_window)
        data['MACD'], data['Signal_Line'], _ = talib.MACD(data['close'], fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal)

        if (data['MA_short'].iloc[-1] > data['MA_long'].iloc[-1] and
            data['RSI'].iloc[-1] < self.rsi_overbought and
            data['MACD'].iloc[-1] > data['Signal_Line'].iloc[-1]):
            return 1
        elif (data['MA_short'].iloc[-1] < data['MA_long'].iloc[-1] and
              data['RSI'].iloc[-1] > self.rsi_oversold and
              data['MACD'].iloc[-1] < data['Signal_Line'].iloc[-1]):
            return -1
        return 0

# 均值回归策略
class MeanReversionStrategy(Strategy):
    def __init__(self):
        self.lookback_period = 20
        self.std_dev_threshold = 2
        self.stoch_overbought = 80
        self.stoch_oversold = 20

    def generate_signal(self, data):
        data['MA'] = data['close'].rolling(window=self.lookback_period).mean()
        data['Std_Dev'] = data['close'].rolling(window=self.lookback_period).std()
        data['Z_Score'] = (data['close'] - data['MA']) / data['Std_Dev']
        data['Stoch_K'], data['Stoch_D'] = talib.STOCH(data['high'], data['low'], data['close'])
        
        if (data['Z_Score'].iloc[-1] < -self.std_dev_threshold and 
            data['Stoch_K'].iloc[-1] < self.stoch_oversold):
            return 1
        elif (data['Z_Score'].iloc[-1] > self.std_dev_threshold and 
              data['Stoch_K'].iloc[-1] > self.stoch_overbought):
            return -1
        return 0

# 突破策略
class BreakoutStrategy(Strategy):
    def __init__(self):
        self.lookback_period = 20

    def generate_signal(self, data):
        upper, middle, lower = talib.BBANDS(data['close'], timeperiod=self.lookback_period)
        if data['close'].iloc[-1] > upper.iloc[-1]:
            return 1
        elif data['close'].iloc[-1] < lower.iloc[-1]:
            return -1
        return 0

# 动量策略
class MomentumStrategy(Strategy):
    def __init__(self):
        self.momentum_period = 10
        self.threshold = 0.02

    def generate_signal(self, data):
        momentum = data['close'].pct_change(self.momentum_period)
        if momentum.iloc[-1] > self.threshold:
            return 1
        elif momentum.iloc[-1] < -self.threshold:
            return -1
        return 0

# 统计套利策略
class StatisticalArbitrageStrategy(Strategy):
    def __init__(self):
        self.lookback_period = 20
        self.z_score_threshold = 2

    def generate_signal(self, data):
        rolling_mean = data['close'].rolling(window=self.lookback_period).mean()
        rolling_std = data['close'].rolling(window=self.lookback_period).std()
        z_score = (data['close'] - rolling_mean) / rolling_std
        if z_score.iloc[-1] < -self.z_score_threshold:
            return 1
        elif z_score.iloc[-1] > self.z_score_threshold:
            return -1
        return 0

# 斐波那契回调策略
class FibonacciRetracementStrategy(Strategy):
    def __init__(self):
        self.lookback_period = 100
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.tolerance = 0.01

    def generate_signal(self, data):
        high = data['high'].rolling(window=self.lookback_period).max().iloc[-1]
        low = data['low'].rolling(window=self.lookback_period).min().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        fib_levels = [high - (high - low) * level for level in self.fib_levels]
        
        for level in fib_levels:
            if abs(current_price - level) / level < self.tolerance:
                if current_price > level:
                    return 1
                else:
                    return -1
        return 0

# 交易系统
class TradingSystem:
    def __init__(self, csv_file, initial_balance=100000, start_date='2024-09-01', end_date=None, state_file='trading_state.json'):
        self.csv_file = csv_file
        self.balance = initial_balance
        self.position = 0
        self.trades = []
        self.open_trades = []
        self.data = None
        self.current_index = 0
        self.last_trade_index = -1
        self.cooling_period = 40
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.total_profit = 0
        self.point_value = 10
        self.commission = 7.2
        self.state_file = state_file
        self.paused = False
        self.command_queue = queue.Queue()
        self.strategies = {
            'trend_following': TrendFollowingStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'breakout': BreakoutStrategy(),
            'momentum': MomentumStrategy(),
            'statistical_arbitrage': StatisticalArbitrageStrategy(),
            'fibonacci_retracement': FibonacciRetracementStrategy()
        }
        self.current_strategy = None
        self.strategy_performance = {name: {'profit': 0, 'trades': 0} for name in self.strategies.keys()}

    def load_data(self):
        try:
            self.data = pd.read_csv(self.csv_file)
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            self.data = self.data.sort_values('datetime')
            self.data = self.data[(self.data['datetime'] >= self.start_date) & 
                                  (self.data['datetime'] <= self.end_date if self.end_date else True)]
            if self.data.empty:
                logger.warning("警告：在指定的时间范围内没有数据")
            else:
                logger.info(f"数据加载成功，共 {len(self.data)} 条记录")
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            raise

    def preprocess_data(self):
        try:
            self.data['Returns'] = self.data['close'].pct_change()
            self.data['Volatility'] = self.data['Returns'].rolling(window=20).std() * np.sqrt(252)
            self.data['MA10'] = self.data['close'].rolling(window=10).mean()
            self.data['MA30'] = self.data['close'].rolling(window=30).mean()
            self.data['RSI'] = talib.RSI(self.data['close'])
            self.data['MACD'], self.data['Signal'], _ = talib.MACD(self.data['close'])
            self.data = self.data.dropna()
            logger.info("数据预处理完成")
        except Exception as e:
            logger.error(f"预处理数据时出错: {str(e)}")
            raise

    def select_strategy(self):
        recent_data = self.data.iloc[-30:]
        volatility = recent_data['close'].pct_change().std()
        trend = recent_data['close'].pct_change(20).iloc[-1]
        momentum = recent_data['close'].pct_change(10).iloc[-1]
        
        if volatility > 0.02 and abs(trend) > 0.05:
            return self.strategies['trend_following']
        elif volatility < 0.01:
            return self.strategies['mean_reversion']
        elif volatility > 0.03:
            return self.strategies['breakout']
        elif abs(momentum) > 0.03:
            return self.strategies['momentum']
        elif abs(trend) > 0.02:
            return self.strategies['fibonacci_retracement']
        else:
            return self.strategies['statistical_arbitrage']

    def generate_signal(self, row):
        self.current_strategy = self.select_strategy()
        return self.current_strategy.generate_signal(self.data.iloc[-30:])

    def calculate_position_size(self, price):
        account_value = self.calculate_portfolio_value()
        risk_per_trade = 0.01 * account_value
        volatility = self.data['Volatility'].iloc[self.current_index]
        if np.isnan(volatility) or np.isnan(price) or volatility <= 1e-10 or price <= 1e-10:
            return 0
        contract_size = max(1, int(risk_per_trade / (price * volatility)))
        return min(contract_size, 5)

    def calculate_portfolio_value(self):
        price = self.data['close'].iloc[self.current_index]
        unrealized_pnl = self.position * (price - self.open_trades[0]['price']) if self.open_trades else 0
        return self.balance + unrealized_pnl

    def execute_trade(self, signal, price):
        if signal == 0 or np.isnan(price):
            return

        if self.current_index - self.last_trade_index <= self.cooling_period:
            return

        timestamp = self.data['datetime'].iloc[self.current_index]
        contract_size = self.calculate_position_size(price)

        if signal == 1 and self.position <= 0:
            if self.position == -1:
                profit = self.open_trades[0]['price'] - price
                self.balance += profit
                self.trades.append(('平空', 1, self.open_trades[0]['price'], self.open_trades[0]['time'], price, timestamp, profit))
                self.open_trades.pop(0)
                self.position = 0

            self.position = 1
            self.open_trades.append({'price': price, 'size': contract_size, 'time': timestamp})
            self.trades.append(('开多', contract_size, price, timestamp, None, None, None))
            logger.info(f"开多 {contract_size} 手，价格：{price:.2f}")

        elif signal == -1 and self.position >= 0:
            if self.position == 1:
                profit = price - self.open_trades[0]['price']
                self.balance += profit
                self.trades.append(('平多', 1, self.open_trades[0]['price'], self.open_trades[0]['time'], price, timestamp, profit))
                self.open_trades.pop(0)
                self.position = 0

            self.position = -1
            self.open_trades.append({'price': price, 'size': -contract_size, 'time': timestamp})
            self.trades.append(('开空', contract_size, price, timestamp, None, None, None))
            logger.info(f"开空 {contract_size} 手，价格：{price:.2f}")

        self.last_trade_index = self.current_index

    def check_stop_loss_take_profit(self):
        if not self.open_trades:
            return None

        current_price = self.data['close'].iloc[-1]
        open_trade = self.open_trades[0]
        open_price = open_trade['price']
        timestamp = self.data['datetime'].iloc[-1]

        stop_loss_pct = 0.015
        take_profit_pct = 0.025

        if self.position == 1:
            if current_price <= open_price * (1 - stop_loss_pct) or current_price >= open_price * (1 + take_profit_pct):
                profit_points = current_price - open_price
                profit = profit_points * self.point_value - self.commission
                self.balance += profit
                self.total_profit += profit
                self.trades.append(('平多', 1, open_price, open_trade['time'], current_price, timestamp, profit))
                self.open_trades.pop(0)
                self.position = 0
                return {
                    "action": "平多(止损/止盈)",
                    "size": 1,
                    "price": current_price,
                    "timestamp": timestamp,
                    "profit": profit,
                    "open_price": open_price,
                    "open_time": open_trade['time']
                }

        elif self.position == -1:
            if current_price >= open_price * (1 + stop_loss_pct) or current_price <= open_price * (1 - take_profit_pct):
                profit_points = open_price - current_price
                profit = profit_points * self.point_value - self.commission
                self.balance += profit
                self.total_profit += profit
                self.trades.append(('平空', 1, open_price, open_trade['time'], current_price, timestamp, profit))
                self.open_trades.pop(0)
                self.position = 0
                return {
                    "action": "平空(止损/止盈)",
                    "size": 1,
                    "price": current_price,
                    "timestamp": timestamp,
                    "profit": profit,
                    "open_price": open_price,
                    "open_time": open_trade['time']
                }

        return None

    def keyboard_listener(self):
        def on_press(key):
            if key.name == 'p':
                self.command_queue.put('pause')
            elif key.name == 'o':
                self.command_queue.put('resume')

        keyboard.on_press(on_press)

    def handle_commands(self):
        while True:
            try:
                command = self.command_queue.get_nowait()
                if command == 'pause':
                    self.paused = True
                    print("程序已暂停。按 'o' 继续。")
                elif command == 'resume':
                    self.paused = False
                    print("程序继续运行。")
            except queue.Empty:
                break

    def run_simulation(self):
        logger.info("开始交易模拟")
        portfolio_values = []

        threading.Thread(target=self.keyboard_listener, daemon=True).start()

        for i in range(len(self.data)):
            self.handle_commands()
            while self.paused:
                time.sleep(0.1)
                self.handle_commands()

            self.current_index = i
            row = self.data.iloc[i]
            price = row['close']
            
            if not np.isnan(price):
                self.check_stop_loss_take_profit()
                signal = self.generate_signal(row)
                self.execute_trade(signal, price)
            
            portfolio_value = self.calculate_portfolio_value()
            portfolio_values.append(portfolio_value)
            
            if i % 100 == 0:
                logger.info(f"模拟进度: {i}/{len(self.data)}")

        self.plot_portfolio_value(portfolio_values)
        self.print_summary(portfolio_values)

    def plot_portfolio_value(self, portfolio_values):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['datetime'], portfolio_values)
        plt.title('投资组合价值随时间变化')
        plt.xlabel('时间')
        plt.ylabel('投资组合价值')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('portfolio_value.png')
        logger.info("投资组合价值图表已保存为 portfolio_value.png")

    def print_summary(self, portfolio_values):
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        logger.info("=== 交易摘要 ===")
        logger.info(f"初始资金: ${initial_value:.2f}")
        logger.info(f"最终资金: ${final_value:.2f}")
        logger.info(f"总回报率: {total_return:.2f}%")
        logger.info(f"总交易次数: {len(self.trades)}")
        
        logger.info("\n=== 详细交易记录 ===")
        
        trade_records = []
        open_trades = {}
        total_profit = 0
        
        for trade in self.trades:
            if trade[0] in ['开多', '开空']:
                open_trades[trade[3]] = trade
            else:
                open_trade = open_trades.pop(trade[3], None)
                if open_trade:
                    trade_type = '多' if open_trade[0] == '开多' else '空'
                    size = trade[1]
                    open_price = open_trade[2]
                    open_time = open_trade[3]
                    close_price = trade[4]
                    close_time = trade[5]
                    profit = trade[6]
                    total_profit += profit
                    
                    trade_record = {
                        "交易类型": trade_type,
                        "手数": size,
                        "开仓价格": round(open_price, 2),
                        "开仓时间": open_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "平仓价格": round(close_price, 2),
                        "平仓时间": close_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "盈利": round(profit, 2)
                    }
                    trade_records.append(trade_record)
        
        for open_trade in open_trades.values():
            trade_type = '多' if open_trade[0] == '开多' else '空'
            size = open_trade[1]
            open_price = open_trade[2]
            open_time = open_trade[3]
            
            trade_record = {
                "交易类型": trade_type,
                "手数": size,
                "开仓价格": round(open_price, 2),
                "开仓时间": open_time.strftime("%Y-%m-%d %H:%M:%S"),
                "平仓价格": "未平仓",
                "平仓时间": "未平仓",
                "盈利": "未实现"
            }
            trade_records.append(trade_record)
        
        json_output = json.dumps(trade_records, ensure_ascii=False, indent=2)
        logger.info(f"交易记录 JSON:\n{json_output}")
        
        logger.info(f"\n最终持仓: {self.position} 手")
        logger.info(f"现金余额: ${self.balance:.2f}")
        logger.info(f"总盈利: ${total_profit:.2f}")

        returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(portfolio_values)

        logger.info(f"夏普比率: {sharpe_ratio:.2f}")
        logger.info(f"最大回撤: {max_drawdown:.2%}")

        profit_trades = [t for t in self.trades if t[6] is not None and t[6] > 0]
        loss_trades = [t for t in self.trades if t[6] is not None and t[6] <= 0]

        logger.info(f"盈利交易次数: {len(profit_trades)}")
        logger.info(f"亏损交易次数: {len(loss_trades)}")
        
        if profit_trades:
            avg_profit = sum(t[6] for t in profit_trades) / len(profit_trades)
            logger.info(f"平均盈利: ${avg_profit:.2f}")
        
        if loss_trades:
            avg_loss = sum(t[6] for t in loss_trades) / len(loss_trades)
            logger.info(f"平均亏损: ${avg_loss:.2f}")

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_max_drawdown(self, portfolio_values):
        cumulative_returns = (pd.Series(portfolio_values) / portfolio_values[0]) - 1
        running_max = cumulative_returns.cummax()
        drawdown = running_max - cumulative_returns
        return drawdown.max()

    def run_backtest(self):
        self.load_data()
        self.preprocess_data()
        self.run_simulation()

    def real_time_trade(self, timestamp, price):
        try:
            self.handle_commands()
            while self.paused:
                time.sleep(0.1)
                self.handle_commands()

            if self.data is None:
                self.load_data()
                self.preprocess_data()

            self.current_index = len(self.data)

            new_data = pd.DataFrame({
                'datetime': [timestamp],
                'close': [price],
                'high': [price],
                'low': [price],
            })
            self.data = pd.concat([self.data, new_data], ignore_index=True)

            self.update_indicators()

            stop_loss_take_profit_action = self.check_stop_loss_take_profit()
            if stop_loss_take_profit_action:
                result = self.format_trade_result(stop_loss_take_profit_action)
                if result['盈利'] != "未实现":
                    profit = float(result['盈利'])
                    self.total_profit += profit
                    self.update_strategy_performance(profit)
                return result, self.get_latest_trade_record(), self.total_profit, type(self.current_strategy).__name__

            self.current_strategy = self.select_strategy()
            signal = self.current_strategy.generate_signal(self.data.iloc[-30:])

            trade_action = self.execute_real_time_trade(signal, price, timestamp)

            if trade_action:
                result = self.format_trade_result(trade_action)
                if result['盈利'] != "未实现":
                    profit = float(result['盈利'])
                    self.total_profit += profit
                    self.update_strategy_performance(profit)
                return result, self.get_latest_trade_record(), self.total_profit, type(self.current_strategy).__name__
            
            return None, None, self.total_profit, type(self.current_strategy).__name__
        except Exception as e:
            logger.error(f"实时交易时出错: {str(e)}")
            return None, None, self.total_profit, None

    def update_indicators(self):
        self.data['Returns'] = self.data['close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std() * np.sqrt(252)
        self.data['MA10'] = self.data['close'].rolling(window=10).mean()
        self.data['MA30'] = self.data['close'].rolling(window=30).mean()
        self.data['RSI'] = talib.RSI(self.data['close'])
        self.data['MACD'], self.data['Signal'], _ = talib.MACD(self.data['close'])

    def execute_real_time_trade(self, signal, price, timestamp):
        if signal == 0 or self.current_index - self.last_trade_index <= self.cooling_period:
            return None

        contract_size = self.calculate_position_size(price)

        if signal == 1 and self.position <= 0:
            if self.position == -1:
                profit = self.open_trades[0]['price'] - price
                self.balance += profit
                self.trades.append(('平空', 1, self.open_trades[0]['price'], self.open_trades[0]['time'], price, timestamp, profit))
                self.open_trades.pop(0)
                self.position = 0

            self.position = 1
            self.open_trades.append({'price': price, 'size': contract_size, 'time': timestamp})
            self.trades.append(('开多', contract_size, price, timestamp, None, None, None))
            return {
                "action": "开多",
                "size": contract_size,
                "price": price,
                "timestamp": timestamp
            }

        elif signal == -1 and self.position >= 0:
            if self.position == 1:
                profit = price - self.open_trades[0]['price']
                self.balance += profit
                self.trades.append(('平多', 1, self.open_trades[0]['price'], self.open_trades[0]['time'], price, timestamp, profit))
                self.open_trades.pop(0)
                self.position = 0

            self.position = -1
            self.open_trades.append({'price': price, 'size': -contract_size, 'time': timestamp})
            self.trades.append(('开空', contract_size, price, timestamp, None, None, None))
            return {
                "action": "开空",
                "size": contract_size,
                "price": price,
                "timestamp": timestamp
            }

        self.last_trade_index = self.current_index
        return None

    def format_trade_result(self, trade_action):
        if trade_action['action'] in ['开多', '开空']:
            return {
                "交易类型": "多" if trade_action['action'] == '开多' else "空",
                "手数": int(trade_action['size']),
                "开仓价格": round(float(trade_action['price']), 2),
                "开仓时间": trade_action['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                "平仓价格": "未平仓",
                "平仓时间": "未平仓",
                "盈利": "未实现",
                "手续费": "未扣除",
                "总盈利": round(self.total_profit, 2)
            }
        elif trade_action['action'] in ['平多', '平空', '平多(止损/止盈)', '平空(止损/止盈)']:
            return {
                "交易类型": "多" if trade_action['action'] in ['平多', '平多(止损/止盈)'] else "空",
                "手数": int(trade_action['size']),
                "开仓价格": round(float(trade_action['open_price']), 2),
                "开仓时间": trade_action['open_time'].strftime("%Y-%m-%d %H:%M:%S"),
                "平仓价格": round(float(trade_action['price']), 2),
                "平仓时间": trade_action['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                "盈利": round(float(trade_action['profit']), 2),
                "手续费": self.commission,
                "总盈利": round(self.total_profit, 2)
            }
        return None

    def get_latest_trade_record(self):
        if self.trades:
            latest_trade = self.trades[-1]
            if len(latest_trade) == 7 and latest_trade[4] is not None:
                return {
                    "交易类型": latest_trade[0],
                    "手数": latest_trade[1],
                    "开仓价格": round(float(latest_trade[2]), 2),
                    "开仓时间": latest_trade[3].strftime("%Y-%m-%d %H:%M:%S"),
                    "平仓价格": round(float(latest_trade[4]), 2),
                    "平仓时间": latest_trade[5].strftime("%Y-%m-%d %H:%M:%S"),
                    "盈利": round(float(latest_trade[6]), 2)
                }
            else:
                return {
                    "交易类型": latest_trade[0],
                    "手数": latest_trade[1],
                    "开仓价格": round(float(latest_trade[2]), 2),
                    "开仓时间": latest_trade[3].strftime("%Y-%m-%d %H:%M:%S"),
                    "平仓价格": "未平仓",
                    "平仓时间": "未平仓",
                    "盈利": "未实现"
                }
        return None

    def update_strategy_performance(self, profit):
        if self.current_strategy:
            strategy_name = type(self.current_strategy).__name__
            self.strategy_performance[strategy_name]['profit'] += profit
            self.strategy_performance[strategy_name]['trades'] += 1

    def get_real_time_data(self):
        url = "https://45.futsseapi.eastmoney.com/sse/114_c2411_qt?token=1101ffec61617c99be287c1bec3085ff"
        try:
            response = requests.get(url, timeout=20, stream=True)
            response.raise_for_status()
            content = ''
            
            for chunk in response.iter_content(chunk_size=1024):
                data = chunk.decode('utf-8')
                if data.find("{") == -1:
                    continue
            
                data = data[data.find("{"):data.rfind("}")+1]
                print(data)  # 打印原始数据，用于调试
                d = json.loads(data)
                
                if 'qt' in d and 'p' in d['qt'] and 'utime' in d['qt']:
                    price = float(d['qt']['p'])
                    timestamp = datetime.fromtimestamp(d['qt']['utime'])
                    return price, timestamp
                
            return None, None
                
        except requests.Timeout:
            print('请求超时')
            return self.get_real_time_data()
        except requests.RequestException as e:
            print(f'请求错误: {e}')
            return None, None

    def run_real_time_trading(self):
        print("开始实时交易...")
        while True:
            price, timestamp = self.get_real_time_data()
            if price is not None and timestamp is not None:
                if self.data is None or len(self.data) == 0:
                    self.load_data()
                    self.preprocess_data()
                
                action, trade_record, total_profit, strategy_name = self.real_time_trade(timestamp, price)
                if action:
                    print(json.dumps(action, ensure_ascii=False, indent=2))
                    print(f"使用策略: {strategy_name}")
                if trade_record:
                    print("最新交易记录:")
                    print(json.dumps(trade_record, ensure_ascii=False, indent=2))
                #print("策略表现:")
                #print(json.dumps(self.strategy_performance, ensure_ascii=False, indent=2))
                self.save_state()
            
            self.handle_commands()
            while self.paused:
                time.sleep(0.1)
                self.handle_commands()
            
            time.sleep(1)  # 每秒请求一次数据

    def save_state(self):
        state = {
            'balance': self.balance,
            'position': self.position,
            'trades': self.trades,
            'open_trades': self.open_trades,
            'total_profit': self.total_profit,
            'current_index': self.current_index,
            'last_trade_index': self.last_trade_index,
            'strategy_performance': self.strategy_performance
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, default=str)

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.balance = state.get('balance', self.balance)
            self.position = state.get('position', self.position)
            self.trades = state.get('trades', self.trades)
            self.open_trades = state.get('open_trades', self.open_trades)
            self.total_profit = state.get('total_profit', 0)
            self.current_index = state.get('current_index', self.current_index)
            self.last_trade_index = state.get('last_trade_index', self.last_trade_index)
            self.strategy_performance = state.get('strategy_performance', self.strategy_performance)
            print(f"已从 {self.state_file} 加载交易状态")
            return True
        return False

    def execute_mouse_click(self, action):
        try:
            if action == "开多":
                pyautogui.moveTo(328, 1170, duration=0.5)
                pyautogui.click()
                print("执行开多操作")
            elif action == "开空":
                pyautogui.moveTo(461, 1169, duration=0.5)
                pyautogui.click()
                print("执行开空操作")
            elif action == "平仓":
                pyautogui.moveTo(601, 1171, duration=0.5)
                pyautogui.click()
                print("执行平仓操作")
            time.sleep(0.5)
        except Exception as e:
            print(f"执行鼠标点击时出错: {str(e)}")

    def format_trade_for_display(self, trade):
        trade_type, size, open_price, open_time, close_price, close_time, profit = trade
        return {
            "交易类型": trade_type,
            "手数": size,
            "开仓价格": round(float(open_price), 2),
            "开仓时间": open_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(open_time, datetime) else str(open_time),
            "平仓价格": round(float(close_price), 2) if close_price is not None else "未平仓",
            "平仓时间": close_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(close_time, datetime) else "未平仓",
            "盈利": round(float(profit), 2) if profit is not None else "未实现"
        }

def main():
    csv_file = "e:/data/29#CL8.csv"
    initial_balance = 100000
    start_date = '2023-01-01'
    end_date = '2024-09-30'
    state_file = 'trading_state.json'
    
    trading_system = TradingSystem(csv_file, initial_balance, start_date, end_date, state_file)
    
    threading.Thread(target=trading_system.keyboard_listener, daemon=True).start()
    
    if trading_system.load_state():
        print("继续之前的交易")
        print("\n=== 之前的交易记录 ===")
        for trade in trading_system.trades:
            trade_info = trading_system.format_trade_for_display(trade)
            print(json.dumps(trade_info, ensure_ascii=False, indent=2))
        print("=== 交易记录结束 ===\n")
    else:
        print("开始新的交易模拟")
    
    trading_system.load_data()
    trading_system.preprocess_data()

    if trading_system.data is None or len(trading_system.data) == 0:
        print("错误：没有加载到交易数据")
        return

    trading_system.run_real_time_trading()

if __name__ == "__main__":
    main()