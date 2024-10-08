import threading
from trading_system import TradingSystem

def main():
    csv_file = "e:/data/29#CL8.csv"
    initial_balance = 100000
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    state_file = 'trading_state.json'
    
    trading_system = TradingSystem(csv_file, initial_balance, start_date, end_date, state_file)
    
    # 启动键盘监听线程
    threading.Thread(target=trading_system.keyboard_listener, daemon=True).start()
    
    # 尝试加载之前的状态
    if trading_system.load_state():
        print("继续之前的交易")
        # 显示之前的交易记录
        print("\n=== 之前的交易记录 ===")
        for trade in trading_system.trades:
            trade_info = trading_system.format_trade_for_display(trade)
            print(json.dumps(trade_info, ensure_ascii=False, indent=2))
        print("=== 交易记录结束 ===\n")
    else:
        print("开始新的交易模拟")
    
    # 无论是否加载了状态，都重新加载和预处理数据
    trading_system.load_data()
    trading_system.preprocess_data()

    if trading_system.data is None or len(trading_system.data) == 0:
        print("错误：没有加载到交易数据")
        return

    # 运行实时交易
    trading_system.run_real_time_trading()

if __name__ == "__main__":
    main()
