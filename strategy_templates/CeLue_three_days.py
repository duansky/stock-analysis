"""
连续三天上涨选股策略
策略描述：
1. 连续三天上涨，单日涨幅不超过7%
2. 近两日换手率排名前100
3. 获利筹码大于70%
4. 90个交易日内必须出现过三次涨停
5. 剔除ST股票
6. 上市时间超过一个月
7. 支持target_date回测

此文件不直接执行，通过xuangu.py或celue_save.py调用
"""
import numpy as np
import pandas as pd
import talib
import time
import func
from func_TDX import rolling_window, REF, MA, SMA, HHV, LLV, COUNT, EXIST, CROSS, BARSLAST
from rich import print


def 策略HS300(df_hs300, start_date='', end_date=''):
    """
    HS300信号的作用是，当信号是0时，当日不买股票，1时买入。传出
    :param start_date:
    :param end_date:
    :return: 布尔序列
    """
    if start_date == '':
        start_date = df_hs300.index[0]  # 设置为df第一个日期
    if end_date == '':
        end_date = df_hs300.index[-1]  # 设置为df最后一个日期
    df_hs300 = df_hs300.loc[start_date:end_date]
    HS300_CLOSE = df_hs300['close']
    HS300_当日涨幅 = (HS300_CLOSE / REF(HS300_CLOSE, 1) - 1) * 100
    HS300_信号 = ~(HS300_当日涨幅 < -1.5) & ~(HS300_当日涨幅 > 1.5)
    return HS300_信号


def 策略1(df, start_date='', end_date='', mode=None):
    """
    连续三天上涨策略1：基础筛选条件
    
    :param DataFrame df:输入具体一个股票的DataFrame数据表。时间列为索引。
    :param mode :str 'fast'为快速模式，只处理当日数据，用于开盘快速筛选股票。
    :param date start_date:可选。留空从头开始。2020-10-10格式，策略指定从某日期开始
    :param date end_date:可选。留空到末尾。2020-10-10格式，策略指定到某日期结束
    :return : 布尔序列
    """
    if start_date == '':
        start_date = df.index[0]  # 设置为df第一个日期
    if end_date == '':
        end_date = df.index[-1]  # 设置为df最后一个日期
    df = df.loc[start_date:end_date]

    O = df['open']
    H = df['high']
    L = df['low']
    C = df['close']
    
    if {'换手率'}.issubset(df.columns):
        换手率 = df['换手率']
    else:
        换手率 = 0

    if mode == 'fast':
        # 天数不足100天，收盘价小于3直接返回FALSE
        if C.shape[0] < 100 or C.iat[-1] < 3:
            return False

        # 剔除ST股票（股票名称包含ST）
        if 'name' in df.columns and df['name'].iloc[-1] and 'ST' in str(df['name'].iloc[-1]).upper():
            return False

        # 基础条件：股价大于3元
        基础条件 = C.iat[-1] > 3

        # 排除涨停股票
        if df['code'][0][0:2] == "68" or df['code'][0][0:2] == "30":
            涨停价 = 1.2
        else:
            涨停价 = 1.1
        非涨停 = ~((C.iat[-1]+0.01) >= np.ceil((np.floor(REF(C, 1).iat[-1]*1000*涨停价)-4)/10)/100)

        result = 基础条件 & 非涨停
    else:
        # 剔除ST股票
        if 'name' in df.columns and df['name'].iloc[-1] and 'ST' in str(df['name'].iloc[-1]).upper():
            return pd.Series([False] * len(df), index=df.index, dtype=bool)

        # TJ01: 上市时间超过一个月（约22个交易日）
        TJ01 = (BARSLAST(C == 0) > 22) & (C > 3)

        # TJ02: 排除涨停股票
        if df['code'][0][0:2] == "68" or df['code'][0][0:2] == "30":
            涨停价 = 1.2
        else:
            涨停价 = 1.1
        TJ02 = ~((C+0.01) >= np.ceil((np.floor(REF(C, 1)*1000*涨停价)-4)/10)/100)

        result = TJ01 & TJ02
    return result


def 策略2(df, HS300_信号, start_date='', end_date='', target_date=''):
    """
    连续三天上涨选股策略主逻辑
    
    :param DataFrame df:输入具体一个股票的DataFrame数据表。时间列为索引。
    :param HS300_信号: HS300信号序列
    :param date start_date:可选。留空从头开始。2020-10-10格式，策略指定从某日期开始
    :param date end_date:可选。留空到末尾。2020-10-10格式，策略指定到某日期结束
    :param date target_date:可选。指定作为连涨第3天的目标日期。留空则使用end_date作为目标日期
    :return bool: 在目标日期这天，策略是否触发。true触发，false不触发
    """

    if start_date == '':
        start_date = df.index[0]  # 设置为df第一个日期
    if end_date == '':
        end_date = df.index[-1]  # 设置为df最后一个日期
    
    # 如果指定了target_date，则以target_date为准；否则使用end_date
    if target_date == '':
        target_date = end_date
    else:
        target_date = pd.to_datetime(target_date)
    
    df = df.loc[start_date:end_date]

    if df.shape[0] < 100:  # 小于100日 直接返回false
        return False

    # 根据df的索引重建HS300信号，为了与股票交易日期一致
    HS300_信号 = pd.Series(HS300_信号, index=df.index, dtype=bool).dropna()

    O = df['open']
    H = df['high']
    L = df['low']
    C = df['close']
    
    if {'换手率'}.issubset(df.columns):
        换手率 = df['换手率']
    else:
        换手率 = pd.Series(0, index=df.index)

    # 计算每日涨跌幅
    当日涨幅 = (C / REF(C, 1) - 1) * 100

    # 条件1：连续三天上涨，单日涨幅不超过7%
    连续3天上涨 = ((当日涨幅 > 0) & (当日涨幅 <= 7) &
                  (REF(当日涨幅, 1) > 0) & (REF(当日涨幅, 1) <= 7) &
                  (REF(当日涨幅, 2) > 0) & (REF(当日涨幅, 2) <= 7))

    # 条件2：近两日换手率排名前100（这里简化为换手率大于某个阈值，实际使用时需要全市场排名）
    # 注意：真实场景需要在xuangu.py中对所有股票的换手率进行排名
    近两日换手率 = (换手率 + REF(换手率, 1)) / 2
    换手率条件 = 近两日换手率 > 3  # 这里设置3%作为阈值，实际应该是全市场排名前100

    # 条件3：获利筹码大于70%
    # 获利筹码 = 当前价格高于历史价格的比例
    # 计算90日内低于当前价格的交易日占比
    if 'close' in df.columns and len(df) >= 90:
        获利筹码比例 = pd.Series(index=df.index, dtype=float)
        for i in range(90, len(df)):
            历史90日价格 = C.iloc[i-90:i]
            当前价格 = C.iloc[i]
            获利筹码比例.iloc[i] = (历史90日价格 < 当前价格).sum() / 90 * 100
        获利筹码条件 = 获利筹码比例 > 70
    else:
        获利筹码条件 = pd.Series([False] * len(df), index=df.index, dtype=bool)

    # 条件4：90个交易日内必须出现过三次涨停
    # 计算涨停：科创板和创业板20%，其他10%
    if df['code'][0][0:2] == "68" or df['code'][0][0:2] == "30":
        涨停阈值 = 19.5  # 考虑误差，设置为19.5%
    else:
        涨停阈值 = 9.5  # 考虑误差，设置为9.5%
    
    涨停日 = 当日涨幅 >= 涨停阈值
    近90日涨停次数 = COUNT(涨停日, 90)
    涨停次数条件 = 近90日涨停次数 >= 3

    # 基础策略1的条件
    策略1条件 = 策略1(df, start_date, end_date)

    # 综合条件
    买入信号 = (HS300_信号 &
              策略1条件 &
              连续3天上涨 &
              换手率条件 &
              获利筹码条件 &
              涨停次数条件)

    # 如果指定了target_date，只返回该日期的信号
    if target_date in 买入信号.index:
        return 买入信号.loc[target_date]
    else:
        # 如果target_date不在数据范围内，返回False
        return False


def 卖策略(df, 策略2, start_date='', end_date='', target_date=''):
    """
    连续三天上涨策略的卖出逻辑

    :param df: 个股Dataframe
    :param 策略2: 买入策略2的结果（布尔值或序列）
    :param start_date:
    :param end_date:
    :param target_date: 目标日期，如果策略2是单个布尔值，需要指定对应的日期
    :return: 卖出策略序列
    """

    # 如果策略2是单个布尔值，需要转换为序列
    if isinstance(策略2, bool):
        if target_date == '':
            target_date = df.index[-1]
        else:
            target_date = pd.to_datetime(target_date)
        
        if not 策略2:  # 如果策略2为False，直接返回
            return pd.Series([False] * len(df), index=df.index, dtype=bool)
        
        # 创建策略2序列，只在target_date为True
        策略2_series = pd.Series([False] * len(df), index=df.index, dtype=bool)
        if target_date in df.index:
            策略2_series.loc[target_date] = True
        策略2 = 策略2_series

    if True not in 策略2.to_list():  # 买入策略2 没有买入点
        return pd.Series([False] * len(策略2), index=策略2.index, dtype=bool)

    if start_date == '':
        start_date = df.index[0]  # 设置为df第一个日期
    if end_date == '':
        end_date = df.index[-1]  # 设置为df最后一个日期
    df = df.loc[start_date:end_date]

    O = df['open']
    H = df['high']
    L = df['low']
    C = df['close']

    # 变量定义
    MA10 = SMA(C, 10)
    MA20 = SMA(C, 20)

    # 计算买入后的天数和价格
    BUY_TODAY = BARSLAST(策略2)
    BUY_PRICE_CLOSE = pd.Series(index=C.index, dtype=float)
    BUY_PRICE_OPEN = pd.Series(index=C.index, dtype=float)
    BUY_PCT = pd.Series(index=C.index, dtype=float)
    BUY_PCT_MAX = pd.Series(index=C.index, dtype=float)

    # 计算买入价格和收益率
    for i in BUY_TODAY[BUY_TODAY == 0].index.to_list()[::-1]:
        BUY_PRICE_CLOSE.loc[i] = C.loc[i]
        BUY_PRICE_OPEN.loc[i] = O.loc[i]
        BUY_PRICE_CLOSE.fillna(method='ffill', inplace=True)  # 向下填充无效值
        BUY_PRICE_OPEN.fillna(method='ffill', inplace=True)  # 向下填充无效值
        BUY_PCT = C / BUY_PRICE_CLOSE - 1
        # 循环计算BUY_PCT_MAX
        for k, v in BUY_PCT[i:].items():
            if np.isnan(BUY_PCT_MAX[k]):
                BUY_PCT_MAX[k] = BUY_PCT[i:k].max()

    # 计算当日涨跌幅
    当日涨幅 = (C / REF(C, 1) - 1) * 100

    # 卖出条件1：连续下跌2天
    连续下跌2天 = (当日涨幅 < 0) & (REF(当日涨幅, 1) < 0)

    # 卖出条件2：单日跌幅超过6%
    单日大跌 = 当日涨幅 < -6

    # 卖出条件3：买入后持有超过12天且收益率小于3%
    持有过久收益低 = (BUY_TODAY > 12) & (BUY_PCT < 0.03)

    # 卖出条件4：收益率达到20%止盈
    止盈 = BUY_PCT > 0.20

    # 卖出条件5：亏损超过10%止损
    止损 = BUY_PCT < -0.10

    # 卖出条件6：跌破MA10均线且收益为负
    跌破均线 = (C < MA10) & (BUY_PCT < 0)

    # 卖出条件7：高点回撤超过12%
    高点回撤 = (BUY_PCT_MAX > 0.05) & (BUY_PCT < BUY_PCT_MAX * 0.88)

    # 综合卖出信号
    卖出信号初步 = 连续下跌2天 | 单日大跌 | 持有过久收益低 | 止盈 | 止损 | 跌破均线 | 高点回撤

    卖出信号 = pd.Series([False] * len(C), index=C.index, dtype=bool)

    # 循环，第一次出现卖出信号时执行卖出
    for i in BUY_TODAY[BUY_TODAY == 0].index.to_list()[::-1]:
        for k, v in 卖出信号初步[i:].items():
            # k != i 排除买入信号当日同时产生卖出信号的极端情况
            if k != i and 卖出信号初步[k]:
                卖出信号[k] = True
                break

    return 卖出信号


if __name__ == '__main__':
    # 调试用代码. 此文件不直接执行。通过xuangu.py或celue_save.py调用
    import pandas as pd
    import os
    import user_config as ucfg

    stock_code = '000001'
    start_date = ''
    end_date = ''
    target_date = '2024-01-15'  # 示例：指定2024年1月15日为连涨第3天的目标日期
    
    df_stock = pd.read_csv(ucfg.tdx['csv_lday'] + os.sep + stock_code + '.csv',
                           index_col=None, encoding='gbk', dtype={'code': str})
    df_stock['date'] = pd.to_datetime(df_stock['date'], format='%Y-%m-%d')  # 转为时间格式
    df_stock.set_index('date', drop=False, inplace=True)  # 时间为索引。方便与另外复权的DF表对齐合并

    df_hs300 = pd.read_csv(ucfg.tdx['csv_index'] + '/000300.csv', index_col=None, encoding='gbk', dtype={'code': str})
    df_hs300['date'] = pd.to_datetime(df_hs300['date'], format='%Y-%m-%d')  # 转为时间格式
    df_hs300.set_index('date', drop=False, inplace=True)  # 时间为索引。方便与另外复权的DF表对齐合并
    
    if '09:00:00' < time.strftime("%H:%M:%S", time.localtime()) < '16:00:00':
        df_today = func.get_tdx_lastestquote((1, '000300'))
        df_hs300 = func.update_stockquote('000300', df_hs300, df_today)
    HS300_信号 = 策略HS300(df_hs300)

    if target_date and pd.to_datetime(target_date) in HS300_信号.index:
        if not HS300_信号.loc[pd.to_datetime(target_date)]:
            print(f'目标日期{target_date}HS300不满足买入条件')
    else:
        if not HS300_信号.iat[-1]:
            print('今日HS300不满足买入条件，停止选股')

    if '09:00:00' < time.strftime("%H:%M:%S", time.localtime()) < '16:00:00':
        df_today = func.get_tdx_lastestquote(stock_code)
        df_stock = func.update_stockquote(stock_code, df_stock, df_today)
        
    celue1_fast = 策略1(df_stock, mode='fast', start_date=start_date, end_date=end_date)
    celue1 = 策略1(df_stock, mode='', start_date=start_date, end_date=end_date)
    celue2 = 策略2(df_stock, HS300_信号, start_date=start_date, end_date=end_date, target_date=target_date)
    celue_sell = 卖策略(df_stock, celue2, start_date=start_date, end_date=end_date, target_date=target_date)
    
    print(f'{stock_code} 连续三天上涨策略结果 target_date={target_date}:')
    print(f'celue1_fast={celue1_fast}')
    print(f'celue1={celue1.iat[-1] if hasattr(celue1, "iat") else celue1}')
    print(f'celue2={celue2}')
    print(f'celue_sell={celue_sell.iat[-1] if hasattr(celue_sell, "iat") else "N/A"}')
