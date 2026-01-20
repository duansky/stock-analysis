"""
连续涨8天选股策略
基于CeLue模板.py修改，实现连续涨8天的选股逻辑

个人实际策略不分享。

MA函数返回的是值。
其余函数输入、输出都是序列。只有序列才能表现出来和通达信一样的判断逻辑。
HHV/LLV/COUNT使用了rolling函数，性能极差，慎用。

"""
import numpy as np
import talib
import time
import func
import pandas as pd
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
    连续涨8天策略的基础筛选条件
    
    :param DataFrame df:输入具体一个股票的DataFrame数据表。时间列为索引。
    :param mode :str 'fast'为快速模式，只处理当日数据，用于开盘快速筛选股票。和策略2结合使用时不能用fast模式
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
    if {'换手率'}.issubset(df.columns):  # 无换手率列的股票，只可能是近几个月的新股。
        换手率 = df['换手率']
    else:
        换手率 = 0

    if mode == 'fast':
        # 天数不足13天，收盘价小于3直接返回FALSE
        if C.shape[0] < 13 or C.iat[-1] < 3:
            return False

        # 基础条件：股价大于3元，有足够的交易数据
        基础条件 = C.iat[-1] > 3

        # 排除涨停股票
        if df['code'][0][0:2] == "68" or df['code'][0][0:2] == "30":
            涨停价 = 1.2
        else:
            涨停价 = 1.1
        非涨停 = ~((C.iat[-1]+0.01) >= np.ceil((np.floor(REF(C, 1).iat[-1]*1000*涨停价)-4)/10)/100)

        result = 基础条件 & 非涨停
    else:
        # 基础条件：股价大于3元，有足够的交易数据
        基础条件 = (C > 3) & (BARSLAST(C == 0) > 13)

        # 排除涨停股票
        if df['code'][0][0:2] == "68" or df['code'][0][0:2] == "30":
            涨停价 = 1.2
        else:
            涨停价 = 1.1
        非涨停 = ~((C+0.01) >= np.ceil((np.floor(REF(C, 1)*1000*涨停价)-4)/10)/100)

        result = 基础条件 & 非涨停
    return result


def 策略2(df, HS300_信号, start_date='', end_date='', target_date=''):
    """
    连续涨8天选股策略主逻辑

    :param DataFrame df:输入具体一个股票的DataFrame数据表。时间列为索引。
    :param date start_date:可选。留空从头开始。2020-10-10格式，策略指定从某日期开始
    :param date end_date:可选。留空到末尾。2020-10-10格式，策略指定到某日期结束
    :param date target_date:可选。指定作为连涨第8天的目标日期。留空则使用end_date作为目标日期
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

    if df.shape[0] < 13:  # 小于13日 直接返回false序列
        return pd.Series([False] * len(df), index=df.index, dtype=bool)

    # 根据df的索引重建HS300信号，为了与股票交易日期一致
    HS300_信号 = pd.Series(HS300_信号, index=df.index, dtype=bool).dropna()

    O = df['open']
    H = df['high']
    L = df['low']
    C = df['close']

    # 计算每日涨跌幅
    当日涨幅 = (C / REF(C, 1) - 1) * 100

    # 连续涨8天的核心逻辑
    # 目标日期上涨
    目标日期上涨 = 当日涨幅 > 0

    # 连续8天都上涨（以target_date为第8天）
    连续8天上涨 = (目标日期上涨 &
                  (REF(当日涨幅, 1) > 0) &
                  (REF(当日涨幅, 2) > 0) &
                  (REF(当日涨幅, 3) > 0) &
                  (REF(当日涨幅, 4) > 0) &
                  (REF(当日涨幅, 5) > 0) &
                  (REF(当日涨幅, 6) > 0) &
                  (REF(当日涨幅, 7) > 0))

    # 基础策略1的条件
    策略1条件 = 策略1(df, start_date, end_date)

    # 成交量条件：避免无量上涨
    if 'amount' in df.columns:
        成交额均值 = SMA(df['amount'], 8)
        成交量条件 = df['amount'] > 成交额均值 * 0.8
    else:
        成交量条件 = pd.Series([True] * len(df), index=df.index, dtype=bool)

    # 涨幅控制：单日涨幅不超过9%，避免异常波动
    涨幅合理 = 当日涨幅 < 9

    # 综合条件
    买入信号 = (HS300_信号 &
              策略1条件 &
              连续8天上涨 &
              成交量条件 &
              涨幅合理)

    # 如果指定了target_date，只返回该日期的信号
    if target_date in 买入信号.index:
        return 买入信号.loc[target_date]
    else:
        # 如果target_date不在数据范围内，返回False
        return False


def 卖策略(df, 策略2, start_date='', end_date='', target_date=''):
    """
    连续涨8天策略的卖出逻辑
    
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

    # 计算买入后的天数和价格
    BUY_TODAY = BARSLAST(策略2)
    BUY_PRICE_CLOSE = pd.Series(index=C.index, dtype=float)
    BUY_PCT = pd.Series(index=C.index, dtype=float)
    
    # 计算买入价格和收益率
    for i in BUY_TODAY[BUY_TODAY == 0].index.to_list()[::-1]:
        BUY_PRICE_CLOSE.loc[i] = C.loc[i]
        BUY_PRICE_CLOSE.fillna(method='ffill', inplace=True)  # 向下填充无效值
        BUY_PCT = C / BUY_PRICE_CLOSE - 1

    # 计算当日涨跌幅
    当日涨幅 = (C / REF(C, 1) - 1) * 100

    # 卖出条件1：连续下跌2天
    连续下跌2天 = (当日涨幅 < 0) & (REF(当日涨幅, 1) < 0)

    # 卖出条件2：单日跌幅超过5%
    单日大跌 = 当日涨幅 < -5

    # 卖出条件3：买入后持有超过10天且收益率小于2%
    持有过久收益低 = (BUY_TODAY > 10) & (BUY_PCT < 0.02)

    # 卖出条件4：收益率达到15%止盈
    止盈 = BUY_PCT > 0.15

    # 卖出条件5：亏损超过8%止损
    止损 = BUY_PCT < -0.08

    # 综合卖出信号
    卖出信号初步 = 连续下跌2天 | 单日大跌 | 持有过久收益低 | 止盈 | 止损
    
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
    target_date = '2024-01-15'  # 示例：指定2024年1月15日为连涨第8天的目标日期
    
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
    print(f'{stock_code} target_date={target_date} celue1_fast={celue1_fast} celue1={celue1.iat[-1] if hasattr(celue1, "iat") else celue1} celue2={celue2} celue_sell={celue_sell.iat[-1] if hasattr(celue_sell, "iat") else "N/A"}')