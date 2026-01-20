"""
停机坪选股策略
策略描述：
1. 最近15日有涨幅大于9.5%，且必须是放量上涨
2. 紧接的下个交易日必须高开，收盘价必须上涨，且与开盘价不能大于等于相差3%
3. 接下2、3个交易日必须高开，收盘价必须上涨，且与开盘价不能大于等于相差3%，且每天涨跌幅在5%间

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
    停机坪策略1：基础筛选条件
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
    V = df['vol']
    
    if {'换手率'}.issubset(df.columns):  # 无换手率列的股票，只可能是近几个月的新股。
        换手率 = df['换手率']
    else:
        换手率 = 0

    if mode == 'fast':
        # 天数不足500天，收盘价小于9直接返回FALSE
        if C.shape[0] < 500 or C.iat[-1] < 9:
            return False

        # 基础条件：排除涨停股票
        if df['code'][0][0:2] == "68" or df['code'][0][0:2] == "30":
            涨停价 = 1.2
        else:
            涨停价 = 1.1
        非涨停 = ~((C+0.01) >= np.ceil((np.floor(REF(C, 1)*1000*涨停价)-4)/10)/100)
        
        result = 非涨停.iat[-1]
    else:
        # TJ01: 基础条件
        TJ01 = (BARSLAST(C == 0) > 500) & (df['close'] > 9)

        # TJ02: 排除涨停股票
        if df['code'][0][0:2] == "68" or df['code'][0][0:2] == "30":
            涨停价 = 1.2
        else:
            涨停价 = 1.1
        TJ02 = ~((C+0.01) >= np.ceil((np.floor(REF(C, 1)*1000*涨停价)-4)/10)/100)

        result = TJ01 & TJ02
    return result


def 策略2(df, HS300_信号, start_date='', end_date=''):
    """
    停机坪策略2：核心选股逻辑
    1. 最近15日有涨幅大于9.5%，且必须是放量上涨
    2. 紧接的下个交易日必须高开，收盘价必须上涨，且与开盘价不能大于等于相差3%
    3. 接下2、3个交易日必须高开，收盘价必须上涨，且与开盘价不能大于等于相差3%，且每天涨跌幅在5%间
    
    :param DataFrame df:输入具体一个股票的DataFrame数据表。时间列为索引。
    :param HS300_信号: HS300信号序列
    :param date start_date:可选。留空从头开始。2020-10-10格式，策略指定从某日期开始
    :param date end_date:可选。留空到末尾。2020-10-10格式，策略指定到某日期结束
    :return bool: 截止日期这天，策略是否触发。true触发，false不触发
    """

    if start_date == '':
        start_date = df.index[0]  # 设置为df第一个日期
    if end_date == '':
        end_date = df.index[-1]  # 设置为df最后一个日期
    df = df.loc[start_date:end_date]

    if df.shape[0] < 251:  # 小于250日 直接返回false序列
        return pd.Series(index=df.index, dtype=bool)

    # 根据df的索引重建HS300信号，为了与股票交易日期一致
    HS300_信号 = pd.Series(HS300_信号, index=df.index, dtype=bool).dropna()

    O = df['open']
    H = df['high']
    L = df['low']
    C = df['close']
    V = df['vol']
    
    if {'换手率'}.issubset(df.columns):
        换手率 = df['换手率']
    else:
        换手率 = pd.Series(0, index=df.index)

    # 计算涨跌幅
    涨跌幅 = (C / REF(C, 1) - 1) * 100
    
    # 计算成交量均线
    成交量均线20 = SMA(V, 20)
    
    # 条件1：最近15日有涨幅大于9.5%，且必须是放量上涨
    # 使用HHV函数计算15日内最高收盘价
    最近15日最高价 = HHV(C, 15)
    最近15日最低价 = LLV(C, 15)
    最近15日最大涨幅 = (最近15日最高价 / 最近15日最低价 - 1) * 100
    
    # 检查15日内是否有放量上涨（成交量大于20日均线且当日上涨）
    放量上涨日 = (V > 成交量均线20) & (涨跌幅 > 0)
    最近15日有放量上涨 = COUNT(放量上涨日, 15) > 0
    
    TJ01 = (最近15日最大涨幅 > 9.5) & 最近15日有放量上涨
    
    # 条件2：紧接的下个交易日必须高开，收盘价必须上涨，且与开盘价不能大于等于相差3%
    # 高开：今日开盘价 > 昨日收盘价
    高开 = O > REF(C, 1)
    # 收盘上涨：今日收盘价 > 今日开盘价
    收盘上涨 = C > O
    # 开收盘价差小于3%
    开收盘价差 = abs((C - O) / O * 100)
    
    TJ02 = 高开 & 收盘上涨 & (开收盘价差 < 3)
    
    # 条件3：接下来的2、3个交易日条件检查
    # 由于无法预知未来，这里检查历史数据中满足前两个条件后的后续表现
    TJ03 = pd.Series(index=df.index, dtype=bool, data=False)
    
    # 遍历每个可能的买入点，检查后续2个交易日是否满足条件
    for i in range(len(df) - 3):
        if i >= 15:  # 确保有足够的历史数据计算15日条件
            # 检查当前位置是否满足前两个条件
            if TJ01.iloc[i] and TJ02.iloc[i]:
                # 检查接下来2个交易日的表现
                next_days_ok = True
                for j in range(1, 3):  # 检查第1、2个交易日
                    if i + j < len(df):
                        # 下一日高开
                        next_高开 = O.iloc[i + j] > C.iloc[i + j - 1]
                        # 下一日收盘上涨
                        next_收盘上涨 = C.iloc[i + j] > O.iloc[i + j]
                        # 下一日开收盘价差小于3%
                        next_开收盘价差 = abs((C.iloc[i + j] - O.iloc[i + j]) / O.iloc[i + j] * 100)
                        # 下一日涨跌幅在5%以内
                        next_涨跌幅 = abs(涨跌幅.iloc[i + j])
                        
                        day_condition = (next_高开 and next_收盘上涨 and 
                                       next_开收盘价差 < 3 and next_涨跌幅 < 5)
                        
                        if not day_condition:
                            next_days_ok = False
                            break
                    else:
                        next_days_ok = False
                        break
                
                if next_days_ok:
                    TJ03.iloc[i] = True
    
    # 策略1基础条件
    TJP1 = 策略1(df, start_date, end_date)
    
    # 综合判断：当前满足条件1和2，且历史数据显示这种模式后续表现良好
    当前停机坪信号 = TJ01 & TJ02
    
    # 最终信号：结合HS300信号、基础条件和停机坪信号
    停机坪信号 = HS300_信号 & TJP1 & 当前停机坪信号
    
    # 避免重复信号：10日内只出现一次信号
    停机坪信号_计数 = COUNT(停机坪信号, 10)
    最终信号 = 停机坪信号 & (REF(停机坪信号_计数, 1) == 0)

    return 最终信号

def 卖策略(df, 策略2, start_date='', end_date=''):
    """
    停机坪卖出策略
    :param df: 个股Dataframe
    :param 策略2: 买入策略2
    :param start_date:
    :param end_date:
    :return: 卖出策略序列
    """

    if True not in 策略2.to_list():  # 买入策略2 没有买入点
        return pd.Series(index=策略2.index, dtype=bool)

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

    # SELL01: 跌破MA10且亏损超过5%
    SELL01 = (C < MA10) & (BUY_PCT < -0.05)

    # SELL02: 高点回撤超过8%（适合停机坪策略的短期操作）
    SELL02 = (BUY_PCT_MAX > 0.05) & (BUY_PCT < BUY_PCT_MAX * 0.92)

    # SELL03: 持股超过7天且收益小于2%（停机坪策略追求快进快出）
    SELL03 = (BUY_TODAY > 7) & (BUY_PCT < 0.02)

    # SELL04: 出现跳空缺口向下
    SELL04 = (H < REF(L, 1)) & (BUY_PCT > 0)

    # SELL05: 连续3日收阴线
    连续收阴 = (C < O) & (REF(C, 1) < REF(O, 1)) & (REF(C, 2) < REF(O, 2))
    SELL05 = 连续收阴 & (BUY_TODAY > 2)

    # 综合卖出信号
    SELLSIGN01 = SELL01 | SELL02 | SELL03 | SELL04 | SELL05
    SELLSIGN = pd.Series(index=C.index, dtype=bool)
    
    # 循环，第一次出现SELLSIGN01=True时，SELLSIGN[k] = True并结束循环
    for i in BUY_TODAY[BUY_TODAY == 0].index.to_list()[::-1]:
        for k, v in SELLSIGN01[i:].items():
            # k != i 排除买入信号当日同时产生卖出信号的极端情况
            if k != i and SELLSIGN01[k]:
                SELLSIGN[k] = True
                break

    return SELLSIGN


if __name__ == '__main__':
    # 调试用代码. 此文件不直接执行。通过xuangu.py或celue_save.py调用
    import pandas as pd
    import os
    import user_config as ucfg

    stock_code = '000887'
    start_date = ''
    end_date = ''
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

    if not HS300_信号.iat[-1]:
        print('今日HS300不满足买入条件，停止选股')

    if '09:00:00' < time.strftime("%H:%M:%S", time.localtime()) < '16:00:00':
        df_today = func.get_tdx_lastestquote(stock_code)
        df_stock = func.update_stockquote(stock_code, df_stock, df_today)
        
    celue1_fast = 策略1(df_stock, mode='fast', start_date=start_date, end_date=end_date)
    celue1 = 策略1(df_stock, mode='', start_date=start_date, end_date=end_date)
    celue2 = 策略2(df_stock, HS300_信号, start_date=start_date, end_date=end_date)
    celue_sell = 卖策略(df_stock, celue2, start_date=start_date, end_date=end_date)
    
    print(f'{stock_code} 停机坪策略结果:')
    print(f'celue1_fast={celue1_fast}')
    print(f'celue1={celue1.iat[-1]}') 
    print(f'celue2={celue2.iat[-1]}')
    print(f'celue_sell={celue_sell.iat[-1]}')