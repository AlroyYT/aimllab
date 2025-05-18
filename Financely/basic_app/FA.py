import pandas as pd
import datetime
import yfinance as yf

def piotroski(ticker):
    try:
        # Get financial data
        bs = yf.Ticker(ticker).balance_sheet
        inc = yf.Ticker(ticker).financials
        cf = yf.Ticker(ticker).cashflow
        
        # Print available columns for debugging
        print(f"Balance Sheet columns: {bs.index.tolist()}")
        
        # Function to safely get value with fallbacks
        def safe_get(dataframe, primary_key, fallback_keys=None, default=0):
            try:
                # Use iloc to avoid FutureWarning
                return dataframe.loc[primary_key].iloc[0]
            except (KeyError, IndexError, AttributeError):
                if fallback_keys:
                    for key in fallback_keys:
                        try:
                            return dataframe.loc[key].iloc[0]
                        except (KeyError, IndexError, AttributeError):
                            continue
                print(f"Warning: Could not find {primary_key} or any fallbacks")
                return default
                
        def safe_get_prev(dataframe, primary_key, idx=1, fallback_keys=None, default=0):
            try:
                # Use iloc to avoid FutureWarning
                return dataframe.loc[primary_key].iloc[idx]
            except (KeyError, IndexError, AttributeError):
                if fallback_keys:
                    for key in fallback_keys:
                        try:
                            return dataframe.loc[key].iloc[idx]
                        except (KeyError, IndexError, AttributeError):
                            continue
                print(f"Warning: Could not find previous period {primary_key} or any fallbacks")
                return default
        
        # Get balance sheet items with fallbacks
        longTermDebt = safe_get(bs, 'Long Term Debt', 
                               ['Total Long Term Debt', 'Long-Term Debt', 'LT Debt', 'Long Term Debt And Capital Lease Obligation'])
        
        longTermDebtPre = safe_get_prev(bs, 'Long Term Debt', 1,
                                      ['Total Long Term Debt', 'Long-Term Debt', 'LT Debt', 'Long Term Debt And Capital Lease Obligation'])
        
        totalAssets = safe_get(bs, 'Total Assets')
        totalAssetsPre = safe_get_prev(bs, 'Total Assets', 1)
        totalAssetsPre2 = safe_get_prev(bs, 'Total Assets', 2)
        
        currentAssets = safe_get(bs, 'Total Current Assets', 
                                ['Current Assets', 'Total Current Assets'])
        
        currentAssetsPre = safe_get_prev(bs, 'Total Current Assets', 1,
                                       ['Current Assets', 'Total Current Assets'])
        
        currentLiabilities = safe_get(bs, 'Total Current Liabilities', 
                                     ['Current Liabilities', 'Total Current Liabilities'])
        
        currentLiabilitiesPre = safe_get_prev(bs, 'Total Current Liabilities', 1,
                                            ['Current Liabilities', 'Total Current Liabilities'])
        
        # Get income statement items
        revenue = safe_get(inc, 'Total Revenue', ['Revenue', 'Total Revenue'])
        revenuePre = safe_get_prev(inc, 'Total Revenue', 1, ['Revenue', 'Total Revenue'])
        
        grossProfit = safe_get(inc, 'Gross Profit')
        grossProfitPre = safe_get_prev(inc, 'Gross Profit', 1)
        
        netIncome = safe_get(inc, 'Net Income', ['Net Income', 'Net Income Common Stockholders'])
        netIncomePre = safe_get_prev(inc, 'Net Income', 1, ['Net Income', 'Net Income Common Stockholders'])
        
        # Get cash flow items
        operatingCashFlow = safe_get(cf, 'Total Cash From Operating Activities', 
                                    ['Operating Cash Flow', 'Cash From Operations'])
        
        operatingCashFlowPre = safe_get_prev(cf, 'Total Cash From Operating Activities', 1,
                                           ['Operating Cash Flow', 'Cash From Operations'])
        
        # Get equity items
        commonStock = safe_get(bs, 'Common Stock', 
                              ['Common Stock', 'Common Shares Outstanding', 'Total Common Shares Outstanding'])
        
        commonStockPre = safe_get_prev(bs, 'Common Stock', 1,
                                     ['Common Stock', 'Common Shares Outstanding', 'Total Common Shares Outstanding'])
        
        # Calculate Piotroski F-Score components
        # Profitability
        ROAFS = int(netIncome > 0)  # Modified to just check if positive
        CFOFS = int(operatingCashFlow > 0)
        
        # Deal with potential division by zero
        avg_assets_current = (totalAssets + totalAssetsPre) / 2 if (totalAssets + totalAssetsPre) != 0 else 1
        avg_assets_prev = (totalAssetsPre + totalAssetsPre2) / 2 if (totalAssetsPre + totalAssetsPre2) != 0 else 1
        
        ROADFS = int((netIncome/avg_assets_current) > (netIncomePre/avg_assets_prev)) if netIncomePre != 0 and avg_assets_prev != 0 else 0
        CFOROAFS = int((operatingCashFlow/totalAssets) > (netIncome/avg_assets_current)) if totalAssets != 0 and avg_assets_current != 0 else 0
        
        # Leverage, Liquidity, and Source of Funds
        LTDFS = int(longTermDebt <= longTermDebtPre)
        
        # Avoid division by zero
        current_ratio_current = currentAssets/currentLiabilities if currentLiabilities != 0 else 0
        current_ratio_prev = currentAssetsPre/currentLiabilitiesPre if currentLiabilitiesPre != 0 else 0
        
        CRFS = int(current_ratio_current > current_ratio_prev)
        NSFS = int(commonStock <= commonStockPre)
        
        # Operating Efficiency
        # Avoid division by zero
        gm_current = grossProfit/revenue if revenue != 0 else 0
        gm_prev = grossProfitPre/revenuePre if revenuePre != 0 else 0
        
        GMFS = int(gm_current > gm_prev)
        
        asset_turnover_current = revenue/avg_assets_current if avg_assets_current != 0 else 0
        asset_turnover_prev = revenuePre/avg_assets_prev if avg_assets_prev != 0 else 0
        
        ATOFS = int(asset_turnover_current > asset_turnover_prev)
        
        # Sum up all components
        f_score = ROAFS + CFOFS + ROADFS + CFOROAFS + LTDFS + CRFS + NSFS + GMFS + ATOFS
        
        # Print breakdown for debugging
        print(f"F-Score Components for {ticker}:")
        print(f"ROAFS: {ROAFS}, CFOFS: {CFOFS}, ROADFS: {ROADFS}, CFOROAFS: {CFOROAFS}")
        print(f"LTDFS: {LTDFS}, CRFS: {CRFS}, NSFS: {NSFS}, GMFS: {GMFS}, ATOFS: {ATOFS}")
        
        return f_score
        
    except Exception as e:
        print(f"Error calculating Piotroski F-Score for {ticker}: {e}")
        return 0