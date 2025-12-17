"""
Utility functions extracted from the main analysis script.
All original inline comments are preserved to allow line-by-line tracking.
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymannkendall as mk
#%%
def find_events_and_lags(df, precip_col='Gauge Precip (in)', q=0.9, min_gap_days=10, window_days=10, require_single_day=True):
    """
    df: DataFrame with precip and discharge columns indexed by Date
    q: percentile for threshold 
    min_gap_days: minimum days between independent events
    window_days: days after event to search for discharge peak
    require_single_day: if True, skip events where the window contains multiple days >= threshold, ensures that there aren't multiple 
                        significant precip days in the window
    returns: DataFrame of events with lag statistics
    """

    working = df[[precip_col, 'Discharge (cfs)']].dropna()  # dropping rows with nan in precip or discharge
    threshold = working[working[precip_col] > 0][precip_col].quantile(q)  # computing threshold based on percentile q

    event_starts = []
    skip_until = None

    for date, rain in working[precip_col].items():  # loops through each row in precip column based on date index and precip value
        if skip_until is not None and date <= skip_until:  # skip this if it is too close to a different event
            continue
        if rain >= threshold:  # if precip exceeds threshold, it is considered an event start
            event_starts.append(date)  # append date to list
            skip_until = date + pd.Timedelta(days=min_gap_days)  # skip ahead to enforce minimum gap between events

    events = []

    for start in event_starts:  # loops through each event start date
        end = start + pd.Timedelta(days=window_days)  # defines the window to search for peak discharge
        window = working.loc[start:end]  # extracts data in the defined window
        if window.empty:
            continue

        n_sig = (window[precip_col] >= threshold).sum()  # counts number of significant precip days in window
        if require_single_day and n_sig > 1:  # skips event if multiple significant precip days in window
            continue

        peak_date = window['Discharge (cfs)'].idxmax()  # identifies date of peak discharge in window
        lag_days = (peak_date - start).days  # computes lag in days between event start and peak discharge
        peak_q = window.loc[peak_date, 'Discharge (cfs)']  # gets peak discharge value

        events.append({
            'event_date': start,
            'peak_date': peak_date,
            'lag_days': lag_days,
            'peak_discharge': peak_q,
            'rain_amount': working.loc[start, precip_col],
            'n_sig_days_in_window': n_sig
        })  # appends event details to list

    events_df = pd.DataFrame(events)  # converts list of events to df
    return events_df, threshold  # returns events dataframe and threshold value
#%%
# runs a non-parametric Mann-Kendall trend test on a given series
def mk_summary(series, label):
    result = mk.original_test(series)
    print(f"{label}:")
    print(f"  Trend: {result.trend}")
    print(f"  p-value: {result.p:.5f}\n")
    return result
#%%
# plots annual totals by water year and overlays Mann-Kendall trend information
def plot_mk_trend_water_year(project_complete, colname, title):
    """
    Plot annual totals by water year with Mann-Kendall trend test.
    Parameters:
    - project_complete: DataFrame with daily data, already filtered for complete water years
    - colname: column to analyze (e.g., 'Gauge Precip (in)')
    - title: plot title
    """

    annual_total = project_complete.groupby('water_year')[colname].sum()  # sum by water year
    result = mk.original_test(annual_total)  # run Mann-Kendall test

    plt.figure(figsize=(12, 5))
    plt.plot(annual_total.index, annual_total.values, marker='o', linestyle='-', label='Annual Total')

    z = np.polyfit(range(len(annual_total)), annual_total.values, 1)  # linear fit for visual trend
    plt.plot(annual_total.index, np.polyval(z, range(len(annual_total))), '--', color='red', label='Trend Line')

    plt.title(f'{title}\nTrend: {result.trend}, p={result.p:.5f}')
    plt.xlabel('Water Year')
    plt.ylabel('Annual Total (in)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return result