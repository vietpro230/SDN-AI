import pandas as pd
import numpy as np

def load_and_process_data(filepath='src/data.csv'):
    """
    Loads the traffic data, aggregates it by switch (datapath_id),
    and prepares it for time-series prediction.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # Fallback for running from different directories
        df = pd.read_csv('../' + filepath)

    # Ensure timestamp is sorted
    df = df.sort_values('timestamp')

    # We want to predict traffic load (byte_count_per_second) for each switch (datapath_id)
    # Group by timestamp and datapath_id, summing the traffic
    # Since timestamps might be slightly different for flows, we bin them.
    # The data seems to be in ~10 second intervals (1763025816, 1763025826...)

    # Round timestamp to nearest 10 seconds to align them
    df['time_bin'] = df['timestamp'].round(-1)

    pivot_df = df.pivot_table(
        index='time_bin',
        columns='datapath_id',
        values='byte_count_per_second',
        aggfunc='sum'
    )

    # Fill missing values with 0 (no traffic)
    pivot_df = pivot_df.fillna(0)

    # Normalize the data (optional but good for LSTM)
    # We will return the raw dataframe and let the model handle scaling
    return pivot_df

if __name__ == "__main__":
    df = load_and_process_data()
    print("Processed Data Shape:", df.shape)
    print(df.head())
