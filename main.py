import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_and_process_data
from model import TrafficPredictor
from sdn_controller import SDNController

def main():
    print("Loading and processing data...")
    df = load_and_process_data('src/data.csv')
    switch_ids = df.columns.tolist()
    print(f"Found switches: {switch_ids}")

    print("Initializing and training model...")
    # look_back = 3 steps (e.g., 30 seconds context)
    predictor = TrafficPredictor(look_back=3, n_features=len(switch_ids))
    predictor.train(df)

    print("Starting simulation...")
    # Capacity: Set high enough so we don't always saturate, but low enough to matter.
    # Max load in sample data was ~1.8GB/s (1.8e9).
    # Let's set capacity to 2GB/s (2e9) per switch.
    controller = SDNController(switch_ids, power_active=200, power_sleep=20, capacity_per_switch=2e9)

    # Simulation loop on the last 20% of data (Test set)
    test_size = int(len(df) * 0.2)
    test_data = df.iloc[-test_size:]

    results = []

    total_energy_standard = 0
    total_energy_optimized = 0

    # We need 'look_back' data points before the test set to start predicting
    # Get the full dataset values
    full_data_values = df.values
    start_index = len(df) - test_size

    for i in range(len(test_data)):
        current_idx = start_index + i

        # --- Step 1: Traffic Monitoring ---
        # Get recent data (simulating the monitoring module)
        recent_data_raw = full_data_values[current_idx - predictor.look_back : current_idx]
        recent_data = controller.traffic_monitoring(recent_data_raw)

        # --- Step 2: Traffic Prediction Model ---
        # Predict next load for all switches
        predicted_loads_array = predictor.predict(recent_data)
        # Map array back to switch IDs
        predicted_loads = {switch_id: load for switch_id, load in zip(switch_ids, predicted_loads_array)}

        # --- Step 3: Energy Efficiency Optimization ---
        decision = controller.energy_efficiency_optimization(predicted_loads)

        # --- Step 4: Actions (Route/Reroute & Sleep/Wake) ---
        # In a real system, these would be sent to the infrastructure
        route_status = controller.route_reroute(decision)
        # commands = controller.sleep_wake_up_command(decision) # Uncomment to see commands

        # Calculate Energy for Statistics
        active_switches_opt = decision['active']
        energy_opt = controller.calculate_energy(active_switches_opt)

        # Standard Policy: All ON
        energy_std = controller.get_standard_energy()

        total_energy_optimized += energy_opt
        total_energy_standard += energy_std

        # Calculate total network load for plotting
        total_load = sum(predicted_loads.values())

        results.append({
            'timestamp': test_data.index[i],
            'total_load': total_load,
            'active_switches_count': len(active_switches_opt),
            'energy_optimized': energy_opt,
            'energy_standard': energy_std
        })
    results_df = pd.DataFrame(results)

    print(f"\nSimulation Complete.")
    print(f"Total Energy (Standard): {total_energy_standard/1000:.2f} kW")
    print(f"Total Energy (Optimized): {total_energy_optimized/1000:.2f} kW")
    if total_energy_standard > 0:
        print(f"Energy Savings: {((total_energy_standard - total_energy_optimized) / total_energy_standard) * 100:.2f}%")

    # Plotting
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(results_df['timestamp'], results_df['total_load'], label='Total Predicted Network Load')
    plt.title('Network Traffic Load')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(results_df['timestamp'], results_df['active_switches_count'], label='Active Switches (AI)', color='green', marker='o')
    plt.axhline(y=len(switch_ids), color='red', linestyle=':', label='Active Switches (Standard)')
    plt.title('Resource Allocation (Switch Sleeping)')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(results_df['timestamp'], results_df['energy_optimized'], label='Energy (AI)', color='green')
    plt.plot(results_df['timestamp'], results_df['energy_standard'], label='Energy (Standard)', color='red', linestyle=':')
    plt.title('Energy Consumption Comparison')
    plt.legend()

    plt.tight_layout()
    plt.savefig('simulation_results.png')
    print("Results plot saved to 'simulation_results.png'")

if __name__ == "__main__":
    main()
