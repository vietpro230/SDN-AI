import networkx as nx

class SDNController:
    def __init__(self, switch_ids, power_active=100, power_sleep=10, capacity_per_switch=1000000):
        """
        Initializes the SDN Controller (Simulating Ryu Controller logic).

        This controller implements Dynamic Resource Allocation to optimize energy efficiency
        in a Data Center network. It uses AI-predicted traffic loads to make decisions
        about putting switches into sleep mode while maintaining network connectivity.
        """
        self.switches = switch_ids
        self.power_active = power_active
        self.power_sleep = power_sleep
        self.capacity = capacity_per_switch

        # --- Infrastructure Layer Representation ---
        self.topology = nx.Graph()
        self.topology.add_nodes_from(switch_ids)
        # Create a ring topology with a cross-link for redundancy
        if len(switch_ids) > 1:
            for i in range(len(switch_ids)):
                self.topology.add_edge(switch_ids[i], switch_ids[(i+1) % len(switch_ids)])
            self.topology.add_edge(switch_ids[0], switch_ids[len(switch_ids)//2])

    # --- Layer 1: Monitoring ---
    def link_measurement(self, network_state):
        """
        Corresponds to 'Link measurement'.
        Measures link latency, bandwidth availability, etc.
        """
        # Simulation: Just returning state, but in real SDN this uses LLDP/Echo.
        return network_state

    def traffic_monitoring(self, current_traffic_data):
        """
        Corresponds to 'Traffic monitoring'.
        In a real SDN, this would query OpenFlow counters (OFPStatsRequest).
        """
        # In this simulation, we pass through the data from the generator
        return current_traffic_data

    # --- Layer 2: Prediction is handled by the external AI Model ---

    # --- Layer 3: Energy Efficiency Optimization ---
    def energy_efficiency_optimization(self, predicted_loads):
        """
        Corresponds to 'Energy efficiency optimization'.
        Input: Predicted traffic from the AI model.
        Output: A decision plan (which nodes to sleep, which to keep active).
        """
        decision = {
            'active': [],
            'sleep': [],
            'reroute_needed': False
        }

        # Sort switches by load (ascending) to try sleeping the least loaded ones
        sorted_switches = sorted(predicted_loads.items(), key=lambda x: x[1])

        # Start assuming all are active
        current_active = set(self.switches)

        for switch_id, load in sorted_switches:
            # Logic: If load is low (< 10% capacity), try to sleep
            if load < (0.1 * self.capacity):
                # Temporarily remove to check connectivity
                current_active.remove(switch_id)

                if len(current_active) == 0:
                    current_active.add(switch_id)
                    continue

                subgraph = self.topology.subgraph(list(current_active))

                # Constraint: Network must remain connected
                if nx.is_connected(subgraph):
                    # Safe to sleep
                    pass
                else:
                    # Must keep active to maintain connectivity
                    current_active.add(switch_id)
            else:
                # Load too high, keep active
                pass

        decision['active'] = list(current_active)
        decision['sleep'] = list(set(self.switches) - current_active)

        if len(decision['sleep']) > 0:
            decision['reroute_needed'] = True

        return decision

    # --- Layer 4: Actions (Route/Reroute & Sleep/Wake) ---
    def route_reroute(self, decision):
        """
        Corresponds to 'Route/Reroute'.
        If topology changes (nodes sleep), we must recalculate paths.
        """
        if decision['reroute_needed']:
            # Simulation of installing new flow rules
            # active_topology = self.topology.subgraph(decision['active'])
            # new_paths = nx.all_pairs_shortest_path(active_topology)
            return "Topology changed. Recalculating and installing new routes..."
        return "Topology unchanged. Existing routes valid."

    def sleep_wake_up_command(self, decision):
        """
        Corresponds to 'Sleep / Wake_up command'.
        Sends commands to the infrastructure.
        """
        commands = []
        for sw in decision['sleep']:
            commands.append(f"CMD: Set Switch {sw} -> SLEEP MODE")
        for sw in decision['active']:
            commands.append(f"CMD: Set Switch {sw} -> ACTIVE MODE")
        return commands

    # --- Metrics ---
    def calculate_energy(self, active_switches_list):
        num_active = len(active_switches_list)
        num_total = len(self.switches)
        num_sleep = num_total - num_active
        return (num_active * self.power_active) + (num_sleep * self.power_sleep)

    def get_standard_energy(self):
        return len(self.switches) * self.power_active
