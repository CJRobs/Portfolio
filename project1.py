import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import yfinance as yf
from datetime import datetime, timedelta
import math
from itertools import permutations

def show():
    """Main function to display Currency Arbitrage Detection project"""
    
    # Project header
    st.markdown('<div class="main-header">Project 1: Currency Exchange Network Analysis: Detecting Arbitrage Opportunities through Graph-Based Optimization</div>', unsafe_allow_html=True)
    
    # Project description
    st.markdown(
        '<div class="project-description">'
        'This project implements a graph-based approach to analyze currency exchange rates and '
        'identify potential arbitrage opportunities in the forex market. By representing currencies as nodes '
        'and exchange rates as weighted edges, we can apply algorithms to detect negative cycles that '
        'indicate profitable trading sequences. This demonstrates advanced data analysis, graph theory '
        'application, and financial market knowledge.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Create tabs for organization
    tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Results & Insights"])
    
    with tab1:
        st.header("Project Overview")
        st.write("""
        ### Currency Arbitrage Detection
        
        Currency arbitrage is a trading strategy that exploits inefficiencies in exchange rates between different currencies. 
        In an efficient market, there should be no way to make risk-free profit by exchanging currencies in a cycle, 
        but temporary market inefficiencies can create these opportunities.
        
        This project:
        - Retrieves real-time currency exchange rates from Yahoo Finance
        - Constructs a knowledge graph representing the currency exchange network
        - Applies the Bellman-Ford algorithm to detect negative cycles (arbitrage opportunities)
        - Visualizes the currency network and potential arbitrage paths
        - Calculates the potential profit from each identified arbitrage opportunity
        
        ### Why This Matters
        Financial institutions and traders constantly monitor markets for arbitrage opportunities. While these 
        inefficiencies typically disappear quickly in liquid markets, having automated systems to detect them 
        can provide a competitive advantage. This project demonstrates how graph theory can be applied to 
        financial market analysis.
        """)
        
        # Data sample
        st.subheader("Currency Data Sample")
        
        # Define currencies to use
        all_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 
                      'SGD', 'HKD', 'SEK', 'NOK', 'MXN', 'ZAR', 'BRL', 'INR',
                      'CNY', 'KRW', 'RUB', 'TRY', 'DKK', 'PLN', 'THB', 'ILS']
        
        # Let the user select currencies to include
        with st.expander("Select Currencies to Include"):
            major_currencies = st.multiselect(
                "Select currencies to analyze",
                options=all_currencies,
                default=['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
            )
        
        # Generate all currency pairs
        currency_pairs = []
        for base, quote in permutations(major_currencies, 2):
            if base != quote:
                currency_pairs.append(f"{base}{quote}=X")
        
        # Move the checkbox outside the cached function
        use_simulated_data = st.checkbox("Use simulated data (for testing)", value=False)
        
        # Fetch the latest exchange rate data
        @st.cache_data(ttl=300)  # Cache data for 5 minutes
        def fetch_forex_data(currency_pairs, use_simulated=False):
            try:
                with st.spinner('Fetching currency data from Yahoo Finance...'):
                    if not currency_pairs:
                        raise ValueError("No currency pairs selected")
                    
                    # Use the parameter instead of creating a new checkbox
                    if use_simulated:
                        raise ValueError("Using simulated data as requested")
                    
                    # Get close prices for all currency pairs
                    data = yf.download(currency_pairs, period="1d")
                    
                    # Create our result dataframe
                    result = pd.DataFrame()
                    
                    # Handle different cases based on data shape
                    if len(currency_pairs) == 1:
                        # Single pair case
                        result = pd.DataFrame({
                            'Date': data.index,
                            currency_pairs[0]: data['Close'].values
                        })
                    else:
                        # Multiple pairs case
                        if isinstance(data.index, pd.DatetimeIndex):
                            # Get the latest date
                            latest_date = data.index[-1]
                            
                            # Extract the data for the latest date only
                            latest_data = data.loc[latest_date]
                            
                            # Create a dataframe with one row
                            result = pd.DataFrame({'Date': [latest_date]})
                            
                            # Add each currency pair's close value
                            for pair in currency_pairs:
                                try:
                                    result[pair] = [latest_data['Close'][pair]]
                                except:
                                    # Skip pairs that don't have data
                                    pass
                    
                    return result
                    
            except Exception as e:
                st.warning(f"Error fetching data: {e}. Using simulated data for demonstration.")
                # Generate dummy data for demonstration
                dummy_data = pd.DataFrame({
                    'Date': [datetime.now()],
                })
                
                # Add random values for each currency pair
                for pair in currency_pairs:
                    # Generate realistic exchange rates based on the currency pair
                    base = pair[:3]
                    quote = pair[3:6]
                    
                    # Set some base rates for major currencies against USD
                    base_rates = {
                        'USD': 1.0, 'EUR': 1.1, 'GBP': 1.3, 'JPY': 0.009, 
                        'CHF': 1.05, 'CAD': 0.75, 'AUD': 0.70, 'NZD': 0.65,
                        'SGD': 0.74, 'HKD': 0.13, 'CNY': 0.14, 'RUB': 0.011,
                        'INR': 0.012, 'ZAR': 0.055, 'BRL': 0.20, 'MXN': 0.059,
                        'SEK': 0.095, 'NOK': 0.095, 'DKK': 0.14, 'PLN': 0.25,
                        'TRY': 0.030, 'KRW': 0.00075, 'THB': 0.028, 'ILS': 0.27
                    }
                    
                    # Add some randomness
                    jitter = np.random.uniform(0.95, 1.05)
                    
                    if '=X' in pair:
                        if base in base_rates and quote in base_rates:
                            # Calculate cross rate
                            rate = base_rates[base] / base_rates[quote] * jitter
                            dummy_data[pair] = [rate]
                        else:
                            # Fallback to random
                            dummy_data[pair] = [np.random.uniform(0.5, 2.0)]
                
                return dummy_data
        
        # Call the function with the checkbox value as parameter
        forex_data = fetch_forex_data(currency_pairs, use_simulated_data)
        
        # Process the data for display
        latest_data = forex_data.iloc[-1:].melt(id_vars='Date', var_name='Currency Pair', value_name='Exchange Rate')
        latest_data['Base Currency'] = latest_data['Currency Pair'].str[:3]
        latest_data['Quote Currency'] = latest_data['Currency Pair'].str[3:6]
        st.dataframe(latest_data[['Base Currency', 'Quote Currency', 'Exchange Rate']], use_container_width=True)
    
    with tab2:
        st.header("Currency Network Analysis")
        
        # Parameters for analysis
        st.subheader("Analysis Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            target_currency = st.selectbox("Target Currency (Start/End)", options=major_currencies, index=major_currencies.index('USD') if 'USD' in major_currencies else 0)
        with col2:
            min_cycle_length = st.slider("Minimum Arbitrage Cycle Length", min_value=3, max_value=5, value=3)
        with col3:
            transaction_cost = st.slider("Transaction Cost (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        
        # Build the currency graph
        def build_currency_graph(forex_data, transaction_cost):
            G = nx.DiGraph()
            
            # Process the latest data
            latest_rates = forex_data.iloc[-1:].to_dict('records')[0]
            
            # First add all nodes (currencies)
            for currency in major_currencies:
                G.add_node(currency)
            
            # Then add the edges
            for pair in [p for p in latest_rates.keys() if p != 'Date']:
                if pair.endswith('=X'):
                    base = pair[:3]
                    quote = pair[3:6]
                    
                    # Only process pairs where both currencies are in our selected list
                    if base in major_currencies and quote in major_currencies:
                        # Add edges with weights as -log(exchange_rate)
                        # Negative log is used so negative cycles indicate arbitrage
                        rate = latest_rates[pair]
                        if isinstance(rate, (int, float)) and rate > 0:  # Avoid log(0) errors and handle NaN
                            # Apply transaction cost
                            adjusted_rate = rate * (1 - transaction_cost/100)
                            weight = -math.log(adjusted_rate)
                            G.add_edge(base, quote, weight=weight, rate=adjusted_rate)
                            
                            # Add the inverse edge
                            inverse_rate = 1 / rate
                            adjusted_inverse = inverse_rate * (1 - transaction_cost/100)
                            G.add_edge(quote, base, weight=-math.log(adjusted_inverse), rate=adjusted_inverse)
            
            return G
        
        # Detect arbitrage opportunities
        def find_arbitrage(G, min_cycle_length=3, target_currency=None):
            arbitrage_opportunities = []
            
            # If target currency is specified, only use that as source
            # Otherwise check all currencies as potential starting points
            source_nodes = [target_currency] if target_currency else G.nodes()
            
            for source in source_nodes:
                if source not in G:
                    continue
                    
                # Initialize distance dictionary with infinity for all nodes
                distance = {node: float('inf') for node in G.nodes()}
                distance[source] = 0
                predecessor = {node: None for node in G.nodes()}
                
                # Relax edges repeatedly to find the shortest path
                for _ in range(len(G) - 1):
                    for u, v, attrs in G.edges(data=True):
                        if distance[u] + attrs['weight'] < distance[v]:
                            distance[v] = distance[u] + attrs['weight']
                            predecessor[v] = u
                
                # Check for negative weight cycles
                for u, v, attrs in G.edges(data=True):
                    if distance[u] + attrs['weight'] < distance[v]:
                        # Found a negative cycle
                        # Reconstruct the cycle
                        cycle = [v, u]
                        current = u
                        
                        # Trace back until we complete the cycle
                        while predecessor[current] not in cycle and current is not None:
                            current = predecessor[current]
                            if current is None:  # No real cycle
                                break
                            cycle.append(current)
                        
                        if current is not None and len(cycle) >= min_cycle_length:
                            # Only process cycles that include our target currency if specified
                            if target_currency and target_currency not in cycle:
                                continue
                                
                            # Calculate the profit for this cycle
                            profit = 1.0
                            cycle_path = cycle.copy()
                            
                            # Arrange the cycle to start and end at the same node
                            try:
                                if target_currency and target_currency in cycle:
                                    # Make target currency the start/end
                                    while cycle_path[0] != target_currency:
                                        cycle_path.append(cycle_path.pop(0))
                                    if cycle_path[0] != cycle_path[-1]:
                                        cycle_path.append(cycle_path[0])  # Close the loop
                                else:
                                    start_idx = cycle.index(predecessor[current])
                                    cycle_path = cycle[start_idx:] + cycle[:start_idx+1]
                                cycle_path.reverse()  # Correct direction
                            except (ValueError, IndexError):
                                # Skip if we can't properly arrange the cycle
                                continue
                            
                            # Calculate the potential profit
                            cycle_edges = []
                            try:
                                for i in range(len(cycle_path) - 1):
                                    start = cycle_path[i]
                                    end = cycle_path[i + 1]
                                    
                                    # Check if edge exists
                                    if G.has_edge(start, end):
                                        rate = G[start][end]['rate']
                                        profit *= rate
                                        cycle_edges.append((start, end, rate))
                                    else:
                                        # Skip cycles with missing edges
                                        raise KeyError(f"Edge {start}->{end} not found")
                                
                                if profit > 1.0:  # Only include profitable cycles
                                    arbitrage_opportunities.append({
                                        'cycle': cycle_path,
                                        'profit_factor': profit,
                                        'profit_percentage': (profit - 1) * 100,
                                        'edges': cycle_edges
                                    })
                            except KeyError:
                                # Skip this cycle if any edge doesn't exist
                                continue
            
            # Remove duplicate cycles
            unique_opportunities = []
            seen_cycles = set()
            
            for opp in arbitrage_opportunities:
                # Create a hashable representation of the cycle
                cycle_str = '->'.join(opp['cycle'])
                if cycle_str not in seen_cycles:
                    seen_cycles.add(cycle_str)
                    unique_opportunities.append(opp)
            
            return unique_opportunities
        
        # Visualize the currency graph
        def visualize_currency_graph(G, arbitrage_cycles=None, highlight_currency=None):
            # Added highlight_currency parameter with default None
            
            # Create positions for each node
            pos = nx.spring_layout(G, seed=42)
            
            # Create the main graph
            edge_trace = []
            
            # Create a trace for each edge with hover information
            for u, v, data in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Create hover text with exchange rate info
                rate = data.get('rate', 0)
                hover_text = f"{u} → {v}: {rate:.4f}"
                
                edge_trace.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='text',
                        hovertext=hover_text,
                        mode='lines',
                        showlegend=False
                    )
                )
            
            # Create a trace for the nodes
            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
            
            # Create colors for nodes - highlight target currency if specified
            node_colors = []
            node_sizes = []
            for node in G.nodes():
                if highlight_currency and node == highlight_currency:
                    node_colors.append('#FF5733')  # Highlight color
                    node_sizes.append(40)  # Larger size for highlight
                else:
                    node_colors.append('#3366CC')  # Default color
                    node_sizes.append(30)  # Default size
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=list(G.nodes()),
                textposition="top center",
                hoverinfo='text',
                marker=dict(
                    showscale=False,
                    color=node_colors,
                    size=node_sizes,
                    line_width=2
                )
            )
            
            # Highlight arbitrage cycles if any
            arbitrage_traces = []
            if arbitrage_cycles:
                for idx, cycle in enumerate(arbitrage_cycles):
                    cycle_x = []
                    cycle_y = []
                    for node in cycle['cycle']:
                        x, y = pos[node]
                        cycle_x.append(x)
                        cycle_y.append(y)
                    
                    arbitrage_traces.append(
                        go.Scatter(
                            x=cycle_x, 
                            y=cycle_y,
                            mode='lines+markers',
                            line=dict(width=3, color=f'rgba(255, 0, 0, {0.7 - idx * 0.1})'),
                            name=f"Arbitrage #{idx+1}: {' → '.join(cycle['cycle'])}",
                            hoverinfo='text',
                            text=[f"{cycle['profit_percentage']:.2f}% profit potential"]
                        )
                    )
            
            # Create the figure
            fig = go.Figure(data=edge_trace + [node_trace] + arbitrage_traces,
                          layout=go.Layout(
                              title='Currency Exchange Network',
                              titlefont_size=16,
                              showlegend=True,
                              hovermode='closest',
                              margin=dict(b=20, l=5, r=5, t=40),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              legend=dict(x=0, y=1.1, orientation="h")
                          ))
            
            return fig
        
        # Build the graph and find arbitrage opportunities
        currency_graph = build_currency_graph(forex_data, transaction_cost)
        arbitrage_opportunities = find_arbitrage(currency_graph, min_cycle_length, target_currency)
        
        # Visualize the graph
        st.subheader("Currency Exchange Network Visualization")
        
        if len(arbitrage_opportunities) > 0:
            st.write(f"Found {len(arbitrage_opportunities)} potential arbitrage opportunities!")
            # Pass target_currency as highlight_currency
            graph_fig = visualize_currency_graph(currency_graph, arbitrage_opportunities[:3], target_currency)  # Show top 3
        else:
            st.write("No arbitrage opportunities detected with current parameters.")
            # Pass target_currency as highlight_currency
            graph_fig = visualize_currency_graph(currency_graph, highlight_currency=target_currency)
            
        st.plotly_chart(graph_fig, use_container_width=True)
        
        # Display exchange rate matrix
        st.subheader("Currency Exchange Rate Matrix")
        
        # Create matrix of exchange rates
        all_currencies = list(currency_graph.nodes())
        rate_matrix = pd.DataFrame(index=all_currencies, columns=all_currencies)
        
        for c1 in all_currencies:
            for c2 in all_currencies:
                if c1 != c2 and currency_graph.has_edge(c1, c2):
                    rate_matrix.loc[c1, c2] = currency_graph[c1][c2]['rate']
                else:
                    rate_matrix.loc[c1, c2] = np.nan
        
        # Display as a heatmap
        fig = px.imshow(rate_matrix, 
                     text_auto='.4f',
                     color_continuous_scale='Blues',
                     title="Exchange Rate Matrix")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Results & Insights")
        
        # Key metrics display
        st.subheader("Arbitrage Opportunities Summary")
        
        if len(arbitrage_opportunities) > 0:
            # Sort by profit potential
            sorted_opportunities = sorted(arbitrage_opportunities, key=lambda x: x['profit_percentage'], reverse=True)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Total Opportunities", 
                    value=len(arbitrage_opportunities)
                )
            with col2:
                max_profit = max([opp['profit_percentage'] for opp in arbitrage_opportunities])
                st.metric(
                    label="Max Potential Profit", 
                    value=f"{max_profit:.2f}%"
                )
            with col3:
                avg_profit = sum([opp['profit_percentage'] for opp in arbitrage_opportunities]) / len(arbitrage_opportunities)
                st.metric(
                    label="Avg Potential Profit", 
                    value=f"{avg_profit:.2f}%"
                )
            
            # Display the arbitrage opportunities
            st.subheader("Top Arbitrage Paths")
            
            for i, opp in enumerate(sorted_opportunities[:5]):  # Show top 5
                with st.expander(f"Arbitrage Path #{i+1}: {' → '.join(opp['cycle'])} ({opp['profit_percentage']:.2f}% profit)"):
                    # Create a DataFrame to display the steps
                    steps_data = []
                    for j, (from_curr, to_curr, rate) in enumerate(opp['edges']):
                        steps_data.append({
                            'Step': j+1,
                            'From': from_curr,
                            'To': to_curr,
                            'Exchange Rate': rate,
                            'Action': f"Convert 1 {from_curr} to {rate:.4f} {to_curr}"
                        })
                    
                    steps_df = pd.DataFrame(steps_data)
                    st.dataframe(steps_df, use_container_width=True)
                    
                    # Show the cumulative effect
                    st.write(f"Starting with 1 {opp['cycle'][0]} and following this path returns {opp['profit_factor']:.4f} {opp['cycle'][0]}")
                    st.write(f"Net profit: {opp['profit_percentage']:.2f}%")
                    
                    # Add visualization of the specific arbitrage path
                    currencies_involved = list(set([edge[0] for edge in opp['edges']] + [edge[1] for edge in opp['edges']]))
                    subgraph = currency_graph.subgraph(currencies_involved)
                    path_fig = visualize_currency_graph(subgraph, [opp], target_currency)
                    st.plotly_chart(path_fig, use_container_width=True)
        else:
            st.info("No arbitrage opportunities were detected with the current parameters.")
            st.write("""
            This is actually expected in efficient markets, especially when transaction costs are considered.
            You can try:
            - Reducing the transaction cost percentage
            - Including more currencies in the analysis
            - Checking during periods of high market volatility
            """)
        
        # Key insights
        st.subheader("Key Insights")
        
        if len(arbitrage_opportunities) > 0:
            st.write("""
            Based on the current analysis:
            
            1. **Potential Inefficiencies**: The detected arbitrage opportunities suggest temporary market inefficiencies that could potentially be exploited.
            
            2. **Transaction Cost Impact**: The profitability of arbitrage opportunities is highly sensitive to transaction costs. Even small increases in costs can eliminate opportunities.
            
            3. **Cycle Complexity**: Longer arbitrage cycles (involving more currencies) can sometimes yield higher profit potential but come with increased execution risk.
            """)
        else:
            st.write("""
            Based on the current analysis:
            
            1. **Market Efficiency**: The absence of arbitrage opportunities suggests that the forex market is currently efficient for the analyzed currency pairs.
            
            2. **Transaction Costs**: When realistic transaction costs are factored in, most theoretical arbitrage opportunities disappear.
            
            3. **Risk vs. Reward**: While no risk-free profits were detected, the methodology can be extended to analyze statistical arbitrage opportunities that involve some risk.
            """)
        
        # Recommendations
        st.subheader("Recommendations")
        st.write("""
        For traders and financial institutions interested in arbitrage:
        
        1. **Real-time Monitoring**: Implement a real-time version of this system to continuously monitor for emerging arbitrage opportunities.
        
        2. **Execution Speed**: Develop high-frequency trading capabilities to capitalize on short-lived arbitrage opportunities before they disappear.
        
        3. **Additional Data Sources**: Integrate additional pricing sources to identify discrepancies between different forex markets or brokers.
        
        4. **Extended Analysis**: Include more exotic currency pairs and cross-rates to uncover less obvious arbitrage paths.
        """)
        
        # Added new visualization - Arbitrage Profit Sensitivity
        if len(arbitrage_opportunities) > 0:
            st.subheader("Arbitrage Profit Sensitivity Analysis")
            
            # Create sensitivity data
            tc_values = np.linspace(0, 0.5, 10)  # Transaction costs from 0% to 0.5%
            
            # Get the top 3 opportunities
            top_opps = sorted_opportunities[:3]
            
            # For each opportunity, calculate profit at different transaction costs
            sensitivity_data = []
            
            for tc in tc_values:
                # Build graph with this transaction cost
                test_graph = build_currency_graph(forex_data, tc)
                
                for i, opp in enumerate(top_opps):
                    # Calculate profit for this cycle with new transaction cost
                    profit = 1.0
                    cycle_valid = True
                    
                    for j in range(len(opp['cycle']) - 1):
                        start = opp['cycle'][j]
                        end = opp['cycle'][j + 1]
                        
                        if test_graph.has_edge(start, end):
                            profit *= test_graph[start][end]['rate']
                        else:
                            cycle_valid = False
                            break
                    
                    if cycle_valid:
                        profit_pct = (profit - 1) * 100
                        sensitivity_data.append({
                            'Transaction Cost (%)': tc * 100,
                            'Profit (%)': profit_pct if profit_pct > 0 else 0,
                            'Arbitrage Path': f"Path #{i+1}: {' → '.join(opp['cycle'][:3])}..."
                        })
                    else:
                        sensitivity_data.append({
                            'Transaction Cost (%)': tc * 100,
                            'Profit (%)': 0,
                            'Arbitrage Path': f"Path #{i+1}: {' → '.join(opp['cycle'][:3])}..."
                        })
            
            # Create the sensitivity plot
            if sensitivity_data:
                sense_df = pd.DataFrame(sensitivity_data)
                fig = px.line(sense_df, 
                           x='Transaction Cost (%)', 
                           y='Profit (%)', 
                           color='Arbitrage Path',
                           title="Arbitrage Profit Sensitivity to Transaction Costs",
                           markers=True)
                
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=0.5*100, y1=0,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("This chart shows how sensitive each arbitrage opportunity is to changes in transaction costs.")
    
    # Technical methodology
    with st.expander("Technical Methodology"):
        st.write("""
        ### Graph-Based Currency Arbitrage Detection
        
        #### Mathematical Foundation
        
        Currency arbitrage detection can be formulated as a graph problem:
        
        1. **Graph Construction**:
           - Each currency is represented as a node
           - Exchange rates form weighted directed edges
           - Edge weights are transformed to -log(exchange_rate)
        
        2. **Negative Cycle Detection**:
           - A negative cycle in this graph indicates an arbitrage opportunity
           - The Bellman-Ford algorithm is used to detect these cycles
           - The total weight of a cycle corresponds to the potential profit
        
        #### Algorithm Details
        
        The logarithmic transformation is key to this approach:
        - If exchange rates are multiplicative (e.g., 1 USD → 0.85 EUR → 1.02 USD = 0.867 USD)
        - Taking -log turns this into addition: -log(0.85) + -log(1.2) = -(-0.16 + -0.18) = 0.34
        - A negative sum indicates profit (e^0.34 = 1.41, or 41% profit)
        
        #### Implementation Challenges
        
        - **Data Quality**: Exchange rate data may have gaps or errors
        - **Transaction Costs**: Real-world costs significantly impact profitability
        - **Execution Lag**: Markets move quickly, reducing opportunity windows
        - **Scale**: Analyzing all possible currency combinations is computationally intensive
        
        #### Extensions
        
        This methodology can be extended to other financial instruments:
        - Cross-exchange cryptocurrency arbitrage
        - Interest rate arbitrage (covered interest parity)
        - Triangular arbitrage in options markets
        """)
        
        # Sample code snippet
        st.code("""
# Core algorithm for arbitrage detection
def bellman_ford_arbitrage(graph, source):
    # Initialize distance and predecessor dictionaries
    distance = {node: float('inf') for node in graph.nodes()}
    distance[source] = 0
    predecessor = {node: None for node in graph.nodes()}
    
    # Relax edges |V|-1 times
    for _ in range(len(graph)-1):
        for u, v, data in graph.edges(data=True):
            weight = data['weight']
            if distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                predecessor[v] = u
    
    # Check for negative cycles
    for u, v, data in graph.edges(data=True):
        if distance[u] + data['weight'] < distance[v]:
            # Negative cycle exists
            return reconstruct_cycle(u, v, predecessor)
    
    return None
        """)
    
    # Add a performance optimization section
    with st.expander("Performance Optimizations"):
        st.write("""
        ### Performance Considerations
        
        This application has been optimized in several ways:
        
        1. **Data Caching**: Exchange rate data is cached for 5 minutes to reduce API calls
        
        2. **Algorithmic Optimizations**:
           - Early termination in cycle detection when no valid arbitrage is possible
           - Filtering currencies before graph construction
           - Using hash-based data structures for cycle deduplication
        
        3. **Selective Processing**:
           - When a target currency is specified, only cycles involving that currency are computed
           - Progressive disclosure pattern with expandable sections for detailed information
        
        4. **User Experience**:
           - Asynchronous data loading with spinner feedback
           - Fast rendering of complex graph structures
           - Interactive visualizations that focus on meaningful information
        """)

# This function will be called from the main app
if __name__ == "__main__":
    show()