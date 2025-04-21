# project1.py
# (Corrected Version)
import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
import os
from datetime import datetime, timedelta
from itertools import permutations, islice, cycle
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import yfinance as yf
import math
import streamlit.components.v1 as components
import io
import traceback
import re
import numpy as np
import plotly.graph_objects as go

# --- Configuration ---
DATA_DIR = "forex_data"
CURRENCIES = [
    'USD', 'EUR', 'JPY', 'GBP', 'CNY', 'AUD', 'CAD', 'CHF', 'HKD', 'SGD',
    'KRW', 'NOK', 'NZD', 'SEK', 'MXN', 'INR', 'RUB', 'ZAR', 'BRL', 'TRY'
]
MAX_ARBITRAGE_CYCLE_LENGTH = 7 # Limit cycle length for performance in simple_cycles
MAX_ARBITRAGE_PATHS_DISPLAY = 50 # Limit number of paths shown in UI

EXCLUDED_PAIRS = set([
    "BRLCAD=X","BRLTRY=X","CNYMXN=X","CNYNOK=X","CNYSEK=X","CNYTRY=X",
    "HKDRUB=X","HKDTRY=X","INRMXN=X","INRNOK=X","INRSEK=X","INRTRY=X",
    "KRWMXN=X","KRWNOK=X","KRWRUB=X","KRWTRY=X","MXNCNY=X","MXNINR=X",
    "MXNKRW=X","MXNNOK=X","MXNNZD=X","MXNRUB=X","MXNSEK=X","MXNTRY=X",
    "NOKCNY=X","NOKINR=X","NOKKRW=X","NOKMXN=X","NOKRUB=X","NOKTRY=X",
    "NZDRUB=X","RUBCAD=X","RUBHKD=X","RUBINR=X","RUBKRW=X","RUBMXN=X",
    "RUBNOK=X","RUBNZD=X","RUBSEK=X","RUBTRY=X","SEKMXN=X","SEKTRY=X",
    "SGDCAD=X","TRYAUD=X","TRYBRL=X","TRYCAD=X","TRYCNY=X","TRYHKD=X",
    "TRYINR=X","TRYKRW=X","TRYMXN=X","TRYNOK=X","TRYNZD=X","TRYRUB=X",
    "TRYSEK=X","ZARTRY=X"
])


# --- Data Download Logic ---
def get_target_date_for_latest_data():
    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    today_weekday = today.weekday()
    if 0 <= today_weekday <= 4: return today
    elif today_weekday == 5: return today - timedelta(days=1)
    else: return today - timedelta(days=2)

def fetch_and_save(symbol, target_date, data_dir):
    start_str = target_date.strftime('%Y-%m-%d'); interval = '1m'
    start_dt = target_date; end_dt = target_date + timedelta(days=1)
    try:
        ticker_obj = yf.Ticker(symbol); data = ticker_obj.history(start=start_dt, end=end_dt, interval=interval)
        if data.empty: print(f"❌ No data {symbol} for {start_str}"); return symbol
        if not data.index.empty:
            data.index = data.index.tz_convert(None); data = data[data.index.date == target_date.date()]
        if data.empty: print(f"❌ No data {symbol} specifically for {start_str} after filtering."); return symbol
        filename = os.path.join(data_dir, f"{symbol}_{start_str}.csv"); data.to_csv(filename)
        return None
    except Exception as e: print(f"⚠️ Error fetch {symbol}: {type(e).__name__}-{e}"); return symbol

def run_data_download(data_dir, currencies):
    st.info("Starting data download process..."); target_date = get_target_date_for_latest_data(); target_date_str = target_date.strftime('%Y-%m-%d')
    st.write(f"Targeting data for: **{target_date_str}** ({target_date.strftime('%A')})"); os.makedirs(data_dir, exist_ok=True)
    currency_pairs = list(permutations(currencies, 2)); all_symbols = [f"{b}{q}=X" for b, q in currency_pairs]
    symbols_to_download = [s for s in all_symbols if s not in EXCLUDED_PAIRS]; excluded_count = len(all_symbols) - len(symbols_to_download)
    st.write(f"Total pairs: {len(all_symbols)}, Excluded: {excluded_count}, Downloading: **{len(symbols_to_download)}**")
    if not symbols_to_download: st.warning("No pairs left."); return False
    invalid_pairs = []; max_threads = min(20, os.cpu_count()*2 if os.cpu_count() else 4, len(symbols_to_download)); print(f"Using {max_threads} threads...")
    start_dl = time.time(); progress_bar = st.progress(0, text="Initializing download..."); status_text = st.empty(); total_proc = len(symbols_to_download); completed = 0
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        f_to_s = {executor.submit(fetch_and_save, s, target_date, data_dir): s for s in symbols_to_download}
        for future in as_completed(f_to_s):
            symbol = f_to_s[future]; completed += 1; progress = completed / total_proc
            try: result = future.result();
            except Exception as exc: st.error(f"‼️ Err result {symbol}: {exc}"); invalid_pairs.append(symbol)
            else:
                if result: invalid_pairs.append(result)
            try:
                 progress_text = f"Downloading... Processed {completed}/{total_proc} pairs."
                 progress_bar.progress(progress, text=progress_text); status_text.text(progress_text)
            except Exception as e: print(f"Error updating progress bar: {e}") # Ignore UI errors if element removed

    end_dl = time.time(); total_dl = end_dl - start_dl
    try: progress_bar.empty(); status_text.empty(); # Clean up progress elements
    except: pass
    success_dl = total_proc - len(invalid_pairs); st.success(f"✅ Download completed in {total_dl:.2f} seconds."); st.write(f"Successfully downloaded: {success_dl}, Failed/No Data: {len(invalid_pairs)}")
    if invalid_pairs:
        with st.expander(f"⚠️ Show {len(invalid_pairs)} Failed/Empty Pairs"): st.write(sorted(invalid_pairs))
    st.cache_data.clear(); st.cache_resource.clear()
    keys_to_clear = ['graph_data', 'arbitrage_result']
    for key in keys_to_clear:
        if key in st.session_state: del st.session_state[key]
    return True

# --- Data Loading Helpers ---

# CORRECTED _find_datetime_column function
def _find_datetime_column(df):
    """ Finds the most likely datetime column using standard indentation. """
    # Prioritize columns with 'datetime' or both 'date' and 'time'
    for col in df.columns:
        low_col = col.lower()
        if 'datetime' in low_col:
            # print(f"Found datetime column: {col}") # Optional debug print
            return col
        if 'date' in low_col and 'time' in low_col:
            # print(f"Found date/time column: {col}") # Optional debug print
            return col

    # Fallback checks for 'date' or 'time' separately
    date_col = None
    time_col = None
    # Find first 'date' column
    for col in df.columns:
        low_col = col.lower()
        if 'date' in low_col:
            # print(f"Found date column (fallback): {col}") # Optional debug print
            date_col = col
            break
    # Find first 'time' column
    for col in df.columns:
         low_col = col.lower()
         if 'time' in low_col:
             # print(f"Found time column (fallback): {col}") # Optional debug print
             time_col = col
             break

    # Return date or time if found, prioritizing date slightly
    if date_col: return date_col
    if time_col: return time_col

    # Check index
    if isinstance(df.index, pd.DatetimeIndex):
        # print(f"Using existing DatetimeIndex: {df.index.name}") # Optional debug print
        return df.index.name # Use index if it's datetime

    # Check for common unnamed index column from read_csv
    if 'Unnamed: 0' in df.columns:
         try:
             # Test if the column can be parsed as datetime without modifying df yet
             pd.to_datetime(df['Unnamed: 0'], errors='raise')
             # print("Found potential datetime in 'Unnamed: 0'") # Optional debug print
             return 'Unnamed: 0'
         except (ValueError, TypeError, OverflowError, KeyError):
             pass # Ignore if it's not datetime-like

    print("No suitable datetime column found.")
    return None # No suitable column found


def _parse_datetime_column(df, dt_col_name):
    """ Attempts to parse and set the datetime index. """
    try:
        # If dt_col_name refers to the existing index
        if dt_col_name == df.index.name:
            if not isinstance(df.index, pd.DatetimeIndex):
                 df.index = pd.to_datetime(df.index, errors='coerce')
        # If dt_col_name refers to a column
        elif dt_col_name in df.columns:
             df[dt_col_name] = pd.to_datetime(df[dt_col_name], errors='coerce')
             df = df.dropna(subset=[dt_col_name]) # Drop rows where parsing failed
             df = df.set_index(dt_col_name)
        else:
            print(f"Column '{dt_col_name}' not found for parsing.")
            return None # Column specified doesn't exist

        # Check if index is now DatetimeIndex and process timezone
        if isinstance(df.index, pd.DatetimeIndex):
            if getattr(df.index, 'tz', None) is not None:
                df.index = df.index.tz_convert(None) # Convert to UTC naive
            return df.sort_index() # Ensure sorted
        else:
            print(f"Failed to set DatetimeIndex using '{dt_col_name}'")
            return None
    except Exception as e:
        print(f"Could not parse datetime column '{dt_col_name}': {e}")
        return None


@st.cache_data(ttl=3600)
def load_pair_data(symbol, data_dir):
    """ Loads, cleans, and returns DataFrame for a specific symbol. """
    target_date = get_target_date_for_latest_data(); target_date_str = target_date.strftime('%Y-%m-%d')
    filename = f"{symbol}_{target_date_str}.csv"; fpath = os.path.join(data_dir, filename)
    if not os.path.exists(fpath): print(f"Data file not found: {fpath}"); return None
    try:
        df = pd.read_csv(fpath)
        dt_col = _find_datetime_column(df) # Use corrected function

        if dt_col:
            df = _parse_datetime_column(df, dt_col) # Use corrected function
        else:
            print(f"No datetime column identified for {filename}"); return None # dt_col search failed

        if df is None: return None # Parsing failed

        # --- Renaming and Column Check (Standard Indentation) ---
        rename_map = {}
        for col in df.columns:
            col_low = col.lower()
            if col_low == 'open' and 'Open' not in df.columns: rename_map[col] = 'Open'
            elif col_low == 'high' and 'High' not in df.columns: rename_map[col] = 'High'
            elif col_low == 'low' and 'Low' not in df.columns: rename_map[col] = 'Low'
            elif col_low == 'close' and 'Close' not in df.columns: rename_map[col] = 'Close'
            elif col_low == 'adj close' and 'Adj Close' not in df.columns: rename_map[col] = 'Adj Close'
            elif col_low == 'volume' and 'Volume' not in df.columns: rename_map[col] = 'Volume'
        if rename_map: df = df.rename(columns=rename_map)

        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        present_cols = [col for col in ohlc_cols if col in df.columns]
        if 'Close' not in present_cols:
            if 'Adj Close' in df.columns and pd.api.types.is_numeric_dtype(df['Adj Close']):
                 df=df.rename(columns={'Adj Close': 'Close'}); present_cols.append('Close')
                 # print(f"Used Adj Close as Close for {symbol}") # Optional debug
            else: # Try find any numeric as Close
                 for col in df.columns:
                     if col not in ['Open','High','Low','Volume'] and pd.api.types.is_numeric_dtype(df[col]):
                          df=df.rename(columns={col: 'Close'}); present_cols.append('Close')
                          # print(f"Used column '{col}' as Close for {symbol}"); # Optional debug
                          break
        if 'Close' not in df.columns: print(f"Missing 'Close' price {filename}"); return None

        for col in present_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Close']) # Drop only if Close is NaN
        if df.empty: print(f"DataFrame empty {filename}"); return None

        return df # Already sorted in _parse_datetime_column
    except Exception as e:
        st.error(f"Error loading {filename}: {e}", icon="❌")
        st.text(traceback.format_exc());
        return None


# --- Network Graph Generation Logic ---
@st.cache_data(ttl=3600)
def load_and_build_graph(data_dir, excluded_pairs):
    """ Loads data, returns graph structure (nodes, edges list), last update time. """
    start_load = time.time(); G_nodes_set = set(); G_edges_list = []; latest_ts_dict = {}; valid_found = False
    if not os.path.exists(data_dir): st.warning(f"Data directory '{data_dir}' not found."); return [], [], "Never", {}
    print(f"--- Loading data from {data_dir} for graph ---"); file_count=0; proc_count=0
    target_date = get_target_date_for_latest_data(); target_date_str = target_date.strftime('%Y-%m-%d')

    all_files = [f for f in os.listdir(data_dir) if f.endswith(f"{target_date_str}.csv")]
    print(f"Found {len(all_files)} files for date {target_date_str}")

    for filename in all_files:
        file_count += 1 # Count only files matching date pattern
        symbol = filename.split('_')[0]
        if symbol in excluded_pairs or len(symbol) < 6 or not symbol.endswith("=X"): continue
        base, quote = symbol[:3], symbol[3:6]; proc_count +=1
        # print(f"Processing: {symbol}") # Verbose logging
        df = load_pair_data(symbol, data_dir) # Uses cache
        if df is not None and not df.empty and 'Close' in df.columns:
            valid_found=True;
            try:
                last_row=df.iloc[-1]; last_ts=last_row.name; rate=last_row['Close']
                if pd.notna(rate) and rate>0:
                    G_nodes_set.add(base); G_nodes_set.add(quote)
                    G_edges_list.append((base, quote, {'weight': float(rate), 'title': f"{base}→{quote}:{rate:.6f}", 'timestamp': last_ts}))
                    if isinstance(last_ts, (pd.Timestamp, datetime)): latest_ts_dict[f"{base}-{quote}"]=last_ts
            except IndexError:
                print(f"IndexError accessing last row for {symbol}. DF empty after load?")
            except Exception as e:
                 print(f"Error processing last row for {symbol}: {e}")

    mrt_str="Never"
    if latest_ts_dict:
        try:
            mrt = max(latest_ts_dict.values())
            # Ensure mrt is datetime before formatting
            if isinstance(mrt, (pd.Timestamp, datetime)):
                 mrt_str=(mrt.tz_convert(None) if hasattr(mrt,'tz') and mrt.tz else mrt).strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                 mrt_str = str(mrt) # Fallback if not a datetime object
        except Exception as e: print(f"Error formatting max timestamp: {e}"); mrt_str = "Error"

    if not valid_found: st.warning("No valid, non-excluded data found for the target date.", icon="⚠️")
    load_dur = time.time()-start_load; print(f"--- Graph build done. Processed {proc_count}/{file_count} relevant files in {load_dur:.2f}s ---")
    return list(G_nodes_set), G_edges_list, mrt_str, latest_ts_dict

# --- Pyvis Generation ---
@st.cache_resource(ttl=3600)
def generate_pyvis_html_string(_G_nodes, _G_edges, selected_currency=None, highlight_cycle_nodes=None, highlight_cycle_edges=None):
    """ Generates Pyvis HTML string. Uses cache_resource. """
    if not _G_nodes: print("Graph nodes list empty for Pyvis."); return None
    # Removed spinner from here as it's better placed where called in UI
    net = Network(notebook=False, directed=True, cdn_resources='remote', height="700px", width="100%", bgcolor="#FFFFFF", font_color="black")
    net.set_options("""
    var options = {
        "nodes": { "font": {"size": 18, "face": "Arial"}, "shape": "dot", "size": 28, "borderWidth": 1.5 },
        "edges": { "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}}, "color": {"inherit": false}, "smooth": {"type": "continuous"}, "font": {"size": 11, "face": "Arial", "color": "#343434", "align": "middle"} },
        "physics": { "forceAtlas2Based": { "gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 230, "springConstant": 0.08, "damping": 0.4, "avoidOverlap": 0 }, "minVelocity": 0.75, "solver": "forceAtlas2Based" },
        "interaction": { "navigationButtons": true, "keyboard": true, "hover": true, "tooltipDelay": 200 }
    }
    """)

    default_node_color = "#ADD8E6"; base_highlight_color = "#FF6347"; cycle_node_color = "#98FB98"
    default_edge_color = "#888888"; cycle_edge_color = "#006400"

    highlight_cycle_nodes_set = set(highlight_cycle_nodes) if highlight_cycle_nodes else set()
    highlight_cycle_edges_set = set((edge[0], edge[1]) for edge in highlight_cycle_edges) if highlight_cycle_edges else set()

    for node in _G_nodes:
        color = default_node_color; border_width = 1.5; node_size = 28
        if node == selected_currency: color = base_highlight_color; border_width = 2.5; node_size = 35
        elif node in highlight_cycle_nodes_set: color = cycle_node_color; border_width = 2.5; node_size = 35
        net.add_node(node, label=node, title=node, color=color, borderWidth=border_width, size=node_size)

    for source, target, data in _G_edges:
        rate = data.get('weight', 0); title_str = f"{source} → {target}: {rate:.6f}"; label_str = f"{rate:.4f}"
        edge_color = default_edge_color; edge_width = 1.5
        if (source, target) in highlight_cycle_edges_set: edge_color = cycle_edge_color; edge_width = 3.5
        net.add_edge(source, target, title=title_str, label=label_str, color=edge_color, width=edge_width)

    try:
        temp_html_path = os.path.join(DATA_DIR, f"temp_network_{int(time.time())}.html")
        net.save_graph(temp_html_path)
        with open(temp_html_path, 'r', encoding='utf-8') as f: html_content = f.read()
        os.remove(temp_html_path)
        return html_content
    except Exception as e:
        # Use st.error for visibility in app, print for logs
        st.error(f"Failed generate Pyvis HTML: {e}", icon="❌")
        print(f"Failed generate Pyvis HTML: {e}"); traceback.print_exc();
        return None


# --- Arbitrage Calculation Logic (Multiple Cycles) ---
@st.cache_data(ttl=600)
def find_multiple_arbitrage_cycles(_G_nodes, _G_edges, base_currency, transaction_cost_percent, max_len=MAX_ARBITRAGE_CYCLE_LENGTH):
    """ Finds profitable arbitrage cycles using simple_cycles, returns sorted list. """
    start_time = time.time()
    print(f"\n--- Finding cycles (max_len={max_len}) for {base_currency} w/ cost {transaction_cost_percent}% ---")

    G = nx.DiGraph(); G.add_nodes_from(_G_nodes); G.add_edges_from(_G_edges)
    if not G or base_currency not in G: print("Graph empty or base not found."); return [], 0.0, []

    cost_mult = 1.0 - (transaction_cost_percent / 100.0)
    if cost_mult <= 0: print("Cost multiplier <= 0."); return [], time.time() - start_time, []
    G_log = nx.DiGraph(); bf_weight_attr = 'bf_weight'; skipped_log = 0
    for u, v, data in G.edges(data=True):
        rate = data.get('weight', 0)
        log_val = float('-inf')
        if rate > 0 and (rate * cost_mult) > 0:
            try: log_val = -math.log(rate * cost_mult)
            except ValueError: skipped_log+=1; continue # Skip if log fails
            G_log.add_edge(u, v, **{bf_weight_attr: log_val})
        else: skipped_log += 1
    print(f"Built log-graph ({len(G_log.nodes())} nodes, {len(G_log.edges())} edges, {skipped_log} skipped) in {time.time()-start_time:.3f}s")
    if not G_log.has_node(base_currency): print(f"Base {base_currency} not in log graph."); return [], time.time()-start_time, []

    opportunities = []; found_cycle_nodes_list = []
    processed_cycles = 0; negative_weight_cycles = 0; verified_cycles = 0

    try:
        # This can still be very slow / memory intensive
        all_simple_cycles_gen = nx.simple_cycles(G_log)

        for cycle in all_simple_cycles_gen:
            processed_cycles += 1
            if len(cycle) > max_len: continue
            if base_currency not in cycle: continue

            try: start_index = cycle.index(base_currency); ordered_cycle = cycle[start_index:] + cycle[:start_index] + [base_currency]
            except ValueError: continue # Should not happen

            cycle_log_weight = 0.0; valid_log_cycle = True
            for i in range(len(ordered_cycle) - 1):
                u, v = ordered_cycle[i], ordered_cycle[i+1]
                try: cycle_log_weight += G_log[u][v][bf_weight_attr]
                except KeyError: valid_log_cycle = False; break
            if not valid_log_cycle: continue

            if cycle_log_weight < -1e-9: # Tolerance
                negative_weight_cycles += 1
                product = 1.0; path_parts = []; valid_orig_cycle = True
                for i in range(len(ordered_cycle) - 1):
                    u, v = ordered_cycle[i], ordered_cycle[i+1]
                    try:
                        orig_rate = G[u][v]['weight']; product *= orig_rate * cost_mult
                        path_parts.append(f"{u}→{v} ({orig_rate:.4f})")
                    except KeyError: valid_orig_cycle = False; break
                if not valid_orig_cycle: continue

                if product > 1.0:
                    verified_cycles += 1
                    profit_pc = (product - 1.0) * 100
                    path_str = " → ".join(path_parts)
                    opportunities.append({
                        "Profit (%)": profit_pc, "Path": path_str,
                        "Nodes": len(ordered_cycle)-1, "End Value": product,
                        "_nodes_list": ordered_cycle # Temp store for sorting
                    })
                    # No need to store in found_cycle_nodes_list yet

        print(f"Processed {processed_cycles} simple cycles.")
        print(f"Found {negative_weight_cycles} cycles with negative log weight sum.")
        print(f"Verified {verified_cycles} cycles with product > 1.")

    except Exception as e:
        st.error(f"Error during cycle finding: {e}", icon="❌")
        print(f"Error during cycle finding: {e}"); traceback.print_exc()
        return [], time.time()-start_time, []

    # Sort opportunities by profit descending
    opportunities.sort(key=lambda x: x['Profit (%)'], reverse=True)

    # Extract sorted node lists and clean up opportunities dict
    final_cycle_nodes_list = []
    for opp in opportunities:
        if '_nodes_list' in opp:
            final_cycle_nodes_list.append(opp['_nodes_list'])
            del opp['_nodes_list'] # Remove temporary key

    total_dur = time.time() - start_time
    print(f"Found {len(opportunities)} verified arbitrage cycles in {total_dur:.3f}s")
    return opportunities, total_dur, final_cycle_nodes_list


# --- Volatility Calculation ---
def calculate_volatility(df):
    if df is None or df.empty or 'Close' not in df.columns: return None, None
    try:
        df_copy = df.copy() # Work on a copy
        df_copy['LogReturn'] = np.log(df_copy['Close'] / df_copy['Close'].shift(1))
        df_copy = df_copy.dropna(subset=['LogReturn'])
        if df_copy.empty: return None, None
        std_dev_1min_log = df_copy['LogReturn'].std()
        minutes_in_day = 24 * 60
        num_minutes = len(df_copy)
        if num_minutes < 2: return std_dev_1min_log, None
        annualized_vol = std_dev_1min_log * np.sqrt(252 * minutes_in_day) # ~252 trading days
        return std_dev_1min_log, annualized_vol
    except Exception as e: print(f"Error calculating volatility: {e}"); return None, None

# --- Streamlit Page Display Logic ---
def show():
    os.makedirs(DATA_DIR, exist_ok=True)
    st.title("📈 Currency Arbitrage & Analysis")

    if 'graph_data' not in st.session_state:
        st.session_state.graph_data = {'nodes': [], 'edges': [], 'last_updated': "Never"}
    if 'arbitrage_result' not in st.session_state:
        st.session_state.arbitrage_result = {'opportunities': [], 'duration': 0.0, 'cycle_nodes_list': [], 'base_currency': None, 'selected_cycle_index': 0}

    tab_list = ["Project Overview", "Network & Arbitrage", "Pair Analysis"]
    tab1, tab2, tab3 = st.tabs(tab_list)

    # --- Tab 1: Project Overview ---
    with tab1:
        st.header("✨ Project Overview: Forex Arbitrage & Analysis")
        st.markdown("""
        This application analyzes foreign exchange (Forex) market data to:
        1.  Identify potential **cyclical arbitrage opportunities**.
        2.  Visualize individual currency pair **price action** (candlestick).
        3.  Calculate basic **volatility metrics**.

        **Goal:** Provide tools for exploring Forex market relationships and finding potential inefficiencies based on historical minute-data.

        **Methodology:**

        1.  **Data Acquisition:** Fetches recent 1-minute OHLC data (`yfinance`), excludes certain pairs, saves locally. *(Data has inherent delays and may not reflect live market conditions)*.
        2.  **Network Construction (Tab: Network & Arbitrage):** Builds a directed graph (`networkx`) where Nodes=Currencies, Edges=Exchange Rates (latest 'Close'). Visualized using `pyvis`.
        3.  **Arbitrage Detection (Tab: Network & Arbitrage):**
            * Uses `networkx.simple_cycles` on a **log-transformed graph** (`weight = -log(rate * cost_multiplier)`) to find potential arbitrage paths.
            * Filters cycles to include the selected **base currency** and adhere to a **maximum length** (e.g., 7 hops) for performance.
            * Calculates the **sum of log-weights**: a negative sum indicates a potential opportunity.
            * **Verifies** potential cycles by calculating the **product of original exchange rates** (including transaction costs) along the path in the *original* graph. A product > 1 confirms the arbitrage.
            * Displays a **list of confirmed opportunities**, sorted by profitability.
            * Allows selecting a found cycle to **highlight** on the network graph.
            * *Note: Finding all simple cycles can be computationally intensive.*
        4.  **Pair Visualization (Tab: Pair Analysis):** Allows selection of Base/Quote, loads 1-min OHLC, displays interactive **Candlestick chart** (`plotly`).
        5.  **Volatility Calculation (Tab: Pair Analysis):** Calculates **Standard Deviation of 1-minute Log Returns** and provides a rough **Annualized Volatility** estimate.

        **Libraries Used:** `streamlit`, `yfinance`, `pandas`, `networkx`, `pyvis`, `numpy`, `plotly`.
        """)
        st.markdown("---")
        st.info("ℹ️ **Disclaimer:** This is an analytical tool using historical data. Market conditions change rapidly, and transaction costs/slippage can vary. This does not constitute financial advice.")


    # --- Load or Get Graph Data ---
    if not st.session_state.graph_data.get('nodes'):
        data_exists = os.path.exists(DATA_DIR) and any(f.endswith('.csv') for f in os.listdir(DATA_DIR))
        if data_exists:
            with st.spinner("⏳ Loading graph data from disk..."):
                nodes_list, edges_list, last_updated, _ = load_and_build_graph(DATA_DIR, EXCLUDED_PAIRS)
                st.session_state.graph_data['nodes'] = nodes_list
                st.session_state.graph_data['edges'] = edges_list
                st.session_state.graph_data['last_updated'] = last_updated
        else:
             st.session_state.graph_data = {'nodes': [], 'edges': [], 'last_updated': "Never"}


    G_nodes = st.session_state.graph_data.get('nodes', [])
    G_edges = st.session_state.graph_data.get('edges', [])
    last_updated = st.session_state.graph_data.get('last_updated', "Never")
    nodes_available = sorted(G_nodes) if G_nodes else []


    # --- Tab 2: Network Visualization & Arbitrage Finder ---
    with tab2:
        st.header("🌐 Network & Arbitrage Finder")
        st.markdown("Explore currency relationships and search for potential arbitrage cycles.")

        with st.container(): # Group data refresh
            st.subheader("Data Management")
            col_ref_1, col_ref_2 = st.columns([1,3])
            with col_ref_1:
                if st.button("🔄 Refresh Data", help="Download latest 1-min Forex data for the most recent trading day."):
                    try:
                        with st.spinner("⏳ Downloading data... This may take a few minutes."):
                            success = run_data_download(DATA_DIR, CURRENCIES)
                        if success:
                            st.success("Data download finished. Rerun triggered.")
                            st.rerun()
                        else: st.error("Data download process encountered issues.")
                    except Exception as e: st.error(f"Data download error: {e}"); st.exception(traceback.format_exc()); st.rerun()
            with col_ref_2:
                 st.caption(f"Data last updated around: **{last_updated}**")

        st.markdown("---")

        st.subheader("🔍 Arbitrage Opportunity Finder")
        st.markdown(f"""
        Find profitable cycles starting and ending with the chosen **Base Currency**.
        Uses `simple_cycles` (limited to **{MAX_ARBITRAGE_CYCLE_LENGTH}** hops) and verifies profitability against **Transaction Costs**.
        *Note: This search can be slow depending on graph complexity.*
        """)

        if not G_nodes:
            st.warning("Graph data not loaded or empty. Please 'Refresh Data' first.", icon="⚠️")
        else:
            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                last_base = st.session_state.arbitrage_result.get('base_currency', 'USD')
                default_idx_arb = 0
                if nodes_available:
                    try: default_idx_arb = nodes_available.index(last_base) if last_base in nodes_available else (nodes_available.index('USD') if 'USD' in nodes_available else 0)
                    except ValueError: default_idx_arb = 0
                selected_base = st.selectbox("Select Base Currency:", options=nodes_available, index=default_idx_arb, disabled=not nodes_available, key="tab2_base_select")

            with col_cfg2:
                transaction_cost = st.slider("Transaction Cost (%) per hop:", min_value=0.0, max_value=0.25, value=0.05, step=0.005, format="%.3f%%", key="tab2_cost_slider", help="Estimated cost per currency conversion.")

            if st.button(f"▶️ Find Arbitrage for {selected_base}", disabled=not selected_base, type="primary"):
                opps = []; search_dur = 0.0; cycle_nodes_list = []
                with st.spinner(f"⏳ Searching for arbitrage cycles for {selected_base} (max len {MAX_ARBITRAGE_CYCLE_LENGTH})..."):
                    # Ensure nodes/edges are passed as tuples for caching
                    opps, search_dur, cycle_nodes_list = find_multiple_arbitrage_cycles(
                        tuple(G_nodes), tuple(G_edges), selected_base, transaction_cost, max_len=MAX_ARBITRAGE_CYCLE_LENGTH
                    )
                st.session_state.arbitrage_result['opportunities'] = opps
                st.session_state.arbitrage_result['duration'] = search_dur
                st.session_state.arbitrage_result['cycle_nodes_list'] = cycle_nodes_list
                st.session_state.arbitrage_result['base_currency'] = selected_base
                st.session_state.arbitrage_result['selected_cycle_index'] = 0 # Reset selection
                st.rerun() # Update UI

            st.markdown("---")

            # --- Display Results ---
            with st.container(): # Group results display
                st.subheader("Scan Results")
                search_duration_result = st.session_state.arbitrage_result.get('duration', 0.0)
                opportunities_found = st.session_state.arbitrage_result.get('opportunities', [])
                result_base_currency = st.session_state.arbitrage_result.get('base_currency', None)

                if result_base_currency:
                     st.write(f"Showing results for **{result_base_currency}** (Search time: {search_duration_result:.3f} seconds)")

                     if opportunities_found:
                         num_opps = len(opportunities_found)
                         st.success(f"✅ Found **{num_opps}** potential arbitrage cycle(s)!", icon="🎉")

                         opps_df = pd.DataFrame(opportunities_found)
                         opps_df['Profit (%)'] = opps_df['Profit (%)'].map('{:.4f}%'.format)
                         opps_df['End Value'] = opps_df['End Value'].map('{:.6f}'.format)
                         opps_df = opps_df.rename(columns={'End Value': f'End Value (per 1 {result_base_currency})', 'Nodes': 'Hops'})
                         opps_df = opps_df[['Profit (%)', 'Hops', 'Path', f'End Value (per 1 {result_base_currency})']]

                         if num_opps > MAX_ARBITRAGE_PATHS_DISPLAY:
                              st.info(f"Displaying top {MAX_ARBITRAGE_PATHS_DISPLAY} opportunities by profit.")
                              opps_df_display = opps_df.head(MAX_ARBITRAGE_PATHS_DISPLAY)
                         else:
                              opps_df_display = opps_df
                         st.dataframe(opps_df_display, use_container_width=True)

                         # --- Cycle Selection ---
                         st.markdown("---")
                         st.markdown("**Graph Highlighting & Profit Simulation**")
                         # Use opps_df_display row count for options
                         if not opps_df_display.empty:
                             cycle_options = [f"Cycle #{i+1} ({opps_df_display.iloc[i]['Profit (%)']})" for i in range(len(opps_df_display))]
                             current_selection = st.session_state.arbitrage_result.get('selected_cycle_index', 0)
                             if current_selection >= len(cycle_options): current_selection = 0

                             # Use columns for selection and simulation
                             col_sel_high, col_sim_high = st.columns([1,1])

                             with col_sel_high:
                                 selected_cycle_idx_str = st.radio(
                                        "Select cycle to highlight:",
                                        options=cycle_options,
                                        index=current_selection,
                                        key="select_highlight_cycle"
                                        # Removed horizontal=True for better wrapping if many cycles
                                 )
                                 try: st.session_state.arbitrage_result['selected_cycle_index'] = int(re.search(r'\d+', selected_cycle_idx_str).group()) - 1
                                 except: st.session_state.arbitrage_result['selected_cycle_index'] = 0

                             with col_sim_high:
                                 # Ensure index is valid before accessing
                                 selected_idx_calc = st.session_state.arbitrage_result['selected_cycle_index']
                                 if 0 <= selected_idx_calc < len(opportunities_found):
                                     selected_opp_calc = opportunities_found[selected_idx_calc]
                                     start_amount_arb = st.number_input(f"Start Amount ({result_base_currency})", min_value=0.0, value=1000.0, step=100.0, format="%.2f", key=f"tab2_amount_input_{selected_idx_calc}") # Unique key
                                     if start_amount_arb > 0:
                                        try:
                                            end_value_factor = float(selected_opp_calc['End Value']) # Use original float
                                            final_amount = start_amount_arb * end_value_factor
                                            profit_amount = final_amount - start_amount_arb
                                            st.metric(label=f"Final Amount ({result_base_currency})", value=f"{final_amount:,.2f}", delta=f"{profit_amount:,.2f} Profit")
                                        except (ValueError, KeyError, IndexError) as e: st.error(f"Calc error: {e}")
                                 else:
                                      st.warning("Selected cycle index is invalid for calculation.")

                         else:
                              st.info("No opportunities available in the current display to select.")


                     elif search_duration_result > 0:
                          st.info(f"ℹ️ No arbitrage opportunities found for {result_base_currency} with current settings.")

                else:
                     st.info("ℹ️ Select a base currency and click 'Find Arbitrage'.")

        st.markdown("---")

        # --- Network Graph Display ---
        with st.container(): # Group graph display
            st.subheader("📊 Network Graph")
            if G_nodes:
                st.write(f"Displaying **{len(G_nodes)}** currencies and **{len(G_edges)}** direct exchange rates.")
                highlight_nodes = None; highlight_edges = None
                graph_base_currency = st.session_state.arbitrage_result.get('base_currency', None)
                selected_index_for_graph = st.session_state.arbitrage_result.get('selected_cycle_index', 0)
                cycle_nodes_list_for_graph = st.session_state.arbitrage_result.get('cycle_nodes_list', [])
                graph_caption = ""

                if graph_base_currency and cycle_nodes_list_for_graph and 0 <= selected_index_for_graph < len(cycle_nodes_list_for_graph):
                     highlight_nodes = cycle_nodes_list_for_graph[selected_index_for_graph]
                     highlight_edges = list(zip(highlight_nodes[:-1], highlight_nodes[1:]))
                     graph_caption = f"Highlighting selected cycle #{selected_index_for_graph + 1} for {graph_base_currency}"
                elif graph_base_currency:
                     graph_caption = f"Highlighting selected base currency: {graph_base_currency}"
                if graph_caption: st.caption(graph_caption)

                with st.spinner("🎨 Generating network visualization..."):
                    graph_display_html = generate_pyvis_html_string(
                        tuple(G_nodes), tuple(G_edges),
                        selected_currency=graph_base_currency,
                        highlight_cycle_nodes=highlight_nodes,
                        highlight_cycle_edges=highlight_edges
                    )

                if graph_display_html:
                    try: components.html(graph_display_html, height=750, scrolling=False)
                    except Exception as e: st.error(f"Could not display graph: {e}"); st.text(traceback.format_exc())
                else: st.warning("Could not generate graph visualization. Check logs.")
            else:
                st.warning("Cannot display network. Please 'Refresh Data'.", icon="⚠️")


    # --- Tab 3: Pair Analysis ---
    with tab3:
        st.header("💹 Currency Pair Analysis")
        st.write(f"Using data updated around: **{last_updated}**")
        st.markdown("Visualize price action and volatility for a selected currency pair.")

        if not nodes_available:
             st.warning("Currency list not available. Refresh data on the 'Network & Arbitrage' tab.", icon="⚠️")
        else:
            st.markdown("---")
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                 default_base_idx = nodes_available.index('EUR') if 'EUR' in nodes_available else 0
                 base_curr = st.selectbox("Select Base Currency:", options=nodes_available, index=default_base_idx, key="tab3_base")
            with col_sel2:
                quote_options = [q for q in nodes_available if q != base_curr]
                sel_quote_idx = 0; quote_disabled = not quote_options
                if quote_options:
                    if 'USD' in quote_options: sel_quote_idx = quote_options.index('USD')
                quote_curr = st.selectbox("Select Quote Currency:", options=quote_options, index=sel_quote_idx, key="tab3_quote", disabled=quote_disabled)

            if base_curr and quote_curr:
                selected_symbol = f"{base_curr}{quote_curr}=X"
                st.markdown("---")
                st.subheader(f"Analysis for: **{base_curr}/{quote_curr}** ({selected_symbol})")

                if selected_symbol in EXCLUDED_PAIRS:
                    st.warning(f"{selected_symbol} is in the excluded list. No data loaded.", icon="🚫")
                else:
                    with st.spinner(f"⏳ Loading data for {selected_symbol}..."):
                         pair_df = load_pair_data(selected_symbol, DATA_DIR)

                    if pair_df is not None and not pair_df.empty:
                        with st.container():
                            st.markdown("#### 📈 Candlestick Chart (1-Minute Data)")
                            if all(col in pair_df.columns for col in ['Open', 'High', 'Low', 'Close']):
                                try:
                                    fig_candle = go.Figure(data=[go.Candlestick(x=pair_df.index, open=pair_df['Open'], high=pair_df['High'], low=pair_df['Low'], close=pair_df['Close'], name=selected_symbol)])
                                    fig_candle.update_layout(title=f"{selected_symbol} Candlestick", xaxis_title="Time", yaxis_title="Price", xaxis_rangeslider_visible=False, height=500, margin=dict(l=20, r=20, t=40, b=20))
                                    st.plotly_chart(fig_candle, use_container_width=True)
                                except Exception as e: st.error(f"Could not generate candlestick: {e}"); st.text(traceback.format_exc())
                            else: st.warning("Full OHLC data not available for candlestick chart.")

                        st.markdown("---")

                        with st.container():
                            st.markdown("#### 📊 Volatility Metrics")
                            if 'Close' in pair_df.columns:
                                std_dev_1min, annual_vol = calculate_volatility(pair_df.copy())
                                vol_col1, vol_col2 = st.columns(2)
                                with vol_col1:
                                    if std_dev_1min is not None: vol_col1.metric(label="Std Dev (1-Min Log Returns)", value=f"{std_dev_1min:.6f}")
                                    else: vol_col1.info("Could not calculate 1-min volatility.")
                                with vol_col2:
                                    if annual_vol is not None: vol_col2.metric(label="Annualized Volatility (Est.)", value=f"{annual_vol:.2%}")
                                    else: vol_col2.info("Could not estimate annualized volatility.")
                                st.caption("Volatility based on log returns for the loaded day. Annualized figure is a rough estimate.")
                            else: st.warning("Cannot calculate volatility: 'Close' column missing.")

                    elif os.path.exists(os.path.join(DATA_DIR, f"{selected_symbol}_{get_target_date_for_latest_data().strftime('%Y-%m-%d')}.csv")):
                         st.warning(f"Data file found for {selected_symbol}, but failed to load/process. Check console.", icon="⚠️")
                    else:
                        st.error(f"Data not found for {selected_symbol}. Please 'Refresh Data' on the 'Network & Arbitrage' tab.", icon="❌")
            else:
                st.info("ℹ️ Select both a base and quote currency to see the analysis.")