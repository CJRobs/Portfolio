# project1.py
# (Strict 4-Space Indentation Version 5)
import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
import os
from datetime import datetime, timedelta
from itertools import permutations
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
    if 0 <= today_weekday <= 4:
        return today
    elif today_weekday == 5:
        return today - timedelta(days=1)
    else:
        return today - timedelta(days=2)

def fetch_and_save(symbol, target_date, data_dir):
    start_str = target_date.strftime('%Y-%m-%d'); interval = '1m'
    start_dt = target_date; end_dt = target_date + timedelta(days=1)
    try:
        ticker_obj = yf.Ticker(symbol); data = ticker_obj.history(start=start_dt, end=end_dt, interval=interval)
        if data.empty:
            print(f"❌ No data {symbol} for {start_str}")
            return symbol
        if not data.index.empty:
            data.index = data.index.tz_convert(None); data = data[data.index.date == target_date.date()]
        if data.empty:
            print(f"❌ No data {symbol} specifically for {start_str} after filtering.")
            return symbol
        filename = os.path.join(data_dir, f"{symbol}_{start_str}.csv"); data.to_csv(filename)
        return None
    except Exception as e:
        print(f"⚠️ Error fetch {symbol}: {type(e).__name__}-{e}")
        return symbol

def run_data_download(data_dir, currencies):
    st.info("Starting data download process...")
    target_date = get_target_date_for_latest_data(); target_date_str = target_date.strftime('%Y-%m-%d')
    st.write(f"Targeting data for: **{target_date_str}** ({target_date.strftime('%A')})")
    os.makedirs(data_dir, exist_ok=True)
    currency_pairs = list(permutations(currencies, 2)); all_symbols = [f"{b}{q}=X" for b, q in currency_pairs]
    symbols_to_download = [s for s in all_symbols if s not in EXCLUDED_PAIRS]; excluded_count = len(all_symbols) - len(symbols_to_download)
    st.write(f"Total pairs: {len(all_symbols)}, Excluded: {excluded_count}, Downloading: **{len(symbols_to_download)}**")
    if not symbols_to_download:
        st.warning("No pairs left.")
        return False
    invalid_pairs = []; max_threads = min(20, os.cpu_count()*2 if os.cpu_count() else 4, len(symbols_to_download)); print(f"Using {max_threads} threads...")
    start_dl = time.time(); progress_bar = st.progress(0, text="Initializing download..."); status_text = st.empty(); total_proc = len(symbols_to_download); completed = 0
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        f_to_s = {executor.submit(fetch_and_save, s, target_date, data_dir): s for s in symbols_to_download}
        for future in as_completed(f_to_s):
            symbol = f_to_s[future]; completed += 1; progress = completed / total_proc
            try:
                result = future.result();
            except Exception as exc:
                st.error(f"‼️ Err result {symbol}: {exc}")
                invalid_pairs.append(symbol)
            else:
                if result:
                    invalid_pairs.append(result)
            try:
                 progress_text = f"Downloading... Processed {completed}/{total_proc} pairs."
                 progress_bar.progress(progress, text=progress_text)
                 status_text.text(progress_text)
            except Exception as e:
                print(f"Error updating progress bar: {e}")

    end_dl = time.time(); total_dl = end_dl - start_dl
    try:
        progress_bar.empty()
        status_text.empty()
    except Exception as e:
        print(f"Error clearing progress bar elements: {e}")

    success_dl = total_proc - len(invalid_pairs)
    st.success(f"✅ Download completed in {total_dl:.2f} seconds.")
    st.write(f"Successfully downloaded: {success_dl}, Failed/No Data: {len(invalid_pairs)}")
    if invalid_pairs:
        with st.expander(f"⚠️ Show {len(invalid_pairs)} Failed/Empty Pairs"):
            st.write(sorted(invalid_pairs))
    st.cache_data.clear()
    st.cache_resource.clear()
    keys_to_clear = ['graph_data', 'arbitrage_result']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    return True

# --- Data Loading Helpers ---

def _find_datetime_column(df):
    """ Finds the most likely datetime column using standard indentation. """
    # Ensure consistent 4-space indentation within this function block

    # 1. Prioritize specific column names
    preferred_cols = ['Datetime', 'DateTime', 'Timestamp', 'timestamp', 'Date & Time']
    for col in preferred_cols:
        if col in df.columns:
            # print(f"Found preferred datetime column: {col}") # Optional debug
            return col

    # 2. Check for typical combinations or keywords in names
    for col in df.columns:
        low_col = col.lower()
        if 'datetime' in low_col:
            # print(f"Found column with 'datetime': {col}") # Optional debug
            return col
        if 'date' in low_col and 'time' in low_col:
            # print(f"Found column with 'date' and 'time': {col}") # Optional debug
            return col

    # 3. Fallback: Check for 'date' or 'time' individually
    date_col = None
    time_col = None
    for col in df.columns:
        low_col = col.lower()
        if 'date' in low_col:
            # print(f"Found potential date column: {col}") # Optional debug
            date_col = col
            break # Take the first 'date' column found
    for col in df.columns:
        low_col = col.lower()
        if 'time' in low_col:
            # print(f"Found potential time column: {col}") # Optional debug
            time_col = col
            break # Take the first 'time' column found

    # Return date or time if found, prioritizing date slightly
    if date_col:
        # print(f"Using fallback date column: {date_col}") # Optional debug
        return date_col
    if time_col:
        # print(f"Using fallback time column: {time_col}") # Optional debug
        return time_col

    # 4. Check the DataFrame index
    if isinstance(df.index, pd.DatetimeIndex):
        # print(f"Using existing DatetimeIndex: {df.index.name}") # Optional debug
        return df.index.name # Use index if it's already datetime

    # 5. Check for common unnamed index column resulting from CSV read
    if 'Unnamed: 0' in df.columns:
         try:
             # Attempt to parse without modifying df, raise error if fail
             pd.to_datetime(df['Unnamed: 0'], errors='raise')
             # print("Found potential datetime in 'Unnamed: 0'") # Optional debug
             return 'Unnamed: 0'
         except (ValueError, TypeError, OverflowError, KeyError):
             pass # Ignore if it's not datetime-like

    print("Warning: No suitable datetime column identified.")
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
             # Check if column still exists after dropna before setting index
             if dt_col_name in df.columns:
                 df = df.set_index(dt_col_name)
             else:
                 # This case should be rare if dropna is based on the column itself
                 print(f"Column '{dt_col_name}' disappeared after dropna.")
                 return None
        else:
            print(f"Column '{dt_col_name}' not found for parsing.")
            return None # Column specified doesn't exist

        # Check if index is now DatetimeIndex and process timezone
        if isinstance(df.index, pd.DatetimeIndex):
            if getattr(df.index, 'tz', None) is not None:
                df.index = df.index.tz_convert(None) # Convert to UTC naive
            return df.sort_index() # Ensure sorted
        else:
            # If index setting failed or resulted in non-datetime index
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
    if not os.path.exists(fpath):
        print(f"Data file not found: {fpath}")
        return None
    try:
        df = pd.read_csv(fpath)
        dt_col = _find_datetime_column(df)

        if dt_col:
            df = _parse_datetime_column(df, dt_col)
        else:
            print(f"No datetime column identified for {filename}")
            return None

        if df is None:
            return None # Parsing failed

        # --- Renaming and Column Check (CORRECTED INDENTATION) ---
        rename_map = {}
        for col in df.columns: # Check indentation relative to 'if df is None:' above
            col_low = col.lower()
            if col_low == 'open' and 'Open' not in df.columns:
                rename_map[col] = 'Open'
            elif col_low == 'high' and 'High' not in df.columns:
                rename_map[col] = 'High'
            elif col_low == 'low' and 'Low' not in df.columns:
                rename_map[col] = 'Low'
            elif col_low == 'close' and 'Close' not in df.columns:
                rename_map[col] = 'Close'
            elif col_low == 'adj close' and 'Adj Close' not in df.columns:
                rename_map[col] = 'Adj Close'
            elif col_low == 'volume' and 'Volume' not in df.columns:
                rename_map[col] = 'Volume'

        if rename_map: # Check indentation relative to 'for col...' above
            df = df.rename(columns=rename_map)

        ohlc_cols = ['Open', 'High', 'Low', 'Close'] # Same level as 'if rename_map:'
        present_cols = [col for col in ohlc_cols if col in df.columns]
        if 'Close' not in present_cols:
            if 'Adj Close' in df.columns and pd.api.types.is_numeric_dtype(df['Adj Close']):
                 df=df.rename(columns={'Adj Close': 'Close'})
                 present_cols.append('Close')
            else:
                 for col in df.columns: # Indented inside the 'else'
                     if col not in ['Open','High','Low','Volume'] and pd.api.types.is_numeric_dtype(df[col]):
                          df=df.rename(columns={col: 'Close'})
                          present_cols.append('Close')
                          break # Indented inside the 'if'

        if 'Close' not in df.columns: # Same level as the first 'if Close...'
            print(f"Missing 'Close' price {filename}")
            return None

        for col in present_cols: # Same level
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Close' in df.columns: # Same level
             df = df.dropna(subset=['Close'])
        else:
             print(f"Critical Error: 'Close' column lost in {filename}")
             return None

        if df.empty: # Same level
            print(f"DataFrame empty after NaN drop for {filename}")
            return None

        return df # Already sorted

    except Exception as e:
        st.error(f"Error loading {filename}: {e}", icon="❌")
        st.text(traceback.format_exc())
        return None


# --- Network Graph Generation Logic ---
@st.cache_data(ttl=3600)
def load_and_build_graph(data_dir, excluded_pairs):
    start_load = time.time(); G_nodes_set = set(); G_edges_list = []; latest_ts_dict = {}; valid_found = False
    if not os.path.exists(data_dir):
        st.warning(f"Data directory '{data_dir}' not found.")
        return [], [], "Never", {}
    print(f"--- Loading data from {data_dir} for graph ---"); file_count=0; proc_count=0
    target_date = get_target_date_for_latest_data(); target_date_str = target_date.strftime('%Y-%m-%d')
    all_files = []
    try:
        all_files = [f for f in os.listdir(data_dir) if f.endswith(f"{target_date_str}.csv")]
    except FileNotFoundError:
         st.warning(f"Data directory '{data_dir}' not found when listing files.")
         return [], [], "Never", {}
    print(f"Found {len(all_files)} files for date {target_date_str}")

    for filename in all_files:
        file_count += 1; symbol = filename.split('_')[0]
        if symbol in excluded_pairs or len(symbol) < 6 or not symbol.endswith("=X"):
            continue
        base, quote = symbol[:3], symbol[3:6]; proc_count +=1
        df = load_pair_data(symbol, data_dir) # Uses cache
        if df is not None and not df.empty and 'Close' in df.columns:
            valid_found=True;
            try:
                last_row=df.iloc[-1]; last_ts=last_row.name; rate=last_row['Close']
                if pd.notna(rate) and rate>0:
                    G_nodes_set.add(base)
                    G_nodes_set.add(quote)
                    G_edges_list.append((base, quote, {'weight': float(rate), 'title': f"{base}→{quote}:{rate:.6f}", 'timestamp': last_ts}))
                    if isinstance(last_ts, (pd.Timestamp, datetime)):
                        latest_ts_dict[f"{base}-{quote}"]=last_ts
            except IndexError:
                print(f"IndexError accessing last row for {symbol}. DF empty after load?")
            except Exception as e:
                 print(f"Error processing last row for {symbol}: {e}")

    mrt_str="Never"
    if latest_ts_dict:
        try:
            mrt = max(latest_ts_dict.values())
            if isinstance(mrt, (pd.Timestamp, datetime)):
                 mrt_str=(mrt.tz_convert(None) if hasattr(mrt,'tz') and mrt.tz else mrt).strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                 mrt_str = str(mrt) # Fallback if not a datetime object
        except Exception as e:
            print(f"Error formatting max timestamp: {e}")
            mrt_str = "Error"

    if not valid_found and proc_count > 0 : # Check if files were processed but no valid data found
        st.warning("Processed currency files, but no valid 'Close' prices found for graph edges.", icon="⚠️")
    elif not valid_found and proc_count == 0: # Check if no relevant files were processed
         st.warning(f"No relevant currency files found for date {target_date_str} in {data_dir}", icon="⚠️")

    load_dur = time.time()-start_load
    print(f"--- Graph build done. Processed {proc_count}/{file_count} relevant files in {load_dur:.2f}s ---")
    return list(G_nodes_set), G_edges_list, mrt_str, latest_ts_dict

# --- Pyvis Generation (Using User's Styling) ---
@st.cache_resource(ttl=3600)
def generate_pyvis_html_string(_G_nodes, _G_edges, selected_currency=None, highlight_cycle_nodes=None, highlight_cycle_edges=None):
    """ Generates Pyvis HTML string using user's preferred styling. """
    if not _G_nodes:
        print("Graph nodes list empty for Pyvis.")
        return None
    net = Network(height="700px", width="100%", notebook=False, directed=True, bgcolor="#FFFFFF", font_color="black")

    # Apply barnesHut physics from user script
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250, spring_strength=0.001)

    # Apply set_options string from user script
    net.set_options("""
    const options = {
      "nodes": {
        "font": {"size": 16, "face": "Tahoma", "color": "black"},
        "color": {"border": "#222222", "background": "#6EA7FF"},
        "shape": "circle",
        "size": 30
      },
      "edges": {
        "color": {"inherit": false, "color": "#666666"},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "smooth": {"enabled": true, "type": "dynamic"},
        "font": {"size": 12, "face": "Arial", "color": "#555555", "align": "middle"}
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -80000,
          "centralGravity": 0.3,
          "springLength": 250,
          "springConstant": 0.001,
          "damping": 0.09
        },
        "minVelocity": 0.75,
        "solver": "barnesHut"
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": true,
        "hover": true
      }
    }
    """)

    # Define colors for highlighting
    base_highlight_color = "#FF6347"; cycle_node_color = "#90EE90"; cycle_edge_color = "#008000"
    default_node_bg = "#6EA7FF"; default_node_border = "#222222"; default_edge_color = "#666666"

    highlight_cycle_nodes_set = set(highlight_cycle_nodes) if highlight_cycle_nodes else set()
    highlight_cycle_edges_set = set((edge[0], edge[1]) for edge in highlight_cycle_edges) if highlight_cycle_edges else set()

    # Add Nodes
    for node in _G_nodes:
        node_color_bg = default_node_bg
        node_color_border = default_node_border
        node_size = 30 # Default size from options
        node_font_settings = {"size": 20, "bold": True, "color": "black"} # Font from user script

        if node == selected_currency:
            node_color_bg = base_highlight_color
        elif node in highlight_cycle_nodes_set:
            node_color_bg = cycle_node_color

        net.add_node(node, label=node, title=node,
                     color={'background': node_color_bg, 'border': node_color_border},
                     size=node_size,
                     font=node_font_settings)

    # Add Edges
    for source, target, data in _G_edges:
        rate = data.get('weight', 0)
        edge_color = default_edge_color
        edge_width = 1.5 # Default width

        if (source, target) in highlight_cycle_edges_set:
            edge_color = cycle_edge_color
            edge_width = 3.5 # Make cycle edges thicker

        title_str=f"{source} → {target}: {rate:.6f}"
        label_str=f"{rate:.4f}"
        net.add_edge(source, target, value=1, title=title_str, label=label_str, color=edge_color, width=edge_width)
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        temp_html_path = os.path.join(DATA_DIR, f"temp_network_{int(time.time())}.html")
        net.save_graph(temp_html_path)
        with open(temp_html_path, 'r', encoding='utf-8') as f: html_content = f.read()
        try: os.remove(temp_html_path)
        except OSError as e: print(f"Warning: Could not remove temp pyvis file {temp_html_path}: {e}")
        return html_content
    except Exception as e:
        st.error(f"Failed generate Pyvis HTML: {e}", icon="❌")
        print(f"Failed generate Pyvis HTML: {e}"); traceback.print_exc()
        return None


# --- Arbitrage Calculation Logic (RESTORED Bellman-Ford) ---
@st.cache_data(ttl=600)
def find_arbitrage_bellman_ford(_G_nodes, _G_edges, base_currency, transaction_cost_percent):
    """
    Finds ONE profitable arbitrage cycle involving base_currency using Bellman-Ford.
    Returns tuple: (list_with_one_opportunity_dict_or_empty, bf_duration_seconds, cycle_nodes_list_or_None)
    """
    start_build = time.time()
    G = nx.DiGraph()
    if not isinstance(_G_nodes, (list, tuple)) or not isinstance(_G_edges, (list, tuple)):
        return [], 0.0, None
    G.add_nodes_from(_G_nodes)
    try:
        G.add_edges_from(_G_edges) # Assumes _G_edges is list of (u, v, data_dict)
    except Exception as e:
        print(f"Error adding edges: {e}")
        return [], 0.0, None

    if not G or base_currency not in G:
        print(f"Graph empty or base {base_currency} not found.")
        return [], 0.0, None
    cost_mult = 1.0 - (transaction_cost_percent / 100.0)
    if cost_mult <= 0:
        print("Cost multiplier <= 0.")
        return [], 0.0, None

    G_log = nx.DiGraph(); bf_weight_attr = 'bf_weight'; skipped_log = 0
    for u, v, data in G.edges(data=True):
        rate = data.get('weight', 0)
        if rate > 0 and (rate * cost_mult) > 0:
            try:
                log_weight = -math.log(rate * cost_mult)
                G_log.add_edge(u, v, **{bf_weight_attr: log_weight})
            except ValueError:
                skipped_log+=1
                print(f"ValueError calculating log for {u}->{v} with rate {rate} * cost {cost_mult}")
        else:
            skipped_log += 1
    build_dur = time.time() - start_build
    print(f"Built log-graph ({len(G_log.nodes())} N, {len(G_log.edges())} E, {skipped_log} skip) in {build_dur:.3f}s")
    if not G_log.edges() or base_currency not in G_log:
        print("Log-graph empty or base missing.")
        return [], build_dur, None

    print(f"Running Bellman-Ford from {base_currency}")
    opportunities = []; bf_dur = 0.0; neg_cycle_nodes = None
    start_bf = time.time()
    try:
        neg_cycle_nodes = nx.find_negative_cycle(G_log, source=base_currency, weight=bf_weight_attr)
        bf_dur = time.time() - start_bf
        print(f"BF found cycle in {bf_dur:.4f}s: {neg_cycle_nodes}")

        if not neg_cycle_nodes or len(neg_cycle_nodes) < 2:
             print(f"BF cycle invalid len ({len(neg_cycle_nodes) if neg_cycle_nodes else 0}).")
             return [], build_dur + bf_dur, None

        # --- Ensure cycle starts and ends with base_currency ---
        if neg_cycle_nodes[0] != base_currency:
            try:
                 start_index = neg_cycle_nodes.index(base_currency)
                 neg_cycle_nodes = neg_cycle_nodes[start_index:] + neg_cycle_nodes[:start_index]
                 print(f"Reordered cycle to start with base: {neg_cycle_nodes}")
            except ValueError:
                 print(f"Error: Base currency {base_currency} not found in detected cycle {neg_cycle_nodes}")
                 return [], build_dur + bf_dur, None

        # Ensure the cycle nodes list ends where it starts for verification path
        if neg_cycle_nodes[-1] != neg_cycle_nodes[0]:
             # Check if the edge back exists in the original graph
             if G.has_edge(neg_cycle_nodes[-1], neg_cycle_nodes[0]):
                 print(f"Appending start node {neg_cycle_nodes[0]} to complete loop.")
                 neg_cycle_nodes.append(neg_cycle_nodes[0]) # Append start node
             else:
                 print(f"Warning: BF cycle {neg_cycle_nodes} doesn't loop and missing final edge back to start.")
                 return [], build_dur + bf_dur, None # Cannot verify

        # --- Verification ---
        product = 1.0; path_parts_detail = []; valid_check = True
        print(f"Verifying cycle: {neg_cycle_nodes}")
        for i in range(len(neg_cycle_nodes) - 1):
            u, v = neg_cycle_nodes[i], neg_cycle_nodes[i+1]
            try:
                orig_rate = G[u][v]['weight']
                print(f"  Step {i+1}: {u} -> {v}, Rate: {orig_rate:.6f}, CostMult: {cost_mult:.6f}")
                product *= orig_rate * cost_mult
                path_parts_detail.append(f"{u}→{v} ({orig_rate:.4f})")
            except KeyError:
                print(f"CRIT ERR: Edge {u}->{v} in cycle not in original graph G!")
                valid_check = False
                break

        print(f"Verification product: {product:.8f}")
        if valid_check and product > 1.0:
            profit_pc = (product - 1.0) * 100
            path_str_detail = " → ".join(path_parts_detail)
            simple_sequence = " → ".join(neg_cycle_nodes) # Path starts and ends with base
            opportunities.append({
                "Profit (%)": profit_pc,
                "Cycle Sequence": simple_sequence,
                "Hops": len(neg_cycle_nodes)-1,
                "End Value": product,
                "Detailed Path": path_str_detail
            })
            print(f"Verified arbitrage: {simple_sequence}, Profit: {profit_pc:.4f}%")
        elif valid_check:
            print(f"Warn: Cycle {neg_cycle_nodes} found, product ({product:.6f}) <= 1 (after cost).")
            neg_cycle_nodes = None # Clear nodes if not profitable
        else:
            # valid_check is False
             neg_cycle_nodes = None # Invalidate if check failed

    except nx.NetworkXError:
        bf_dur = time.time() - start_bf
        print(f"BF exec time: {bf_dur:.4f}s. No negative cycle found from {base_currency}.")
        return [], build_dur + bf_dur, None
    except Exception as e:
         bf_dur = time.time() - start_bf
         print(f"BF exec time: {bf_dur:.4f}s. Unexpected error during BF/Verification: {e}")
         traceback.print_exc()
         return [], build_dur + bf_dur, None

    return opportunities, build_dur + bf_dur, neg_cycle_nodes if opportunities else None


# --- Streamlit Page Display Logic ---
def show():
    os.makedirs(DATA_DIR, exist_ok=True)
    st.title("📈 Currency Arbitrage & Analysis")

    # Adjust session state initialization for single opportunity result
    if 'graph_data' not in st.session_state:
        st.session_state.graph_data = {'nodes': [], 'edges': [], 'last_updated': "Never"}
    if 'arbitrage_result' not in st.session_state:
        st.session_state.arbitrage_result = {'opportunity': None, 'duration': 0.0, 'cycle_nodes': None, 'base_currency': None}

    # Adjusted Tabs
    tab_list = ["Project Overview", "Network & Arbitrage", "Pair Analysis"]
    tab1, tab2, tab3 = st.tabs(tab_list)

    # --- Tab 1: Project Overview ---
    with tab1:
        st.header("✨ Project Overview: Forex Arbitrage & Analysis")
        st.markdown("""
        This application analyzes foreign exchange (FOREX) market data to:
        1.  Identify potential **cyclical arbitrage opportunities** using the Bellman-Ford algorithm.
        2.  Visualize individual currency pair **price action** (candlestick).

        **Goal:** Provide tools for exploring Forex market relationships and finding potential inefficiencies based on historical minute-data.

        **Methodology:**

        1.  **Data Acquisition:** Fetches recent 1-minute OHLC data (`yfinance`), excludes certain pairs, saves locally. *(Data has inherent delays)*.
        2.  **Network Construction (Tab: Network & Arbitrage):** Builds a directed graph (`networkx`) where Nodes=Currencies, Edges=Exchange Rates (latest 'Close'). Visualized using `pyvis`.
        3.  **Arbitrage Detection (Tab: Network & Arbitrage):**
            * Uses the **Bellman-Ford algorithm** (`networkx.find_negative_cycle`) on a **log-transformed graph** to find *one example* negative cycle (if one exists) reachable from the selected base currency.
            * **Verifies** the found cycle by calculating the **product of original exchange rates** (including transaction costs). A product > 1 confirms the arbitrage.
            * Displays the confirmed opportunity, including the **cycle sequence** (starting and ending with the base currency) and profitability.
            * Highlights the found path on the network graph.
        4.  **Pair Visualization (Tab: Pair Analysis):** Allows selection of Base/Quote, loads 1-min OHLC, displays interactive **Candlestick chart** (`plotly`).

        **Libraries Used:** `yfinance`, `pandas`, `networkx`, `pyvis`, `numpy`, `plotly`.
        """)
        st.markdown("---")
        st.info("ℹ️ **Disclaimer:** This is an analytical tool using historical data. Market conditions change rapidly, and transaction costs/slippage can vary.")

    # --- Load or Get Graph Data ---
    if not st.session_state.graph_data.get('nodes'):
        data_exists = os.path.exists(DATA_DIR) and any(f.endswith('.csv') for f in os.listdir(DATA_DIR))
        if data_exists:
            with st.spinner("⏳ Loading graph data from disk..."):
                nodes_list, edges_list, last_updated, _ = load_and_build_graph(DATA_DIR, EXCLUDED_PAIRS)
                if nodes_list or edges_list:
                    st.session_state.graph_data['nodes'] = nodes_list
                    st.session_state.graph_data['edges'] = edges_list
                    st.session_state.graph_data['last_updated'] = last_updated
                else:
                    st.session_state.graph_data = {'nodes': [], 'edges': [], 'last_updated': "Load Failed"}
        else:
             st.session_state.graph_data = {'nodes': [], 'edges': [], 'last_updated': "No Data Files"}


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
                if st.button("🔄 Refresh Data", help="Download latest 1-min Forex data."):
                    try:
                        with st.spinner("⏳ Downloading data..."):
                            success = run_data_download(DATA_DIR, CURRENCIES)
                        if success:
                            st.success("Data download finished.")
                        else:
                            st.error("Data download process encountered issues.")
                    except Exception as e:
                        st.error(f"Download error: {e}")
                        st.exception(traceback.format_exc())
                    st.rerun() # Rerun after action

            with col_ref_2:
                 st.caption(f"Data last updated around: **{last_updated}**")
        st.markdown("---")

        st.subheader("🔍 Arbitrage Opportunity Finder (Bellman-Ford)")
        st.markdown("Finds **one example** profitable cycle (if one exists) using Bellman-Ford.")

        if not G_nodes:
            st.warning("Graph data not loaded or empty. Please 'Refresh Data' first.", icon="⚠️")
        else:
            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                last_base = st.session_state.arbitrage_result.get('base_currency', 'USD')
                default_idx_arb = 0
                if nodes_available:
                    try:
                        default_idx_arb = nodes_available.index(last_base) if last_base in nodes_available else (nodes_available.index('USD') if 'USD' in nodes_available else 0)
                    except ValueError:
                        default_idx_arb = 0 # Fallback if index fails
                selected_base = st.selectbox("Select Base Currency:", options=nodes_available, index=default_idx_arb, disabled=not nodes_available, key="tab2_base_select_bf") # Unique key

            with col_cfg2:
                # Slider for transaction cost
                transaction_cost = st.slider("Transaction Cost (%) per hop:", min_value=0.0, max_value=0.25, value=0.05, step=0.005, format="%.3f%%", key="tab2_cost_slider_bf", help="Estimated cost per currency conversion.")

            # Button to trigger arbitrage search
            if st.button(f"▶️ Find Arbitrage for {selected_base}", disabled=not selected_base, type="primary"):
                opps_list = []; search_dur = 0.0; cycle_nodes = None
                with st.spinner(f"⏳ Running Bellman-Ford for {selected_base}..."):
                    # Ensure nodes/edges are passed as tuples for caching
                    opps_list, search_dur, cycle_nodes = find_arbitrage_bellman_ford(
                        tuple(G_nodes), tuple(G_edges), selected_base, transaction_cost
                    )
                # Update session state with results
                st.session_state.arbitrage_result['opportunity'] = opps_list[0] if opps_list else None
                st.session_state.arbitrage_result['duration'] = search_dur
                st.session_state.arbitrage_result['cycle_nodes'] = cycle_nodes
                st.session_state.arbitrage_result['base_currency'] = selected_base
                # Rerun necessary to display new results immediately
                st.rerun()

            st.markdown("---")

            # --- Display Results ---
            with st.container():
                st.subheader("Scan Results")
                search_duration_result = st.session_state.arbitrage_result.get('duration', 0.0)
                opportunity_found = st.session_state.arbitrage_result.get('opportunity', None) # Get the single dict
                result_base_currency = st.session_state.arbitrage_result.get('base_currency', None)

                if result_base_currency:
                     st.write(f"Showing results for **{result_base_currency}** (Search time: {search_duration_result:.3f} seconds)")

                     if opportunity_found:
                         st.success(f"✅ Found a potential arbitrage cycle!", icon="🎉")
                         # Prepare DataFrame for display
                         opp_df = pd.DataFrame([opportunity_found]) # Create DF from the single dict
                         # Format columns for display
                         opp_df['Profit (%)'] = opp_df['Profit (%)'].map('{:.4f}%'.format)
                         opp_df['End Value'] = opp_df['End Value'].map('{:.6f}'.format)
                         opp_df = opp_df.rename(columns={'End Value': f'End Value (per 1 {result_base_currency})'})
                         # Select and reorder columns - use NEW 'Cycle Sequence'
                         opp_df_display = opp_df[['Cycle Sequence','Profit (%)', 'Hops', f'End Value (per 1 {result_base_currency})']]
                         # Display the table
                         st.dataframe(opp_df_display, use_container_width=True, hide_index=True) # Hide index for single row

                         # --- Profit Calculation ---
                         st.markdown("**Profit Simulation**")
                         start_amount_arb = st.number_input(f"Start Amount ({result_base_currency})", min_value=0.0, value=1000.0, step=100.0, format="%.2f", key="tab2_amount_input_bf") # Unique key needed if reused
                         if start_amount_arb > 0:
                            try:
                                # Use the original float value stored in the opportunity dict
                                end_value_factor = float(opportunity_found['End Value'])
                                final_amount = start_amount_arb * end_value_factor
                                profit_amount = final_amount - start_amount_arb
                                st.metric(label=f"Final Amount ({result_base_currency})", value=f"{final_amount:,.2f}", delta=f"{profit_amount:,.2f} Profit")
                            except (ValueError, KeyError, TypeError) as e:
                                st.error(f"Calculation error: {e}")

                     elif search_duration_result > 0: # Check if search was run
                          st.info(f"ℹ️ No arbitrage opportunities found for {result_base_currency} with current settings.")
                     # No message if search not run yet (duration == 0)

                else: # No search run yet in this session
                     st.info("ℹ️ Select a base currency and click 'Find Arbitrage'.")

        st.markdown("---")

        # --- Network Graph Display ---
        with st.container():
            st.subheader("📊 Network Graph")
            if G_nodes:
                st.write(f"Displaying **{len(G_nodes)}** currencies and **{len(G_edges)}** direct exchange rates.")
                highlight_nodes = None; highlight_edges = None
                # Get data for highlighting the single cycle found
                graph_base_currency = st.session_state.arbitrage_result.get('base_currency', None)
                cycle_nodes_for_graph = st.session_state.arbitrage_result.get('cycle_nodes', None) # Get the single list
                graph_caption = ""

                # Determine caption and highlighting data
                if graph_base_currency and cycle_nodes_for_graph:
                     highlight_nodes = cycle_nodes_for_graph
                     # Ensure cycle_nodes_for_graph has at least 2 nodes for zipping
                     if len(highlight_nodes) > 1:
                          highlight_edges = list(zip(highlight_nodes[:-1], highlight_nodes[1:]))
                     else:
                          highlight_edges = [] # Handle edge case of 1-node cycle? (Shouldn't happen with verification)
                     graph_caption = f"Highlighting found cycle for {graph_base_currency}"
                elif graph_base_currency:
                     # Highlight just the base if selected but no cycle found
                     highlight_nodes = [graph_base_currency]
                     highlight_edges = []
                     graph_caption = f"Highlighting selected base currency: {graph_base_currency}"

                if graph_caption:
                    st.caption(graph_caption)

                # Generate and display the graph
                with st.spinner("🎨 Generating network visualization..."):
                    graph_display_html = generate_pyvis_html_string(
                        tuple(G_nodes), tuple(G_edges), # Pass tuples for caching
                        selected_currency=graph_base_currency, # Highlight base even if no cycle found/selected
                        highlight_cycle_nodes=highlight_nodes,
                        highlight_cycle_edges=highlight_edges
                    )

                if graph_display_html:
                    try:
                        components.html(graph_display_html, height=750, scrolling=False)
                    except Exception as e:
                        st.error(f"Could not display graph: {e}")
                        st.text(traceback.format_exc())
                else:
                    # generate_pyvis_html_string already printed/logged error
                    st.warning("Graph visualization could not be generated.")
            else:
                # G_nodes is empty
                st.warning("Cannot display network graph. Load data first.", icon="⚠️")


    # --- Tab 3: Pair Analysis ---
    with tab3:
        st.header("💹 Currency Pair Analysis")
        st.write(f"Using data updated around: **{last_updated}**")
        st.markdown("Visualize price action for a selected currency pair.")

        if not nodes_available:
             st.warning("Currency list not available. Refresh data on the 'Network & Arbitrage' tab.", icon="⚠️")
        else:
            st.markdown("---")
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                 default_base_idx = 0
                 if 'EUR' in nodes_available: default_base_idx = nodes_available.index('EUR')
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
                    st.warning(f"{selected_symbol} is in the excluded list.", icon="🚫")
                else:
                    with st.spinner(f"⏳ Loading data for {selected_symbol}..."):
                         pair_df = load_pair_data(selected_symbol, DATA_DIR) # Uses cache

                    if pair_df is not None and not pair_df.empty:
                        # --- Candlestick Chart ---
                        with st.container():
                            st.markdown("#### 📈 Candlestick Chart (1-Minute Data)")
                            # Check necessary columns exist
                            if all(col in pair_df.columns for col in ['Open', 'High', 'Low', 'Close']):
                                try:
                                    fig_candle = go.Figure(data=[go.Candlestick(x=pair_df.index, open=pair_df['Open'], high=pair_df['High'], low=pair_df['Low'], close=pair_df['Close'], name=selected_symbol)])
                                    fig_candle.update_layout(title=f"{selected_symbol} Candlestick", xaxis_title="Time", yaxis_title="Price", xaxis_rangeslider_visible=False, height=500, margin=dict(l=20, r=20, t=40, b=20))
                                    st.plotly_chart(fig_candle, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not generate candlestick: {e}")
                                    st.text(traceback.format_exc())
                            else:
                                st.warning("Full OHLC data not available for candlestick chart (requires Open, High, Low, Close columns).")

                        # --- Volatility Section Removed ---

                    elif os.path.exists(os.path.join(DATA_DIR, f"{selected_symbol}_{get_target_date_for_latest_data().strftime('%Y-%m-%d')}.csv")):
                         # File exists but loading failed
                         st.warning(f"Data file found for {selected_symbol}, but failed to load/process. Check logs.", icon="⚠️")
                    else:
                        # File doesn't exist
                        st.error(f"Data file not found for {selected_symbol}. Please 'Refresh Data' first.", icon="❌")
            else:
                st.info("ℹ️ Select both a base and quote currency to see the analysis.")