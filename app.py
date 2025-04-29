import streamlit as st
import project1
import project2
import about_me
import sys
import io

class WarningFilter(io.StringIO):
    def write(self, s):
        if "missing ScriptRunContext" not in s and "No runtime found" not in s:
            sys.__stderr__.write(s)
                
sys.stderr = WarningFilter()

# Configure the page
st.set_page_config(
    page_title="Cameron Roberts",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .project-description {
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# Create the sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["About Me", "Currency Arbitrage Network", "Equity Valuations"])

# Display the selected page
if selection == "About Me":
    about_me.show()
elif selection == "Currency Arbitrage Network":
    project1.show()
elif selection == "Equity Valuations":
    project2.show()