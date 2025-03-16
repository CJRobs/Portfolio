import streamlit as st
import project1
import project2
import project3
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
selection = st.sidebar.radio("Go to", ["About Me", "Project 1", "Project 2", "Project 3"])

# Display the selected page
if selection == "About Me":
    about_me.show()
elif selection == "Project 1":
    project1.show()
elif selection == "Project 2":
    project2.show()
elif selection == "Project 3":
    project3.show()