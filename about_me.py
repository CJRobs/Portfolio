import streamlit as st
from PIL import Image
import pandas as pd
import base64

def show():
    """Main function to display About Me content"""

    # Page header
    st.markdown('<div class="main-header">About Me: Cameron Roberts</div>', unsafe_allow_html=True)
    
    # Profile section
    col1, col2, col3 = st.columns([1, 2, 3])
    
    with col1:
        # Profile picture placeholder - replace with your own image path
        st.image("stock_images/Cameron Roberts.jpg", width=230)        
        
    with col2:            
        st.subheader("Contact")
        
        # Regular links
        st.markdown("""
        * 📧 [thecjrobs@gmail.com](mailto:thecjrobs@gmail.com)
        * 🔗 [LinkedIn](https://www.linkedin.com/in/cjrobs/)
        * 💻 [GitHub](https://github.com/CJRobs)
        """)
        
        # Create a download button for the resume instead of a markdown link
        with open("stock_images/Cameron-Roberts-Resume-Two-Pages.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()
        
        st.download_button(
            label="📄 Download Resume",
            data=PDFbyte,
            file_name="Cameron-Roberts-Resume.pdf",
            mime="application/pdf"
        )
        
    with col3:
        st.markdown("""
            Text about me goes here why do i want to work in financial markets
        """)

    # Skills section
    st.header("Skills")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Languages & Skills")
        st.markdown("""
        * Python (pandas, numpy)
        * R (dplyr, tidyrm, ggplot2, shiny)
        * SQL & Database Management (Big Query & ERD diagrams)
        * Cypher (Graph Query Language)
        * Data Storage (JSON & Parquets)
        * Javascript, HTML & CSS
        * Haskell (Functional Programming)
        """)
    
    with col2:
        st.subheader("Domain Expertise")
        st.markdown("""
        * Geo Experiments, A/B & Incrementally Tests
        * Model Forecasting
        * Customer Segmentation
        * Bayesian Time Series Regression Modelling
        * Optimisations
        * Operational Efficiency & Automation
        * Knowledge Graphs
        """)
    
    with col3:
        st.subheader("Tools & Packages")
        st.markdown("""
        * Jupyter Notebooks, VS Code, R Studio
        * Git & GitHub
        * Neo4j & Langchain
        * Streamlit, Dash, Flask, Shiny
        * PowerBI
        * PostgreSQL
        * Scikit-learn, PyTorch, SciPy
        """)
    
    # Experience section
    st.header("Employment History")
    # Format: Company, Position, Date Range
    experiences = [
        {
            "company": "Independent Marketing Sciences",
            "company_url": "https://im-sciences.com/",
            "highest_position": "Lead Data Analyst",
            "period": "01 June 2023 - Present",
        },
        {
            "company": "Omnicom Group",
            "company_url": "https://www.omnicomgroup.com/",
            "highest_position": "Data Analyst",
            "period": "01 February 2022 - 31 May 2023",
        }
    ]
    
    for exp in experiences:
        col1, col2 = st.columns([2, 3])
        with col1:
            if "company_url" in exp:
                st.markdown(f"### [{exp['company']}]({exp['company_url']})")
            else:
                st.subheader(exp["company"])
            st.write(exp["highest_position"])
            st.write(exp["period"])
        st.markdown("---")
    
    # Education section
    st.header("Education")
    
    col1, col2, = st.columns([2, 3])
    with col1:
        st.markdown("### [University of Bath](https://www.bath.ac.uk/)")
        st.write("Master of Science in Computer Science")
        st.write("2020 - 2021")
    with col2:
        st.markdown("### [University of Reading](https://www.reading.ac.uk/)")
        st.write("Bachelors of Science in Economics")
        st.write("2017 - 2020")
                
    # Projects highlight section
    st.header("Project Highlights")
    st.markdown("""
    Explore my projects using the navigation menu on the left. Here's a quick overview:
    
    **Project 1:** Currency Exchange Network: Detecting Arbitrage Opportunities through Graph-Based Optimization
    
    **Project 2:** [Brief one-line description]
    
    **Project 3:** [Brief one-line description]
    """)
    
    # Call to action
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #3a3a3a; border-radius: 10px; color: white;">
    <h3 style="color: white;">Interested in collaborating or hiring?</h3>
    <p>I'm always open to discussing new projects, opportunities, and ideas.</p>
    <p>📧 <a href="mailto:thecjrobs@gmail.com" style="color: #4da6ff;">thecjrobs@gmail.com</a></p>
    </div>
    """, unsafe_allow_html=True)

# This function will be called from the main app
if __name__ == "__main__":
    show()