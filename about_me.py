import streamlit as st
from PIL import Image
import pandas as pd

def show():
    """Main function to display About Me content"""

    # Page header
    st.markdown('<div class="main-header">About Me</div>', unsafe_allow_html=True)
    
    # Profile section
    col1, col2, col3 = st.columns([1, 2, 3])
    
    with col1:
        # Profile picture placeholder - replace with your own image path
        st.image("/Users/cameronroberts/Portfolio/stock_images/Cameron Roberts.jpg", width=230)        
        
    with col2:
        st.subheader("Contact")
        st.markdown("""
        * 📧 [thecjrobs@gmail.com](mailto:thecjrobs@gmail.com)
        * 🔗 [LinkedIn](https://www.linkedin.com/in/cjrobs/)
        * 💻 [GitHub](https://github.com/CJRobs)
        * 📄 [Download Resume](link-to-your-resume.pdf)
        """)

    with col3:
        # Bio section
        st.subheader("Hello, I'm Cameron")
        st.markdown("""
        I am a Results-driven data professional with a strong foundation in statistics, economics and computer science, 
        passionate about transforming complex data into actionable insights for impactful business decisions. 
        Proficient in Python, R, SQL and Cypher, I consistently deliver solutions that enhance decision-making and optimise performance. 
        
        Leveraging frameworks such as PyTorch, Scikit-learn and Neo4j I have tackled complex optimisation problems, 
        improved predictive forecasts, and advanced model development. Eager to embrace new challenges, 
        I am dedicated to expanding my skill set and driving meaningful change through innovative data science.
        
        This portfolio showcases some of my key projects that demonstrate my skills in 
        data analysis, machine learning, and data visualization, with a focus on applications 
        in finance, fintech, energy markets, and insurance.
        """)

    # Skills section
    st.header("Skills")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Languages & Skills")
        st.markdown("""
        * Python (pandas, numpy)
        * R (dplyr, tidyrm, ggplot2, shiny)
        * Cypher (Graph Query Language)
        * SQL & Database Management (ERD diagrams)
        * Data Storage (JSON & Parquets)
        * Javascript, HTML & CSS
        * Haskell (Functional Programming)
        """)
    
    with col2:
        st.subheader("Domain Expertise")
        st.markdown("""
        * A/B & Incrementally Tests
        * Geo Experiments
        * Model Forecasting
        * Customer Segmentation
        * Bayesian Regression Modelling
        * Optimizations
        * Operational Efficiency & Automation
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
    st.header("Experience")
    
    # Format: Company, Position, Date Range, Description
    experiences = [
        {
            "company": "Independent Marketing Sciences",
            "company_url": "https://im-sciences.com/",
            "position": "Lead Data Analyst",
            "period": "Nov 2024 - Present",
            "description": """
            • Text
            """
        },
        {
            "company": "Independent Marketing Sciences",
            "company_url": "https://im-sciences.com/",
            "position": "Senior Data Analyst",
            "period": "June 2023 - October 2024",
            "description": """
            • Text
            """
        },
        {
            "company": "Omnicom Group",
            "company_url": "https://www.omnicomgroup.com/",
            "position": "Data Analyst",
            "period": "February 2022 - May 2023",
            "description": """
            • Text
            """
        }
    ]
    
    for exp in experiences:
        col1, col2 = st.columns([2, 3])
        with col1:
            if "company_url" in exp:
                st.markdown(f"### [{exp['company']}]({exp['company_url']})")
            else:
                st.subheader(exp["company"])
            st.write(exp["position"])
            st.write(exp["period"])
        with col2:
            st.markdown(exp["description"])
        st.markdown("---")
    
    # Education section
    st.header("Education")
    
    col1, col2, col3, col4, = st.columns([2, 3 , 4, 5])
    with col1:
        st.markdown("### [University of Bath](https://www.bath.ac.uk/)")
        st.write("Master of Science in Computer Science")
        st.write("2020 - 2021")
    with col2:
        st.markdown("""
        • Grade: Distinction
        • Thesis: "Deep Learning Architectures for High-Frequency Algorithmic Trading: A Comparative Performance Analysis"
        • Relevant Courses: Functional Programming, Cryptography, Deep Reinforcement Learning, Software Engineering, Intelligent Control & Cognitive Systems. 
        """)
    with col3:
        st.markdown("### [University of Reading](https://www.reading.ac.uk/)")
        st.write("Bachelors of Science in Economics")
        st.write("2017 - 2020")
    with col4:
        st.markdown("""
        • Grade: First Class (1.1)
        • Thesis: "Spatial Determinants of Agricultural Productivity: A Geospatial Regression Analysis of Yield Variability"
        • Relevant Courses: Applied Econometrics, Financial Economics, Accounting & Finance, Money & Banking, History of Economic Thought, Advanced Microeconomics & Advanced Macroeconomics.
        """)
        
        
    # Certificates section
    st.header("Certifications")
    
    certs = [
        "AWS Certified Machine Learning - Specialty",
        "Google Professional Data Engineer",
        "Deep Learning Specialization - Coursera",
        "Financial Risk Manager (FRM) - GARP"
    ]
    
    # Display certifications in two columns
    cols = st.columns(2)
    for i, cert in enumerate(certs):
        cols[i % 2].markdown(f"• {cert}")
    
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