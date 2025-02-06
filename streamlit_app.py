import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from typing import Tuple, Callable

BRAND_COLORS = {
    'primary': '#1b3a6f',
    'secondary': '#c5a46d',
    'background': '#f5f1e3',
    'accent1': '#5ab9ea',
    'accent2': '#b7bbc2'
}

DISTRIBUTIONS = {
    'Normal (e.g., Blood Pressure Readings)': stats.norm,
    'Beta (e.g., Medical Test Accuracy)': stats.beta,
    'Gamma (e.g., Patient Wait Times)': stats.gamma,
    'Lognormal (e.g., Drug Absorption Rates)': stats.lognorm,
    'Exponential (e.g., Time Between Hospital Admissions)': stats.expon
}

def get_distribution_params(dist_name: str) -> Tuple[dict, Callable]:
    if 'Normal' in dist_name:
        mu = st.slider('Mean (μ)', -10.0, 10.0, 0.0)
        sigma = st.slider('Standard Deviation (σ)', 0.1, 5.0, 1.0)
        return {'loc': mu, 'scale': sigma}, lambda x: DISTRIBUTIONS[dist_name].pdf(x, **{'loc': mu, 'scale': sigma})
    
    elif 'Beta' in dist_name:
        a = st.slider('α (Shape)', 0.1, 10.0, 2.0)
        b = st.slider('β (Shape)', 0.1, 10.0, 2.0)
        return {'a': a, 'b': b}, lambda x: DISTRIBUTIONS[dist_name].pdf(x, a, b)
    
    elif 'Gamma' in dist_name:
        a = st.slider('Shape (k)', 0.1, 10.0, 2.0)
        scale = st.slider('Scale (θ)', 0.1, 5.0, 1.0)
        return {'a': a, 'scale': scale}, lambda x: DISTRIBUTIONS[dist_name].pdf(x, a, scale=scale)
    
    elif 'Lognormal' in dist_name:
        s = st.slider('Shape (s)', 0.1, 2.0, 0.5)
        scale = st.slider('Scale', 0.1, 5.0, 1.0)
        return {'s': s, 'scale': scale}, lambda x: DISTRIBUTIONS[dist_name].pdf(x, s, scale=scale)
    
    else:  # Exponential
        scale = st.slider('Scale (β)', 0.1, 5.0, 1.0)
        return {'scale': scale}, lambda x: DISTRIBUTIONS[dist_name].pdf(x, scale=scale)

def main():
    st.set_page_config(page_title="Medical Statistics Explorer")
    
    st.title("Medical Distribution Explorer")
    
    # Distribution selection
    dist_name = st.selectbox("Select Distribution", list(DISTRIBUTIONS.keys()))
    
    # Get distribution parameters and PDF function
    params, pdf_func = get_distribution_params(dist_name)
    
    # Sampling controls
    show_samples = st.checkbox("Show Sample Distribution", value=False)
    if show_samples:
        n_samples = st.slider("Number of Samples", 1, 500, 100)
    
    # Generate distribution data
    if 'Normal' in dist_name:
        range = np.abs(params['loc']) + (4*params['scale'])
        x = np.linspace(-range, range, 1000)
    else:
        x = np.linspace(0, 10, 1000)
    y = pdf_func(x)
    
    # Calculate moments
    dist = DISTRIBUTIONS[dist_name](**params)
    mean = dist.mean()
    std = dist.std()
    skew = dist.stats(moments='s')
    kurt = dist.stats(moments='k')
    
    # Create plot
    fig = go.Figure()
    
    # Plot theoretical distribution
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', 
                            name='Theoretical'))
    
    # Plot sampled distribution if requested
    if show_samples:
        samples = dist.rvs(size=n_samples)
        fig.add_trace(go.Histogram(x=samples, 
                                 histnorm='probability density',
                                 name='Sampled',
                                 opacity=0.7,
                                 nbinsx=30))
    
    fig.update_layout(
        title=dict(
            text=f"{dist_name.split('(')[0].strip()} Distribution",
            font=dict(size=24, color=BRAND_COLORS['primary'])
        ),
        showlegend=show_samples
    )
    
    st.plotly_chart(fig)


     # Display statistics
    st.subheader("Theoretical Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{mean:.2f}")
    with col2:
        st.metric("Std Dev", f"{std:.2f}")
    with col3:
        st.metric("Skewness", f"{float(skew):.2f}")
    with col4:
        st.metric("Kurtosis", f"{float(kurt):.2f}")
        
    if show_samples:
        st.subheader("Sample Statistics")
        sample_mean = np.mean(samples)
        sample_std = np.std(samples)
        sample_skew = stats.skew(samples)
        sample_kurt = stats.kurtosis(samples)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sample Mean", f"{sample_mean:.2f}")
        with col2:
            st.metric("Sample Std", f"{sample_std:.2f}")
        with col3:
            st.metric("Sample Skewness", f"{sample_skew:.2f}")
        with col4:
            st.metric("Sample Kurtosis", f"{sample_kurt:.2f}")

if __name__ == "__main__":
    main()