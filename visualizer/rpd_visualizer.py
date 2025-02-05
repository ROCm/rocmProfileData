"""
MIT License

ROCm Profile Data Visualizer

A visualization tool for ROCm Profile Data (.rpd) files that provides interactive
analysis of kernel executions, memory operations, and API calls.


"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

def load_rpd_data(rpd_file):
    """Load data from RPD file into DataFrames"""
    conn = sqlite3.connect(rpd_file)
    
    # First, get the string mappings
    strings_df = pd.read_sql_query("SELECT id, string FROM rocpd_string", conn)
    strings_dict = dict(zip(strings_df.id, strings_df.string))
    
    # Kernel executions
    kernel_query = """
    SELECT 
        k.kernelName_id,
        CAST(o.gpuId AS INTEGER) as gpu_id,
        CAST(o.queueId AS INTEGER) as queue_id,
        CAST(o.start AS INTEGER) as begin_ns,
        CAST(o.end AS INTEGER) as end_ns,
        CAST(k.gridX AS INTEGER) as gridX,
        CAST(k.gridY AS INTEGER) as gridY,
        CAST(k.gridZ AS INTEGER) as gridZ,
        CAST(k.workgroupX AS INTEGER) as workgroupX,
        CAST(k.workgroupY AS INTEGER) as workgroupY,
        CAST(k.workgroupZ AS INTEGER) as workgroupZ
    FROM rocpd_op o
    JOIN rocpd_api_ops ao ON o.id = ao.op_id
    JOIN rocpd_kernelapi k ON ao.api_id = k.api_ptr_id
    ORDER BY o.start
    """
    
    # Memory operations
    memory_query = """
    SELECT 
        CAST(o.gpuId AS INTEGER) as gpu_id,
        CAST(o.queueId AS INTEGER) as queue_id,
        CAST(o.start AS INTEGER) as begin_ns,
        CAST(o.end AS INTEGER) as end_ns,
        CAST(c.size AS INTEGER) as bytes,
        CAST(c.kind AS INTEGER) as kind,
        CAST(c.sync AS INTEGER) as sync
    FROM rocpd_op o
    JOIN rocpd_api_ops ao ON o.id = ao.op_id
    JOIN rocpd_copyapi c ON ao.api_id = c.api_ptr_id
    ORDER BY o.start
    """
    
    # API calls
    api_query = """
    SELECT 
        a.apiName_id,
        CAST(a.start AS INTEGER) as begin_ns,
        CAST(a.end AS INTEGER) as end_ns,
        a.args_id,
        CAST(a.pid AS INTEGER) as pid,
        CAST(a.tid AS INTEGER) as tid
    FROM rocpd_api a
    ORDER BY a.start
    """
    
    kernels_df = pd.read_sql_query(kernel_query, conn)
    memory_df = pd.read_sql_query(memory_query, conn)
    api_df = pd.read_sql_query(api_query, conn)
    
    # Map string IDs to actual strings
    if not kernels_df.empty:
        kernels_df['name'] = kernels_df['kernelName_id'].map(strings_dict)
    
    if not api_df.empty:
        api_df['name'] = api_df['apiName_id'].map(strings_dict)
        api_df['args'] = api_df['args_id'].map(strings_dict)
    
    # Add operation types to memory operations
    if not memory_df.empty:
        memory_df['name'] = memory_df['kind'].apply(
            lambda x: 'HtoD_Copy' if x == 1 else 
                     'DtoH_Copy' if x == 2 else 
                     'DtoD_Copy' if x == 3 else 'Unknown'
        )
        memory_df['name'] = memory_df.apply(
            lambda x: f"{x['name']}_{'Sync' if x['sync'] else 'Async'}", 
            axis=1
        )
    
    # Calculate durations
    for df in [kernels_df, memory_df, api_df]:
        if not df.empty:
            df['duration_ns'] = df['end_ns'] - df['begin_ns']
    
    conn.close()
    return kernels_df, memory_df, api_df

def format_time(ns):
    """Convert nanoseconds to appropriate time unit"""
    if ns < 1000:
        return f"{ns:.0f}ns"
    elif ns < 1000000:
        return f"{ns/1000:.1f}µs"
    elif ns < 1000000000:
        return f"{ns/1000000:.1f}ms"
    else:
        return f"{ns/1000000000:.2f}s"

def create_timeline(kernels_df, memory_df, api_df):
    """Create a simplified visualization with hover text for long names"""
    fig = make_subplots(
        rows=3, 
        cols=1,
        subplot_titles=(
            "<b>Kernel Operations</b>",
            "<b>Memory Operations</b>",
            "<b>API Operations</b>"
        ),
        vertical_spacing=0.3
    )
    
    # Process kernel data
    if not kernels_df.empty:
        kernel_counts = kernels_df['name'].value_counts()
        kernel_durations = kernels_df.groupby('name')['duration_ns'].mean() / 1000  # Convert to μs
        
        # Create shortened labels for x-axis
        shortened_names = [f"K{i+1}" for i in range(len(kernel_counts[:10]))]
        
        fig.add_trace(
            go.Bar(
                name='Execution Count',
                x=shortened_names,  # Use shortened names for display
                y=kernel_counts.values[:10],
                marker_color='rgb(55, 83, 109)',
                text=kernel_counts.values[:10],
                textposition='auto',
                hovertemplate=(
                    "Kernel: %{customdata}<br>" +
                    "Count: %{y}<br>" +
                    "Avg Duration: %{text}"
                ),
                customdata=kernel_counts.index[:10],  # Original kernel names
                textfont=dict(size=10)
            ),
            row=1, col=1
        )
    
    # Process memory operations
    if not memory_df.empty:
        memory_types = memory_df['name'].value_counts()
        memory_sizes = memory_df.groupby('name')['bytes'].mean() / (1024 * 1024)  # Convert to MB
        
        fig.add_trace(
            go.Bar(
                name='Memory Operations',
                x=memory_types.index,
                y=memory_sizes[memory_types.index],
                marker_color='rgb(26, 118, 255)',
                text=[f"{size:.2f} MB" for size in memory_sizes[memory_types.index]],
                textposition='auto',
                hovertemplate=(
                    "Operation: %{x}<br>" +
                    "Size: %{y:.2f} MB<br>" +
                    "Count: %{customdata}"
                ),
                customdata=memory_types.values,
                textfont=dict(size=10)
            ),
            row=2, col=1
        )
    
    # Process API calls
    if not api_df.empty:
        api_counts = api_df['name'].value_counts()
        api_durations = api_df.groupby('name')['duration_ns'].mean() / 1000  # Convert to μs
        
        # Create shortened labels for x-axis
        shortened_api_names = [f"API{i+1}" for i in range(len(api_counts[:10]))]
        
        fig.add_trace(
            go.Bar(
                name='API Calls',
                x=shortened_api_names,  # Use shortened names for display
                y=api_durations[api_counts.index[:10]],
                marker_color='rgb(158, 202, 225)',
                text=[f"{dur:.2f} μs" for dur in api_durations[api_counts.index[:10]]],
                textposition='auto',
                hovertemplate=(
                    "API: %{customdata}<br>" +
                    "Duration: %{y:.2f} μs<br>" +
                    "Count: %{text}"
                ),
                customdata=api_counts.index[:10],  # Original API names
                textfont=dict(size=10)
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=1000,
        showlegend=False,
        title={
            'text': "<b>ROCm Profile Data Analysis</b>",
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        plot_bgcolor='white',
        font=dict(size=12),
    )
    
    # Update axes
    fig.update_xaxes(
        row=1,
        col=1,
        title="Kernel Index",
        showgrid=True,
        gridcolor='lightgray',
        tickangle=0  # Reset tick angle since we have short names now
    )
    fig.update_yaxes(
        row=1,
        col=1,
        title="Execution Count",
        showgrid=True,
        gridcolor='lightgray'
    )
    
    fig.update_xaxes(
        row=2,
        col=1,
        title="Memory Operation Types",
        showgrid=True,
        gridcolor='lightgray',
        tickangle=30  # Reduced angle for better readability
    )
    fig.update_yaxes(
        row=2,
        col=1,
        title="Average Size (MB)",
        showgrid=True,
        gridcolor='lightgray'
    )
    
    fig.update_xaxes(
        row=3,
        col=1,
        title="API Index",
        showgrid=True,
        gridcolor='lightgray',
        tickangle=0  # Reset tick angle since we have short names now
    )
    fig.update_yaxes(
        row=3,
        col=1,
        title="Average Duration (μs)",
        showgrid=True,
        gridcolor='lightgray'
    )
    
    # Add more spacing between plots and adjust margins
    fig.update_layout(
        margin=dict(t=100, b=50, l=100, r=50),
        height=1200,
    )
    
    return fig

def create_summary_stats(kernels_df, memory_df, api_df):
    """Create summary statistics visualizations with proper time units"""
    figures = []
    
    try:
        # Kernel execution time distribution
        if not kernels_df.empty and len(kernels_df) > 0:
            fig_kernel = px.box(
                kernels_df,
                y='duration_ns',
                title="Kernel Execution Time Distribution",
                labels={'duration_ns': 'Duration'}
            )
            fig_kernel.update_layout(
                yaxis=dict(
                    tickformat='.2s',
                    title='Duration (ns)'
                ),
                height=400
            )
            figures.append(fig_kernel)
        
        # Memory operation sizes
        if not memory_df.empty and len(memory_df) > 0:
            memory_df['size_mb'] = memory_df['bytes'] / (1024 * 1024)
            fig_memory = px.bar(
                memory_df,
                x='name',
                y='size_mb',
                title="Memory Operation Sizes",
                labels={'size_mb': 'Size (MB)', 'name': 'Operation'},
                text=memory_df['size_mb'].apply(lambda x: f'{x:.2f} MB')
            )
            fig_memory.update_layout(
                xaxis_tickangle=-45,
                height=400
            )
            figures.append(fig_memory)
        
        # API call duration distribution
        if not api_df.empty and len(api_df) > 0:
            fig_api = px.box(
                api_df,
                y='duration_ns',
                title="API Call Duration Distribution",
                labels={'duration_ns': 'Duration'}
            )
            fig_api.update_layout(
                yaxis=dict(
                    tickformat='.2s',
                    title='Duration (ns)'
                ),
                height=400
            )
            figures.append(fig_api)
        
        return figures
    except Exception as e:
        st.error(f"Error creating summary statistics: {str(e)}")
        return []

def create_interactive_table(df, title):
    """Create an interactive table with filtering and sorting"""
    if df is not None and not df.empty:
        st.subheader(title)
        
        # Configure grid options
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(
            filterable=True,
            sorteable=True,
            resizable=True,
            min_column_width=100
        )
        
        # Enable multi-sorting
        gb.configure_grid_options(enableRangeSelection=True)
        
        # Add pagination
        gb.configure_pagination(
            paginationAutoPageSize=False,
            paginationPageSize=20
        )
        
        # Build grid options
        grid_options = gb.build()
        
        # Create AgGrid component
        return AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            theme='streamlit',
            height=400
        )
    return None

def main():
    st.set_page_config(layout="wide")
    st.title("ROCm Profile Data Visualizer")
    
    uploaded_file = st.file_uploader("Choose an RPD file", type=['rpd'])
    
    if uploaded_file is not None:
        with open("temp.rpd", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        try:
            # Load and process the data
            kernels_df, memory_df, api_df = load_rpd_data("temp.rpd")
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                if not kernels_df.empty:
                    total_kernel_time = kernels_df['duration_ns'].sum() / 1e6  # Convert to ms
                    st.metric("Total Kernel Time", f"{total_kernel_time:.2f} ms")
                    st.metric("Total Kernel Calls", len(kernels_df))
            
            with col2:
                if not memory_df.empty:
                    total_memory = memory_df['bytes'].sum() / (1024 * 1024)  # Convert to MB
                    st.metric("Total Memory Transferred", f"{total_memory:.2f} MB")
                    st.metric("Memory Operations", len(memory_df))
            
            with col3:
                if not api_df.empty:
                    total_api_time = api_df['duration_ns'].sum() / 1e6  # Convert to ms
                    st.metric("Total API Time", f"{total_api_time:.2f} ms")
                    st.metric("API Calls", len(api_df))
            
            # Display main visualization
            st.plotly_chart(create_timeline(kernels_df, memory_df, api_df), use_container_width=True)
            
            # Add Raw Data section with tabs
            st.header("Raw Data")
            tabs = st.tabs(["Kernel Operations", "Memory Operations", "API Calls"])
            
            # Prepare dataframes with readable time units
            if not kernels_df.empty:
                kernels_display_df = kernels_df.copy()
                kernels_display_df['duration_us'] = kernels_display_df['duration_ns'] / 1000
                kernels_display_df = kernels_display_df.drop('duration_ns', axis=1)
            
            if not memory_df.empty:
                memory_display_df = memory_df.copy()
                memory_display_df['duration_us'] = memory_display_df['duration_ns'] / 1000
                memory_display_df['size_mb'] = memory_display_df['bytes'] / (1024 * 1024)
                memory_display_df = memory_display_df.drop(['duration_ns', 'bytes'], axis=1)
            
            if not api_df.empty:
                api_display_df = api_df.copy()
                api_display_df['duration_us'] = api_display_df['duration_ns'] / 1000
                api_display_df = api_display_df.drop('duration_ns', axis=1)
            
            # Display interactive tables in tabs
            with tabs[0]:
                if not kernels_df.empty:
                    create_interactive_table(kernels_display_df, "Kernel Operations Data")
                else:
                    st.info("No kernel operations data available")
            
            with tabs[1]:
                if not memory_df.empty:
                    create_interactive_table(memory_display_df, "Memory Operations Data")
                else:
                    st.info("No memory operations data available")
            
            with tabs[2]:
                if not api_df.empty:
                    create_interactive_table(api_display_df, "API Calls Data")
                else:
                    st.info("No API calls data available")
            
        except Exception as e:
            st.error(f"Error processing RPD file: {str(e)}")

if __name__ == "__main__":
    main() # ... rest of the visualizer code ... 