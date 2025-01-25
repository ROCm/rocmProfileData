"""
MIT License

ROCm Profile Data Schema Inspector

A tool to inspect the schema and contents of ROCm Profile Data (.rpd) files.
Provides interactive exploration of tables, columns, and sample data.
"""

import streamlit as st
import sqlite3
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

def create_interactive_table(df, title):
    """Create an interactive table with filtering and sorting"""
    if df is not None and not df.empty:
        # Configure grid options
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(
            filterable=True,
            sorteable=True,
            resizable=True,
            min_column_width=100
        )
        
        # Enable features
        gb.configure_grid_options(enableRangeSelection=True)
        gb.configure_pagination(
            paginationAutoPageSize=False,
            paginationPageSize=20
        )
        
        grid_options = gb.build()
        
        return AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            theme='streamlit',
            height=400
        )
    return None

def find_relationships(table_info):
    """Find relationships between tables based on foreign key patterns"""
    relationships = []
    for table1 in table_info.keys():
        cols1 = [(col[1].lower(), col[2].lower()) for col in table_info[table1]['columns']]
        for table2 in table_info.keys():
            if table1 >= table2:  # Avoid duplicate relationships
                continue
            cols2 = [(col[1].lower(), col[2].lower()) for col in table_info[table2]['columns']]
            
            # Look for foreign key patterns
            for col1_name, col1_type in cols1:
                for col2_name, col2_type in cols2:
                    if (col1_type == col2_type and 
                        ('id' in col1_name or 'key' in col1_name or 
                         'id' in col2_name or 'key' in col2_name)):
                        relationships.append((table1, table2, col1_name, col2_name))
    return relationships

def inspect_rpd(rpd_file):
    """Inspect RPD file schema and contents"""
    conn = sqlite3.connect(rpd_file)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    table_info = {}
    for table in tables:
        table_name = table[0]
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        # Get sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        sample_data = cursor.fetchall()
        
        # Store table information
        table_info[table_name] = {
            'columns': columns,
            'row_count': row_count,
            'sample_data': sample_data
        }
    
    conn.close()
    return table_info

def main():
    st.set_page_config(layout="wide", page_title="RPD Schema Inspector")
    st.title("ROCm Profile Data Schema Inspector")
    
    uploaded_file = st.file_uploader("Choose an RPD file", type=['rpd'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_schema.rpd", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        try:
            # Inspect the RPD file
            table_info = inspect_rpd("temp_schema.rpd")
            relationships = find_relationships(table_info)
            
            # Display summary
            st.header("Database Tables")
            st.metric("Total Tables", len(table_info))
            
            # Create tabs for each table
            table_tabs = st.tabs(list(table_info.keys()))
            
            # Display information for each table
            for tab, (table_name, info) in zip(table_tabs, table_info.items()):
                with tab:
                    # Table information section
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.subheader("Table Structure")
                        columns_df = pd.DataFrame(
                            info['columns'],
                            columns=['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk']
                        )
                        st.dataframe(columns_df[['name', 'type', 'notnull', 'pk']])
                    
                    with col2:
                        st.metric("Row Count", info['row_count'])
                    
                    # Sample data section
                    st.subheader("Sample Data")
                    if info['sample_data']:
                        sample_df = pd.DataFrame(
                            info['sample_data'],
                            columns=[col[1] for col in info['columns']]
                        )
                        create_interactive_table(sample_df, f"Sample Data for {table_name}")
                    else:
                        st.info("No data available in this table")
                    
                    # SQL Query Section
                    st.subheader("Custom Query")
                    query = st.text_area(
                        "Enter SQL query:",
                        value=f"SELECT * FROM {table_name} LIMIT 10;",
                        key=f"query_{table_name}"
                    )
                    
                    if st.button("Run Query", key=f"run_{table_name}"):
                        try:
                            conn = sqlite3.connect("temp_schema.rpd")
                            result_df = pd.read_sql_query(query, conn)
                            conn.close()
                            
                            if not result_df.empty:
                                st.subheader("Query Results")
                                create_interactive_table(result_df, "Query Results")
                            else:
                                st.info("Query returned no results")
                        except Exception as e:
                            st.error(f"Error executing query: {str(e)}")
            
            # Add schema visualization at the bottom
            st.header("Database Schema Visualization")
            
            # Create ER Diagram showing only tables
            mermaid_code = ["erDiagram"]
            
            # Add tables and their columns
            for table_name, info in table_info.items():
                mermaid_code.append(f"    {table_name} {{")
                for col in info['columns']:
                    col_type = col[2].upper()
                    col_name = col[1]
                    is_pk = "PK" if col[5] == 1 else ""
                    mermaid_code.append(f"        {col_type} {col_name} {is_pk}")
                mermaid_code.append("    }")
            
            # Display the ER diagram
            st.markdown(f"```mermaid\n" + "\n".join(mermaid_code) + "\n```")
            
        except Exception as e:
            st.error(f"Error inspecting RPD file: {str(e)}")

if __name__ == "__main__":
    main()