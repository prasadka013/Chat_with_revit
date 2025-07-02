import streamlit as st
import pandas as pd
import boto3
import json
from rapidfuzz import fuzz
import sys
from io import StringIO
import os
import tempfile

AWS_REGION = "us-east-2"
AWS_ACCESS_KEY_ID = "AKIA3U77TXZUEEBJKBH4"          # your AccessKeyId
AWS_SECRET_ACCESS_KEY = "nG1YETlYptB0xHPIlH/UCFB7D8xy3RNRGGnDunZU"  # your SecretAccessKey
MODEL_ARN = "arn:aws:bedrock:us-east-2:801010597480:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"

# Initialize AWS Bedrock client
@st.cache_resource
def get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

# Function to capture print output
class OutputCapture:
    def __init__(self):
        self.output = StringIO()

    def __enter__(self):
        sys.stdout = self.output
        return self.output

    def __exit__(self, *args):
        sys.stdout = sys.__stdout__

# Function to process Excel file
def process_excel(df):
    column_names = df.columns.tolist()
    priority_columns = ["Type Name : String", "Category : String", "Family : String"]
    priority_column_values = {
        col: df[col].dropna().astype(str).unique().tolist()
        for col in priority_columns
        if col in df.columns
    }
    
    priority_column_summary = "\nPriority Column Values:\n"
    for col, values in priority_column_values.items():
        preview = values[:1000]  # only first 1000 values
        priority_column_summary += f"{col} → {preview}\n"
    
    return column_names, priority_column_summary

# Function to get Claude's response with retry logic
def get_claude_response(prompt, client, max_retries=3, delay=2):
    import time
    
    for attempt in range(max_retries):
        try:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8000,
                "temperature": 0.5,
            }
            
            response = client.invoke_model(
                modelId=MODEL_ARN,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            
            body_bytes = response["body"].read()
            body = json.loads(body_bytes.decode("utf-8"))
            
            assistant_text = None
            if "completion" in body:
                assistant_text = body["completion"]
            elif "content" in body:
                for chunk in body["content"]:
                    if chunk.get("type") == "text":
                        assistant_text = chunk.get("text")
                        break
            # console.log(assistant_text)
            return assistant_text
            
        except client.exceptions.ThrottlingException:
            if attempt < max_retries - 1:  # if not the last attempt
                st.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                st.error("Failed to get response after multiple retries. Please try again later.")
                return None
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None

# Streamlit UI
st.title("Excel Chat Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

# After file upload, add size check
if uploaded_file is not None:
    # Check file size (limit to 100MB for example)
    file_size = uploaded_file.size
    if file_size > 300 * 1024 * 1024:  # 100MB in bytes
        st.error("File is too large. Please upload a file smaller than 100MB.")
        st.stop()
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        excel_xl = pd.ExcelFile(file_path)
        main_sheet = excel_xl.sheet_names[0]
        df = pd.read_excel(uploaded_file)
        if df.empty:
            st.error("The uploaded Excel file is empty.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        st.stop()
    st.success("Excel file loaded successfully!")
    
    # Process Excel file
    column_names, priority_column_summary = process_excel(df)
    
    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your Excel data"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Prepare prompt for Claude
        claude_prompt = f"""
        You are a Python pandas assistant. The user uploaded an Excel file.

        Start by importing:
        import pandas as pd
        load the df below:
        file_path = {repr(file_path)}
        df = pd.read_excel(file_path, sheet_name='{main_sheet}')

        {priority_column_summary}

        first search for the terms are there in the above data for user query this is {prompt}

        then also look for the any columns has the relevant data for the user query

        Here are the column names:
        {column_names}
        The user asked: "{prompt}"

        Based on this query, generate a complete and accurate Python script that:
        1. Uses the existing 'df' DataFrame that's already loaded
        2. Filters or processes the data according to the user's intent
        3. MUST follow this exact format for output:
 a. Then, create the result DataFrame with proper formatting:
              result = filtered_df.copy()
              # Format the result DataFrame here as result  always strictly 
              # Set display options for better formatting
              pd.set_option('display.max_columns', None)
              pd.set_option('display.max_rows', 50)
              pd.set_option('display.width', 1000)
             4. The script must end by printing the summary and storing the formatted DataFrame in 'result' variable

        Respond **only** with valid Python code using `pandas`. Do **not** include explanations or markdown formatting.
"""
        
        # Get Claude's response
        with st.chat_message("assistant"):
            client = get_bedrock_client()
            response = get_claude_response(claude_prompt, client)
            
            if response:
                # Clean up the code if it contains markdown code blocks
                if response.startswith("```"):
                    response = response.strip("`")
                    response = response.replace("python", "")
                    response = response.strip()
                
                # Display the code
                st.code(response, language="python")
                
                try:
                    # Create a new local namespace for execution
                    local_ns = {"df": df, "pd": pd}
                    
                    # Capture print output
                    with OutputCapture() as output:
                        exec(response, {}, local_ns)
                    
                    # Display the captured output
                    output_text = output.getvalue().strip()
                    if output_text:
                        st.markdown(output_text)
                    
                    # Display any returned result as a table
                    # Modify the table display section to handle large datasets
                    if "result" in local_ns:
                        result_df = local_ns["result"]
                        if isinstance(result_df, pd.DataFrame):
                            try:
                                # Limit rows for display if too large
                                display_df = result_df.head(1000) if len(result_df) > 1000 else result_df
                                
                                # Basic styling without excessive formatting
                                styled_df = display_df.style.set_properties(**{
                                    'text-align': 'left',
                                    'padding': '5px'
                                })
                                
                                # Display the table with size limits
                                st.dataframe(
                                    styled_df,
                                    use_container_width=True,
                                    height=min(400, len(display_df) * 35)  # Adjust height based on row count
                                )
                                
                                if len(result_df) > 1000:
                                    st.info(f"Showing first 1000 rows out of {len(result_df)} total rows")
                                    
                            except Exception as e:
                                st.error(f"Error displaying table: {str(e)}")
                                # Fallback to basic display
                                st.dataframe(display_df)
                            # Add download button
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="Download results as CSV",
                                data=csv,
                                file_name="results.csv",
                                mime="text/csv"
                            )
                    
                            # Add to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"```python\n{response}\n```\n\n{output_text}"
                            })
                        else:
                            st.write(result_df)
                    
                    # Add assistant response to chat history with formatted output
                    content = f"```python\n{response}\n```\n\n"
                    if output_text:
                        content += f"Summary:\n```\n{output_text}\n```\n\n"
                    if "result" in local_ns:
                        if isinstance(local_ns["result"], pd.DataFrame):
                            df_info = local_ns["result"]
                            content += f"Table Info:\n- Rows: {len(df_info)}\n- Columns: {len(df_info.columns)}\n"
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": content
                    })
                except Exception as e:
                    st.error(f"Error executing the code: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"```python\n{response}\n```\n\nError: {str(e)}"
                    })
            else:
                st.error("Failed to get a response from the assistant")

else:
    st.info("Please upload an Excel file to start chatting")