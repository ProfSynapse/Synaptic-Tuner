import streamlit as st
import json
import os
import glob
import re
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

st.set_page_config(layout="wide", page_title="Dataset Editor")

# --- Session State Management ---
if 'view' not in st.session_state:
    st.session_state.view = 'datasets' # options: 'datasets', 'examples', 'editor'
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
if 'selected_example_index' not in st.session_state:
    st.session_state.selected_example_index = None
if 'data' not in st.session_state:
    st.session_state.data = []

# --- Helper Functions ---

def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def save_data(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def parse_thinking(content):
    if content is None:
        return None
    match = re.search(r'<thinking>\s*({.*})\s*</thinking>', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None

def construct_thinking(thinking_data):
    return f"<thinking>\n{json.dumps(thinking_data, indent=2)}\n</thinking>"

def recursive_json_editor(data, prefix, key_prefix):
    new_data = data.copy()
    for key, value in data.items():
        field_key = f"{key_prefix}_{key}"
        label = f"{prefix}{key}"
        
        if isinstance(value, dict):
            st.markdown(f"**{label}**")
            new_data[key] = recursive_json_editor(value, f"{label}.", field_key)
        elif isinstance(value, list):
            val_str = json.dumps(value)
            new_val_str = st.text_area(label, value=val_str, key=field_key, height=70)
            try:
                new_data[key] = json.loads(new_val_str)
            except:
                st.error(f"Invalid JSON for {label}")
        else:
            if isinstance(value, str):
                height = 150 if len(value) > 50 else 70
                new_data[key] = st.text_area(label, value=value, key=field_key, height=height)
            elif isinstance(value, bool):
                new_data[key] = st.checkbox(label, value=value, key=field_key)
            elif isinstance(value, (int, float)):
                new_data[key] = st.number_input(label, value=value, key=field_key)
            elif value is None:
                 new_data[key] = st.text_input(label, value="null", key=field_key)
                 if new_data[key] == "null":
                     new_data[key] = None
    return new_data

def get_dataset_files(category):
    root_dir = "Datasets"
    if category == "Behaviors":
        search_path = os.path.join(root_dir, "behavior_datasets", "**", "*.jsonl")
    elif category == "Toolsets":
        search_path = os.path.join(root_dir, "tools_datasets", "**", "*.jsonl")
    else:
        return []
    
    files = glob.glob(search_path, recursive=True)
    dataset_info = []
    
    for f in files:
        rel_path = os.path.relpath(f, root_dir)
        size_kb = os.path.getsize(f) / 1024
        try:
            with open(f, 'r', encoding='utf-8') as file_handle:
                line_count = sum(1 for line in file_handle if line.strip())
        except:
            line_count = 0
            
        # Extract dataset name from the parent folder
        parent_dir = os.path.dirname(f)
        dataset_name = os.path.basename(parent_dir)
            
        dataset_info.append({
            "Dataset Name": dataset_name,
            "File Name": os.path.basename(f),
            "Name": os.path.basename(f), # Keep for compatibility if needed
            "Path": rel_path,
            "Full Path": f,
            "Size (KB)": f"{size_kb:.2f}",
            "Examples": line_count
        })
        
    return pd.DataFrame(dataset_info)

def get_examples_summary(data):
    summary = []
    for i, entry in enumerate(data):
        conversations = entry.get("conversations", [])
        
        # Find first user message
        user_msg = "No user message"
        for msg in conversations:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_msg = (content[:75] + '...') if len(content) > 75 else content
                break
        
        # Count turns
        turns = len(conversations)
        
        # Check for tools
        has_tools = any("tool_calls" in msg for msg in conversations)
        
        summary.append({
            "Index": i,
            "First User Query": user_msg,
            "Has Tools": has_tools
        })
    return pd.DataFrame(summary)

# --- Navigation Functions ---
def go_to_datasets():
    st.session_state.view = 'datasets'
    st.session_state.selected_dataset = None
    st.session_state.data = []

def go_to_examples(dataset_path):
    st.session_state.selected_dataset = dataset_path
    st.session_state.data = load_data(dataset_path)
    st.session_state.view = 'examples'

def go_to_editor(index):
    st.session_state.selected_example_index = index
    st.session_state.view = 'editor'

# --- Views ---

def view_datasets():
    st.title("Dataset Library")
    
    category = st.sidebar.radio("Category", ["Behaviors", "Toolsets"])
    df = get_dataset_files(category)
    
    if not df.empty:
        st.markdown("### Available Datasets")
        
        # Group by Dataset Name
        grouped = df.groupby("Dataset Name")
        dataset_names = list(grouped.groups.keys())
        
        # Grid layout for cards
        cols = st.columns(3)
        for i, dataset_name in enumerate(dataset_names):
            group_df = grouped.get_group(dataset_name)
            # Sort versions descending by filename (simple heuristic for "latest")
            group_df = group_df.sort_values("File Name", ascending=False)
            
            col = cols[i % 3]
            
            with col:
                with st.container(border=True):
                    st.subheader(dataset_name)
                    
                    # Version selection
                    versions = dict(zip(group_df["File Name"], group_df["Full Path"]))
                    
                    # Default to most recent or first? Just first for now.
                    selected_version_name = st.selectbox(
                        "Select Version", 
                        options=list(versions.keys()),
                        key=f"ver_{i}"
                    )
                    
                    selected_full_path = versions[selected_version_name]
                    
                    # Get stats for selected version
                    selected_row = group_df[group_df["File Name"] == selected_version_name].iloc[0]
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Examples", selected_row['Examples'])
                    with c2:
                        st.metric("Size (KB)", selected_row['Size (KB)'])
                    
                    if st.button("Open Dataset", key=f"btn_{i}", use_container_width=True):
                        go_to_examples(selected_full_path)
                        st.rerun()
    else:
        st.warning("No datasets found.")

def view_examples():
    st.title("Dataset Explorer")
    st.caption(f"Editing: {os.path.basename(st.session_state.selected_dataset)}")
    
    if st.button("← Back to Datasets"):
        go_to_datasets()
        st.rerun()
        
    data = st.session_state.data
    if not data:
        st.warning("Dataset is empty.")
        return

    # Search/Filter
    search_term = st.text_input("Search examples...", placeholder="Type to filter by content...")
    
    df_summary = get_examples_summary(data)
    
    if search_term:
        # Filter the dataframe based on search term (searching in the summary for speed, 
        # but ideally we search the full data. For now, let's search the summary + raw json dump)
        # Actually, let's filter the indices first
        filtered_indices = []
        for i, entry in enumerate(data):
            if search_term.lower() in json.dumps(entry).lower():
                filtered_indices.append(i)
        df_display = df_summary[df_summary["Index"].isin(filtered_indices)]
    else:
        df_display = df_summary

    st.write(f"Showing {len(df_display)} examples")

    # Configure AgGrid
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_selection(selection_mode='single', use_checkbox=False)
    gb.configure_column("Index", hide=True)
    gb.configure_column("First User Query", wrapText=True, autoHeight=True, flex=1)
    gb.configure_column("Has Tools", width=100)
    gridOptions = gb.build()

    grid_response = AgGrid(
        df_display,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=600,
        theme='streamlit'
    )

    selected = grid_response['selected_rows']
    
    # Handle case where selected_rows is a DataFrame (newer st_aggrid versions)
    if isinstance(selected, pd.DataFrame):
        if not selected.empty:
            real_index = selected.iloc[0]["Index"]
            go_to_editor(int(real_index))
            st.rerun()
    # Handle case where selected_rows is a list of dicts
    elif selected is not None and len(selected) > 0:
        first_selected = selected[0]
        if isinstance(first_selected, dict):
            real_index = first_selected.get("Index")
            if real_index is not None:
                go_to_editor(int(real_index))
                st.rerun()

def view_editor():
    idx = st.session_state.selected_example_index
    st.title(f"Editor: Example #{idx}")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back"):
            st.session_state.view = 'examples'
            st.rerun()
    
    current_entry = st.session_state.data[idx]
    
    with st.form(key=f"edit_form_{idx}"):
        # Conversations
        conversations = current_entry.get("conversations", [])
        updated_conversations = []
        
        st.subheader("Conversations")
        for i, msg in enumerate(conversations):
            role = msg.get("role", "unknown")
            content = msg.get("content")
            if content is None:
                content = ""
            
            st.markdown(f"**Message {i+1}: {role.upper()}**")
            
            msg_copy = msg.copy()
            
            # Check for Thinking Block
            thinking_data = parse_thinking(content)
            
            if role == "assistant" and thinking_data:
                st.info("Detected Thinking Block - Editing Fields")
                
                # Goal
                new_goal = st.text_input(f"Goal ({i})", value=thinking_data.get("goal", ""))
                
                # Memory
                new_memory = st.text_area(f"Memory ({i})", value=thinking_data.get("memory", ""), height=100)
                
                # Requirements (List)
                reqs = thinking_data.get("requirements", [])
                reqs_str = "\n".join(reqs) if isinstance(reqs, list) else str(reqs)
                new_reqs_str = st.text_area(f"Requirements (one per line) ({i})", value=reqs_str, height=100)
                new_reqs = [line.strip() for line in new_reqs_str.split('\n') if line.strip()]
                
                # Assessment (Object)
                st.markdown("Assessment")
                col1, col2 = st.columns(2)
                assessment = thinking_data.get("assessment", {})
                with col1:
                    is_complex = st.checkbox(f"Complex ({i})", value=assessment.get("complex", False))
                with col2:
                    is_risky = st.checkbox(f"Risky ({i})", value=assessment.get("risky", False))
                new_assessment = {"complex": is_complex, "risky": is_risky}
                
                # Confidence
                new_confidence = st.number_input(f"Confidence ({i})", value=float(thinking_data.get("confidence", 0.0)), min_value=0.0, max_value=1.0, step=0.01)
                
                # Plan (List)
                plan = thinking_data.get("plan", [])
                plan_str = "\n".join(plan) if isinstance(plan, list) else str(plan)
                new_plan_str = st.text_area(f"Plan (one per line) ({i})", value=plan_str, height=100)
                new_plan = [line.strip() for line in new_plan_str.split('\n') if line.strip()]
                
                # Reconstruct Thinking Data
                updated_thinking = {
                    "goal": new_goal,
                    "memory": new_memory,
                    "requirements": new_reqs,
                    "assessment": new_assessment,
                    "confidence": new_confidence,
                    "plan": new_plan
                }
                
                # Preserve other keys in thinking if any
                for k, v in thinking_data.items():
                    if k not in updated_thinking:
                        updated_thinking[k] = v
                        
                msg_copy["content"] = construct_thinking(updated_thinking)
                
            else:
                # Standard Content Editing
                new_content = st.text_area(f"Content {i}", value=content, height=150, key=f"content_{idx}_{i}")
                msg_copy["content"] = new_content
            
            # Handle Tool Calls
            if "tool_calls" in msg:
                st.markdown("---")
                st.markdown(f"**Tool Calls ({i})**")
                
                tool_calls = msg.get("tool_calls", [])
                updated_tool_calls = []
                
                for j, tool_call in enumerate(tool_calls):
                    st.markdown(f"**Tool Call {j+1}**")
                    
                    tc_copy = tool_call.copy()
                    
                    # ID and Type
                    col1, col2 = st.columns(2)
                    with col1:
                        tc_copy["id"] = st.text_input(f"ID ({i}-{j})", value=tool_call.get("id", ""), key=f"tc_id_{idx}_{i}_{j}")
                    with col2:
                        tc_copy["type"] = st.text_input(f"Type ({i}-{j})", value=tool_call.get("type", "function"), key=f"tc_type_{idx}_{i}_{j}")
                    
                    # Function
                    func = tool_call.get("function", {})
                    func_copy = func.copy()
                    
                    func_copy["name"] = st.text_input(f"Function Name ({i}-{j})", value=func.get("name", ""), key=f"tc_name_{idx}_{i}_{j}")
                    
                    # Arguments
                    args_str = func.get("arguments", "{}")
                    try:
                        args_json = json.loads(args_str)
                        st.markdown(f"**Arguments ({i}-{j})**")
                        
                        # Recursive Editor for Arguments
                        updated_args_json = recursive_json_editor(args_json, "", f"tc_args_{idx}_{i}_{j}")
                        func_copy["arguments"] = json.dumps(updated_args_json)
                        
                    except json.JSONDecodeError:
                        st.warning(f"Arguments are not valid JSON, editing as string.")
                        func_copy["arguments"] = st.text_area(f"Arguments (Raw) ({i}-{j})", value=args_str, key=f"tc_args_raw_{idx}_{i}_{j}")
                    
                    tc_copy["function"] = func_copy
                    updated_tool_calls.append(tc_copy)
                
                msg_copy["tool_calls"] = updated_tool_calls
            
            updated_conversations.append(msg_copy)

        # Other fields (like label, etc.)
        st.subheader("Other Fields")
        other_fields = {k: v for k, v in current_entry.items() if k != "conversations"}
        
        # Use recursive editor for other fields
        new_other_fields = recursive_json_editor(other_fields, "", f"other_fields_{idx}")
        
        submit_button = st.form_submit_button(label="Save Changes")
        
        if submit_button:
            try:
                # Construct new entry
                new_entry = new_other_fields
                new_entry["conversations"] = updated_conversations
                
                # Update session state
                st.session_state.data[idx] = new_entry
                
                # Save to file
                save_data(st.session_state.selected_dataset, st.session_state.data)
                st.success("Saved successfully!")
                
            except Exception as e:
                st.error(f"Error saving: {e}")

    # Raw JSON View (Read-only or for copying)
    with st.expander("View Raw JSON"):
        st.code(json.dumps(current_entry, indent=2), language="json")

# --- Main Router ---

if st.session_state.view == 'datasets':
    view_datasets()
elif st.session_state.view == 'examples':
    view_examples()
elif st.session_state.view == 'editor':
    view_editor()
