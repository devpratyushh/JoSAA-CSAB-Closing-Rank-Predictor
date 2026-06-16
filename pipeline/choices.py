import streamlit as st
import json

def _get_ls():
    if "localS" not in st.session_state:
        from streamlit_local_storage import LocalStorage
        st.session_state["localS"] = LocalStorage()
    return st.session_state["localS"]

def get_choices() -> set[tuple[str, str]]:
    try:
        ls = _get_ls()
        data = ls.getItem("user_choices")
        if data:
            if isinstance(data, str):
                choices_list = json.loads(data)
            else:
                choices_list = data
            return {(c[0], c[1]) for c in choices_list}
    except Exception as e:
        print("Error reading choices:", e)
    return set()

def toggle_choice(institute: str, program: str, is_starred: bool):
    choices = get_choices()
    choice = (institute, program)
    if is_starred:
        choices.add(choice)
    else:
        choices.discard(choice)
        
    try:
        ls = _get_ls()
        ls.setItem("user_choices", json.dumps(list(choices)))
    except Exception as e:
        print("Error saving choices:", e)
