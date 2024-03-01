import streamlit as st

def main() -> int:
    home, content, students = st.tabs(["Home", "Content", "Stundents"])
    with home:
        st.markdown("""
                    # Home
                    ---

                    This is the home of the app
                    """)
    with content:
        res = st.selectbox(label="Select option:", options=["Uno", "Dos", "Tres"])
        st.write(res)

    with students:
        with st.expander("See this cool hidden feature"):
            st.image("https://static.streamlit.io/examples/dice.jpg")
    return 0

if __name__ == "__main__":
    exit(main())
