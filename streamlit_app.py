import streamlit as st

from agent.orchestrator import run_agent

st.set_page_config(page_title="Agentic Research Assistant", page_icon="🔎")
st.title("🔎 Agentic Research Assistant")
st.caption(
    "A multi-step agent that plans, searches the web, reads pages, and writes "
    "a cited report — powered by Groq's free-tier open-source models."
)

question = st.text_area("Research question", placeholder="e.g. What are the main approaches to LLM agent evaluation in 2026?")
max_steps = st.slider("Max steps", min_value=2, max_value=15, value=8)

if st.button("Run research", type="primary", disabled=not question.strip()):
    with st.spinner("Researching..."):
        try:
            result = run_agent(question, max_steps=max_steps)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    with st.expander(f"Agent trace ({result['steps']} steps)"):
        for entry in result["trace"]:
            if entry["type"] == "tool_call":
                st.markdown(f"**Tool call:** `{entry['name']}`")
                st.json(entry["args"])
                st.write(entry["result"])
            else:
                st.markdown("**Final answer generated**")

    st.markdown("## Report")
    st.markdown(result["report"])
    st.download_button("Download report.md", result["report"], file_name="report.md")
