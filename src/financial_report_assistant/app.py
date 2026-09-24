from pathlib import Path

import streamlit as st

from generate_answer import (
    call_ollama,
    format_context,
    validate_answer,
)
from hybrid_search import hybrid_search


st.set_page_config(
    page_title="Financial Report Assistant",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Financial Report Assistant")
st.caption(
    "Ask questions about an annual report and inspect the supporting evidence."
)

with st.sidebar:
    st.header("⚙️ Settings")

    index_directory = st.text_input(
        "FAISS index directory",
        value="./output/alphabet_2025_index",
    )

    ollama_model = st.text_input(
        "Ollama model",
        value="qwen2:7b",
        help="Use the exact model name displayed by 'ollama list'.",
    )

    ollama_url = st.text_input(
        "Ollama server",
        value="http://localhost:11434",
    )

    top_k = st.slider(
        "Evidence passages",
        min_value=1,
        max_value=10,
        value=5,
    )

    candidate_k = st.slider(
        "Retrieval candidates",
        min_value=top_k,
        max_value=50,
        value=max(20, top_k),
    )

with st.form("question_form"):
    question = st.text_area(
        "Ask a question about the report",
        placeholder="What was Alphabet's total revenue in 2025?",
        height=100,
    )

    submitted = st.form_submit_button(
        "🔎 Analyze report",
        type="primary",
        use_container_width=True,
    )

if submitted:
    cleaned_question = question.strip()
    index_path = Path(index_directory)

    if not cleaned_question:
        st.warning("Enter a question before submitting.")

    elif not index_path.exists():
        st.error(
            f"The index directory does not exist: {index_path}"
        )

    elif not (index_path / "index.faiss").exists():
        st.error(
            f"No index.faiss file was found inside: {index_path}"
        )

    elif not ollama_model.strip():
        st.warning("Enter an Ollama model name.")

    else:
        try:
            with st.status(
                "Analyzing the financial report...",
                expanded=True,
            ) as status:
                st.write("🔍 Retrieving relevant passages...")

                results = hybrid_search(
                    index_dir=index_path,
                    query=cleaned_question,
                    top_k=top_k,
                    candidate_k=candidate_k,
                )

                if not results:
                    raise RuntimeError(
                        "No relevant passages were retrieved."
                    )

                context, citations = format_context(results)

                st.write(
                    f"🧠 Generating an answer with {ollama_model}..."
                )

                answer = call_ollama(
                    base_url=ollama_url,
                    model=ollama_model,
                    question=cleaned_question,
                    context=context,
                )

                st.write("✅ Validating citations...")

                validate_answer(
                    answer=answer,
                    citations=citations,
                )

                status.update(
                    label="Analysis complete",
                    state="complete",
                    expanded=False,
                )

            st.subheader("💬 Answer")
            st.write(answer)

            st.subheader("📚 Retrieved evidence")

            for source_number, result in enumerate(
                results,
                start=1,
            ):
                chunk = result["chunk"]

                pages = ", ".join(
                    str(page)
                    for page in chunk.get("page_numbers", [])
                )
                pages = pages or "unknown"

                section = " > ".join(
                    chunk.get("headings", [])
                )
                section = section or "Unlabelled section"

                chunk_id = chunk.get(
                    "chunk_id",
                    "unknown",
                )

                title = (
                    f"[S{source_number}] "
                    f"Page {pages} — {section}"
                )

                with st.expander(title):
                    st.caption(f"Chunk: {chunk_id}")
                    st.caption(
                        f"RRF score: {result['rrf_score']:.6f}"
                    )
                    st.write(chunk.get("text", ""))

        except Exception as error:
            st.error("The analysis could not be completed.")
            st.exception(error)

st.divider()

st.caption(
    "Educational project only. Verify financial information against "
    "the original cited report."
)