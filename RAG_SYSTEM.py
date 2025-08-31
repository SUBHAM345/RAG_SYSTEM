# src/ui_streamlit.py
import streamlit as st
from rag_local import RAGLocal
import base64, os
from PIL import Image, ImageDraw

st.set_page_config(page_title="RAG Local Demo", layout="wide")
st.title("📚 Local RAG Demo (PDF → Search → Answer + Citation)")

@st.cache_resource
def get_rag():
    return RAGLocal()

rag = get_rag()

query = st.text_input("Enter your question about uploaded PDFs:")
k = st.slider("Top-K retrieved chunks", min_value=1, max_value=8, value=4)
if st.button("Ask") and query.strip():
    with st.spinner("Retrieving..."):
        results = rag.retrieve(query, k=k)
    if not results:
        st.warning("No results found.")
    else:
        st.subheader("Retrieved passages")
        for i, r in enumerate(results,1):
            st.markdown(f"**{i}.** `{r['file_name']}`, page **{r['page_no']}**")
            st.write(r['text'][:800])
        st.subheader("Generating answer (local LLM or fallback)")
        with st.spinner("Generating..."):
            ans = rag.generate(query, results, max_tokens=512)
        st.markdown("### Answer")
        st.write(ans)
        st.markdown("### Evidence (snippets)")
for i, r in enumerate(results,1):
    clean_text = r['text'][:200].replace("\n", " ")
    st.markdown(f"- **{r['file_name']}**, p.{r['page_no']} — `{clean_text}`")
# show first page image if available and try to highlight first snippet
first = results[0]
img_path = first.get("image_path")
if img_path and os.path.exists(img_path):
    st.image(img_path, caption=f"{first['file_name']} p.{first['page_no']}")
else:
    st.info("No page image for first result (page may be native PDF). Highlighting is optional.")
