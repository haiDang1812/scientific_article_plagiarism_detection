import streamlit as st
import requests

# URL của Backend FastAPI
SERVER_URL = "http://localhost:8000/upload"

st.set_page_config(page_title="Upload Paper System", page_icon="📄")

st.title("Tải lên bài báo (Upload Paper)")
st.markdown("Hệ thống Mock API hỗ trợ upload và polling file PDF.")

with st.form("upload_form"):
    paper_name = st.text_input("Tên Bài báo (Paper Name):", placeholder="Ví dụ: Attention is All You Need")
    
    # Selectbox hiển thị đẹp mắt, trả về giá trị viết thường
    source = st.selectbox(
        "Nguồn (Source):", 
        options=["acl", "arxiv", "cvf"],
        format_func=lambda x: "ACL" if x == "acl" else ("arXiv" if x == "arxiv" else "CVF")
    )
    
    uploaded_file = st.file_uploader("Chọn file PDF:", type=["pdf"])
    
    submit_button = st.form_submit_button("Tải lên (Upload)")
    
    if submit_button:
        if not paper_name.strip():
            st.warning("⚠️ Vui lòng nhập tên bài báo.")
        elif uploaded_file is None:
            st.warning("⚠️ Vui lòng chọn một file PDF.")
        else:
            with st.spinner("Đang tải dữ liệu lên server..."):
                # Chuẩn bị dữ liệu Form Data và File
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {
                    "paper_name": paper_name,
                    "source": source
                }
                
                try:
                    response = requests.post(SERVER_URL, files=files, data=data)
                    if response.status_code == 200:
                        res_json = response.json()
                        st.success(f"✅ {res_json['message']} Tên file lưu trên server: **{res_json['paper']['filename']}**")
                    else:
                        st.error("❌ Có lỗi xảy ra khi tải file lên máy chủ (Backend).")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Không thể kết nối đến máy chủ. Vui lòng đảm bảo bạn đã chạy `python server.py` chưa.")