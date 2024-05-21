import streamlit as st
from st_copy_to_clipboard import st_copy_to_clipboard
from brain import *  
from PIL import Image
from langchain.docstore.document import Document
# import textsplit
fav = Image.open("indFav.PNG")
st.set_page_config(page_title="INDChatBot", page_icon = fav, layout="wide")

img = Image.open("indiumLogo.PNG")
st.image(img, width=200)
st.title("Question Answering App")

model_options = ["mistral", "llama3"]
selected_model = st.selectbox("Select the Model: ", model_options)
temp = st.slider('Select a temperture value', 0.0, 1.0, 0.5, 0.1)

if "model" not in st.session_state:
    st.session_state["model"] = selected_model
if "temperature" not in st.session_state:
    st.session_state["temperature"] = temp



def create_or_update_vector_db(chunks):
    """Creates or updates the vector database based on the provided data."""
    if st.session_state["vector_db"] is None:
        new_vector_db = vectorDataBase(chunks)
        st.session_state["vector_db"] = new_vector_db
    else:
        del st.session_state["vector_db"]  # Clear existing database
        new_vector_db = vectorDataBase(chunks)
        st.session_state["vector_db"] = new_vector_db
    # return new_vector_db

# def create_or_update_vector_db(chunks, selection):
#     """Creates or updates the vector database based on the provided data and selection."""
#     if st.session_state["vector_db"] is None or st.session_state.get("selection") != selection:
#         # Clear existing database if selection has changed
#         if "vector_db" in st.session_state:
#             del st.session_state["vector_db"]

#         new_vector_db = vectorDataBase(chunks)
#         st.session_state["vector_db"] = new_vector_db
#         st.session_state["selection"] = selection
#     else:
#         # No need to recreate the vector database
#         pass

#     return st.session_state["vector_db"]

def get_data_based_on_selection():
    selection = st.session_state.get("selection")
    if selection == "File upload":
        file = st.file_uploader("Upload here...")
        if file is not None:
            if "setup2" in st.session_state:
                del st.session_state["setup2"]
            if "setup3" in st.session_state:
                del st.session_state["setup3"]
            if "setup1" not in st.session_state:
                print(f"---------------{selection} has been selected!!--------------\n")
                
                pdf_path = save_file(file)
                pages = upload_pdf(pdf_path)
                chunks = split_data_into_chunks(pages)
                llm, prompt = initialize_the_model(selected_model, temp)
                st.session_state["llm"] = llm
                st.session_state["prompt"] = prompt   
                st.session_state["setup1"] = True
                return chunks
    
    elif selection == "Content text":
        
        content = st.text_area("Enter content:", height=200,max_chars=5000)
        if st.button("Submit"):
            if "setup1" in st.session_state:
                del st.session_state["setup1"]
            if "setup3" in st.session_state:
                del st.session_state["setup3"]
            if "setup2" not in st.session_state:
                print(f"---------------{selection} has been selected!!--------------\n")
                
                # document = Document(content)
                # chunks = split_data_into_chunks([document])
                document = Document(page_content=content)  # Create a Document object with the content
                chunks = split_data_into_chunks([document])
                llm, prompt = initialize_the_model(selected_model, temp)
                st.session_state["llm"] = llm
                st.session_state["prompt"] = prompt 
                st.session_state["setup2"] = True
                return chunks
    
    elif selection == "URL":
        
        url = st.text_input("Enter URL's:")
        if url:
            if "setup1" in st.session_state:
                del st.session_state["setup1"]
            if "setup2" in st.session_state:
                del st.session_state["setup2"]
            if "setup3" not in st.session_state:
                print(f"---------------{selection} has been selected!!--------------\n")
                
                data = contentLoader(url)
                chunks = split_data_into_chunks(data)
                llm, prompt = initialize_the_model(selected_model, temp)
                st.session_state["llm"] = llm
                st.session_state["prompt"] = prompt 
                st.session_state["setup3"] = True
                return chunks
    
    return None

if "selection" not in st.session_state:
  st.session_state["selection"] = None  # Initialize selection state

if "vector_db" not in st.session_state:
  st.session_state["vector_db"] = None  # Initialize vector_db state

selection = st.radio("Select input method:", ("File upload", "Content text", "URL"))
st.session_state["selection"] = selection
data = get_data_based_on_selection()

if data is not None:
    create_or_update_vector_db(data)
       



question_type_options = ("Informational Questions","Clarification Questions","Probing Questions","Closed Questions",
"Open-Ended Questions","Leading Questions","Reflective Questions","Hypothetical Questions"
,"Descriptive Questions","Diagnostic Questions","Predictive Questions","Prescriptive Questions","Zero-Shot Questions"
,"One-Shot Questions","Few-Shot Questions","Chain-of-Thought Questions","Opinion-Based Questions","Instructional Questions"
,"Creative Questions","Problem-Solving Questions","Conversational Questions"
,"Role-Playing Questions","Contrastive Questions","Vague Questions","Troubleshooting Questions","Recommendation Questions",
"Location - Based Questions","Computational Questions","Benchmarking Questions","Task Based Questions",
"Multimodal and Multimedia Questions","Programming Questions","Language assistance", 
"Affirmative and Negative Questions")
selected_Qtype = st.sidebar.selectbox("Select question type:", question_type_options)

num_questions = st.sidebar.number_input("No.of Questions: ",min_value=None, max_value=None, step=1, format='%d')

difficulty = st.sidebar.radio("Difficulty level:", ("Easy","Medium","Hard"))

if st.session_state["vector_db"] is not None:   
    if st.sidebar.button("Generate Questions"):
        vector_db = st.session_state.get("vector_db")
        if selected_model != st.session_state["model"] or temp != st.session_state["temperature"]:
            # Update model and prompt based on selected model
            llm, prompt = initialize_the_model(selected_model, temp)
            st.session_state["llm"] = llm
            st.session_state["prompt"] = prompt
        retriever = retriver_of_Data(vector_db.as_retriever(), st.session_state.get("llm"))
        qa_chain = retrieval_qa_chain(retriever, st.session_state.get("llm"), st.session_state.get("prompt"))

        # qa_chain = retrieval_qa_chain(vector_db.as_retriever(), llm, prompt)
        query = f"""Please generate {num_questions} {selected_Qtype} with answers. The difficulty level should be {difficulty}. based on the content provided?"""
        result = generate_answer(query, qa_chain)
        st.success(result)
    

    query = st.text_input("Enter Your Question : ")

    if st.button("Enter"):
        vector_db = st.session_state.get("vector_db")  # Safe retrieval

        # Initialize based on model, but reuse vector_db if it exists
        if selected_model != st.session_state["model"] or temp != st.session_state["temperature"]:
            # Update model and prompt based on selected model
            llm, prompt = initialize_the_model(selected_model, temp)
            st.session_state["llm"] = llm
            st.session_state["prompt"] = prompt
        
        
        retriever = retriver_of_Data(vector_db.as_retriever(), st.session_state.get("llm"))
        qa_chain = retrieval_qa_chain(retriever, st.session_state.get("llm"), st.session_state.get("prompt"))

        result = generate_answer(query, qa_chain)
        st.success(result)
        st_copy_to_clipboard(result)



st.sidebar.write("Powered by Indium Software")
