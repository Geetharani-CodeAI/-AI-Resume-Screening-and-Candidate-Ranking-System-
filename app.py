import streamlit as st
import pdfplumber
import numpy as np
import faiss
import google.generativeai as genai

from sentence_transformers import SentenceTransformer

# -----------------------------------
# GEMINI API CONFIG
# -----------------------------------

genai.configure(api_key="AIzaSyBbxKs5ufU6hxe-CY_m6u23Azx2JoRIoFw")

gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")


# STREAMLIT TITLE


st.title("AI Resume Screening System")


# JOB DESCRIPTION INPUT


job_description = st.text_area("Enter Job Description")


# FILE UPLOAD

uploaded_files = st.file_uploader("Upload Resumes", type=["pdf"],
    accept_multiple_files=True
)


# LOAD EMBEDDING MODEL


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# PROCESS RESUMES


if uploaded_files and job_description:

    all_resumes = []

    # Extract Resume Text
    for file in uploaded_files:

        resume_text = ""

        with pdfplumber.open(file) as pdf:

            for page in pdf.pages:

                text = page.extract_text()

                if text:
                    resume_text += text

        # Store Resume Data
        all_resumes.append({

            "filename": file.name,
            "text": resume_text

        })

    st.success("Files Uploaded Successfully")

    st.write( "Total Resumes Uploaded:", len(all_resumes))

   
    # JOB DESCRIPTION EMBEDDING
  

    jd_embedding = embedding_model.encode([job_description]).astype("float32")

    # Normalize JD Embedding
    faiss.normalize_L2(jd_embedding)

    
    # RESUME EMBEDDINGS


    resume_embeddings = []

    for resume in all_resumes:

        embedding = embedding_model.encode([resume["text"]]).astype("float32")

        resume_embeddings.append(embedding[0])

    # Convert to numpy
    resume_embeddings = np.array(resume_embeddings).astype("float32")

    # Normalize Resume Embeddings
    faiss.normalize_L2(resume_embeddings)

    
    # CREATE FAISS INDEX
    

    dimension = resume_embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # Add Resume Embeddings
    index.add(resume_embeddings)

    
    # SEARCH TOP CANDIDATES
   

    k = min(3, len(all_resumes))

    scores, indices = index.search(jd_embedding, k)

   
    # STORE RANKED RESULTS
   

    ranked_results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        ranked_results.append({

            "filename":
            all_resumes[idx]["filename"],

            "text":
            all_resumes[idx]["text"],

            "score":
            float(score)

        })

    
    # DISPLAY RANKINGS
    
    st.subheader("Top Candidates")

    for rank, result in enumerate(ranked_results, start=1):

        st.write(f"Rank {rank}")

        st.write(f"Resume: {result['filename']}")

        st.write(f"Match Score: {round(result['score'] * 100, 2)}%")

        st.write("---")

      
    # AI Candidate Analysis
  
    
    st.subheader(
        "AI Candidate Analysis"
    )
    
    for candidate in ranked_results:
    
        filename = candidate["filename"]
    
        resume_text = candidate["text"]
    
        st.markdown(
            f"## 📄 {filename}"
        )
    
        prompt = f"""
    
        You are an AI Hiring assistant.
    
        Analyse this candidate against
        the job description.
    
        Give output in this exact format:
    
        1. Candidate Summary:
        (2 lines with bullet points only)
    
        2. Strengths:
        (3 lines with bullet points only)
    
        3. Missing Skills: 
        (3 lines with bullet points only)
    
        4. Hiring Recommendation:
        (1 lines with bullet points only)
    
        Job Description:
        {job_description}
    
        Resume:
        {resume_text}
    
        """
    
        response = gemini_model.generate_content(
    
            prompt,
    
            generation_config={
    
                "temperature": 0.3,
                "max_output_tokens": 300
    
            }
    
        )
    
        st.markdown(
            response.text
        )
    
        st.write("---")
       
    # SKILL KEYWORDS
   

    skill_keywords = [

        "Python",
        "Machine Learning",
        "NLP",
        "Deep Learning",
        "SQL"

    ]

   
    # QUESTION BANK
    

    question_bank = {

        "Python": [

            "Explain list vs tuple.",
            "What are decorators in Python?",
            "Explain generators."

        ],

        "Machine Learning": [

            "What is overfitting?",
            "Explain bias vs variance.",
            "Difference between supervised and unsupervised learning?"

        ],

        "NLP": [

            "What is TF-IDF?",
            "Explain tokenization.",
            "What are embeddings?"

        ],

        "Deep Learning": [

            "Explain CNN architecture.",
            "What is backpropagation?",
            "Difference between CNN and RNN?"

        ],

        "SQL": [

            "Difference between WHERE and HAVING?",
            "Explain joins.",
            "What is normalization?"

        ]
    }

   
    # GENERATE QUESTIONS
 

    st.subheader("AI Interview Questions")

    for candidate in ranked_results:

        filename = candidate["filename"]

        resume_text = candidate["text"]

        st.markdown(f"## 👤 Candidate: {filename}")

       
        # SKILL EXTRACTION
      

        skills = []

        for skill in skill_keywords:

            if skill.lower() in resume_text.lower():

                skills.append(skill)

        st.markdown(f"**Skills:** {', '.join(skills)}")

     
        # RETRIEVE QUESTIONS
       

        retrieved_questions = []

        for skill in skills:

            if skill in question_bank:

                retrieved_questions.extend(question_bank[skill])

        st.markdown("### 📘 Retrieved Questions (RAG)")

        for q in retrieved_questions:

            st.write("-", q)

       
        # GENERATIVE AI QUESTIONS
     

        prompt = f"""

        You are an AI Technical Interviewer.

        Candidate Skills:
        {skills}

        Reference Interview Questions:
        {retrieved_questions}

        Generate interview questions in CLEAN FORMAT in a bullet numbered format.

        Format strictly like this:

        1. 5 technical interview questions

        2. 2 scenario-based questions

        3. 1 project-based question

        Keep questions personalized
        to the candidate profile.
        
        Keep questions:
        - short
        - professional
        - personalized
        - easy to read
        
        Do not give explanations.
        Do not give introductions.
        Do not give paragraphs.

        """

        response = gemini_model.generate_content(

            prompt,

            generation_config={
                "temperature": 0.5,
                "max_output_tokens": 400

            }

        )

        st.subheader("AI Generated Questions based on the skillset")

        st.markdown(response.text)

        st.write("---")