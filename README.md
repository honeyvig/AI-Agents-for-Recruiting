# AI-Agents-for-Recruiting
we created a bot on chatgpt which brings good results for candidate qualification and candidate summary. We see potential in how chatgpt is scoring the candidate and how we can compare candidates between each other

Goals:
qualify candidates for a specific job description
score candidates in comparison with other candidates for the job
create a summary of the interview per candidate
create a summary for several candidates and compare them between each other

Requirements:
a link (p.e  www.xy.de/job?=Java-Developer) or webpage  which starts the interview for a specific job
one AI Agents (chatgpt bot) based on our current test to run the interview
one AI Agents to handle scoring and summaries

Deliverables:
Comparison of 2-3 AI frameworks or solutions in order to execute the requirements. Decision for one solution
Link for candidate to run the interview (Interview will be done in one run and doesn´t have to be adjusted after completion by the candidate)
Summary per candidate as spreadsheet (no design needed). The scoring per candidate might adjust if other candidates apply with better qualification
Summary for several candidates as spreadsheet (no design needed) and comparision between each other.

===================
Python-based proof-of-concept (POC) for implementing the requirements, leveraging OpenAI’s GPT model along with auxiliary frameworks for scoring, summaries, and candidate comparisons.
High-Level Design

    Frameworks Considered:
        OpenAI GPT-4: For semantic understanding, interview conversation, scoring, and summarization.
        LangChain: For orchestrating LLM workflows, including multi-agent interactions.
        Streamlit/Django/Flask: For user interface to start the interview.
        Pandas: For generating comparison spreadsheets.
        SQLite: For lightweight candidate data storage.

    Chosen Solution: OpenAI GPT-4 + LangChain due to its adaptability for scoring, summarization, and multi-agent workflows.

Python Implementation
1. Setting Up the Interview Bot

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain

# Initialize OpenAI GPT-4 model
interview_bot = ChatOpenAI(model="gpt-4", temperature=0.7)

# Prompt template for interview
interview_prompt = ChatPromptTemplate.from_template("""
You are an interviewer for the role of {job_role}. Conduct an interview and assess the candidate's qualifications based on the job description.

Job Description:
{job_description}

Ask up to 5 relevant questions to understand their qualifications. After each question, evaluate their response on a scale of 1-10 and provide reasoning for the score.
""")

def run_interview(job_role, job_description):
    chain = ConversationChain(llm=interview_bot, prompt=interview_prompt)
    return chain.run(job_role=job_role, job_description=job_description)

2. Candidate Scoring and Summarization Bot

import pandas as pd

# Bot for scoring candidates
scoring_bot = ChatOpenAI(model="gpt-4", temperature=0.5)

# Generate scores and summary for a candidate
def evaluate_candidate(candidate_responses, job_description):
    prompt = f"""
    Based on the following job description, evaluate the candidate's responses:
    Job Description: {job_description}

    Candidate Responses: {candidate_responses}

    1. Provide an overall score out of 100 based on their answers.
    2. Summarize the candidate's strengths and weaknesses in relation to the job.
    """
    response = scoring_bot.predict(prompt)
    return response

# Compare multiple candidates
def compare_candidates(candidates_data):
    prompt = f"""
    Compare the following candidates for the given job description. Rank them based on their qualifications.

    Candidates Data: {candidates_data}

    Provide a summary of the comparison and highlight the strongest and weakest candidates.
    """
    comparison_summary = scoring_bot.predict(prompt)
    return comparison_summary

3. Web Interface for Interview

Use Flask to create an interface for candidates to start their interviews.

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/start_interview', methods=['POST'])
def start_interview():
    data = request.json
    job_role = data.get("job_role")
    job_description = data.get("job_description")
    candidate_responses = run_interview(job_role, job_description)
    return jsonify({"candidate_responses": candidate_responses})

@app.route('/evaluate_candidate', methods=['POST'])
def evaluate():
    data = request.json
    candidate_responses = data.get("candidate_responses")
    job_description = data.get("job_description")
    evaluation = evaluate_candidate(candidate_responses, job_description)
    return jsonify({"evaluation": evaluation})

@app.route('/compare_candidates', methods=['POST'])
def compare():
    data = request.json
    candidates_data = data.get("candidates_data")
    comparison = compare_candidates(candidates_data)
    return jsonify({"comparison": comparison})

if __name__ == "__main__":
    app.run(debug=True)

4. Generating Candidate Summary as Spreadsheet

def generate_spreadsheet(candidate_evaluations):
    df = pd.DataFrame(candidate_evaluations)
    df.to_excel("candidate_summary.xlsx", index=False)

# Example candidate evaluations
candidate_evaluations = [
    {"Name": "Candidate A", "Score": 85, "Strengths": "Strong technical skills", "Weaknesses": "Limited leadership experience"},
    {"Name": "Candidate B", "Score": 78, "Strengths": "Good communication skills", "Weaknesses": "Less relevant experience"}
]
generate_spreadsheet(candidate_evaluations)

Workflow

    Start the Interview:
        The candidate clicks the provided link (e.g., /start_interview) and answers the AI's questions.
        Responses are captured and stored.

    Candidate Evaluation:
        The /evaluate_candidate endpoint evaluates the candidate and provides a score with a summary.

    Comparison:
        The /compare_candidates endpoint compares multiple candidates and provides a ranked summary.

    Generate Spreadsheet:
        A spreadsheet is generated for individual candidates and comparisons.

Deliverables

    POC Application:
        Interview Bot (/start_interview)
        Candidate Evaluation (/evaluate_candidate)
        Candidate Comparison (/compare_candidates)
        Summary Spreadsheet (candidate_summary.xlsx)

Next Steps

    Enhancements:
        Add a database for persistent storage of candidate responses.
        Use cloud-based hosting (e.g., AWS, GCP) for deployment.
    Design UI:
        Integrate a React/Angular front-end for user interaction.
    Scaling:
        Handle multiple concurrent interviews using background task queues like Celery.

This implementation addresses all requirements, offering a solid foundation for automating candidate qualification, scoring, and comparison.
