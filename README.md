# QueryBot 🤖
Powered by Langchain 🦜🔗 + OpenAI + Streamlit

QueryBot is a chatbot designed for the e-learning company CodeBasics (website: codebasics.io). 
CodeBasics offers data-related courses and bootcamps and serves thousands of learners who utilize Discord servers and emails to ask questions. 
This system provides a Streamlit-based user interface for students to ask questions and receive answers.

![image](https://github.com/ahmadhassan-2609/QueryBot/assets/163967175/b5fcb2bf-da73-4416-86f1-0dd7a52795b8)

## Project Highlights
* LLM based question and answer system to reduce the workload of human staff
* RAG implementation to retrieve data from a CSV file of FAQs used by the company staff to provide reponses
* Memory feature to be able to remember context of past conversations
* Use of agent in order for the model to reason which actions to take and in which order

## Installation
To use QueryBot, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running pip install -r requirements.txt.
3. Set up your environment variables by creating a .env file and adding your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key
```
5. Run the application using Streamlit:
```
streamlit run main.py.
```

## Usage
Once the application is running, visit the provided URL to access the QueryBot interface. 
You can ask questions related to CodeBasics courses and bootcamps, and the bot will provide relevant answers. 
You can also download the conversation history for future reference.

## Project Structure
* main.py: The main Streamlit application script.
* requirements.txt: A list of required Python packages for the project.
* .env: Configuration file for storing your OpenAI API key.
* codebasics_faqs.csv: FAQs used by the company staff to provide reponses

## Credits

This project was inspired by the following works:

- [Codebasics Q&A: Question and Answer System Based on Google Palm LLM and Langchain for E-learning company](https://github.com/codebasics/langchain/tree/main/3_project_codebasics_q_and_a) by codebasics
  
- [MemoryBot](https://github.com/avrabyt/MemoryBot/tree/298e88d895d20c7e3480653291346e398d8e5f9c) by avrabyt
  
- [PersonalMemoryBot](https://github.com/avrabyt/PersonalMemoryBot/tree/d979015b4c472a95241d33b5102d0fb141d46338) by avrabyt
