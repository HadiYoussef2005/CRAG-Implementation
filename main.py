import os, json
import http.client
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document  

load_dotenv()

search_api_key = os.getenv('SEARCH_API_KEY')
open_ai_key = os.getenv('OPEN_AI_KEY')

output_dir = './sygen_storage'

# Initialize the sentence transformer
embedding_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

# Creating the choice chat prompt template
choice_system_template = """
You are a choice-maker. You will be given a list of answers alongside a question, and you are to return the answer out of them that you believe is the most fitting to the question

Return output in a $JSON_BLOB, as shown:\n\n\
{{\n\
    "action": "Choosing",\n\
    "answer": "The best answer out of the list of options"\n\
}}\n\n\
Reminder to ALWAYS respond with a valid $JSON_BLOB of a single action and nothing else.
"""

choice_human_template = """
Here is the list of answers: {answers}
Here is the question: {question}

Please choose the best answer out of that list
"""

choice_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],
                                                      template=choice_system_template)),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['answers', 'question'],
                                                      template=choice_human_template)),
])

# Creating the evaluator chat prompt template
evaluator_system_template = """
You are an evaluator. You return a score from 0.00-1.00 for the relevance of context to the given question. It is imperative that the score is not below 0.00 or over 1.00.
You are also given a score that was calculated by another model BASE, you can return the exact same score if you believe that this score is correct.

Return output in a $JSON_BLOB, as shown:\n\n\
{{\n\
  "action": "Relevance Score",\n\
  "score": "Final Score between [0.00, 1.00]"\n\
}}\n\n\
Reminder to ALWAYS respond with a valid $JSON_BLOB of a single action and nothing else.
"""

evaluator_human_template = """
Here is the question: {question}
The provided context is: {context}
Here is the BASE model score: {eval}
Score the relevance of the context to the question from [0.00, 1.00].
"""

evaluator_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],
                                                      template=evaluator_system_template)),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['question', 'context', 'eval'],
                                                      template=evaluator_human_template)),
])

# Creating the web search chat prompt template
# search_system_template = """You are a web searcher. 
# You change a question into a query that is fit for a google search utilizing key words, and search for the answer."""

# search_human_template = """Here is the question: {question}
# Can you reword this question into a query and in order to find the answer from the web?"""

# search_prompt = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],
#                                                       template=search_system_template)),
#     HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['question'],
#                                                       template=search_human_template)),
# ])

# Creating the final answer chat prompt template
answer_system_template = """You are a Q and A bot.
You are given context and a question, and are to provide the answer"""

answer_human_template = """Here is the context: {context}
Here is the question: {question}

Please answer the question given the context"""

answer_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],
                                                      template=answer_system_template)),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'],
                                                     template=answer_human_template))
])

# Creating the query changer chat prompt template
query_system_template = """You are an assistant for question-answering tasks and can use a RAG tool to search relevant information in a knowledge base. \
Formulate a natural language QUERY that can be used by the RAG tool to retrieve semantically relevant information from the knowledge base. \
The query should be detailed and enable the RAG tool to provide an answer to the QUESTION being asked by the user. \
Remember to always respond with the QUERY and nothing else."""

query_human_template = """Here is the question: {question}. Please change the question to a query fitting semantic search.
Return output in a $JSON_BLOB, as shown:\n\n\
{{\n\
  "action": "Rewording question",\n\
  "query": "The Reworded Query"\n\
}}\n\n\
"""

query_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],
                                                      template=query_system_template)),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['question'],
                                                     template=query_human_template))
])

# Creating the specificity score chat prompt template
specificity_system_template = """I want you to evaluate the specificity of questions and assign a "specificity score" on a scale from 0 to 5. 
The specificity score indicates how specific the question, which is based on the likert scale.

A score of 0 means "the question is completely ambiguous."
A score of 5 means "the question is very, very, very specific."
Here are some examples to guide you in assigning the scores:

Example Question: "At what date and time did Argentina play Mexico in the 2022 World Cup?"

Specificity Score: 5
Reasoning: This question is highly specific as it asks for an exact date and time for a particular event.

Example Question: "What was the score of the match between Argentina and Brazil on July 10, 2021?"

Specificity Score: 4
Reasoning: This question is quite specific, asking for the score of a particular match on a particular date.

Example Question: "What are the main causes of climate change?"

Specificity Score: 2
Reasoning: This question is somewhat specific but still broad, as it asks for a general overview of causes.

Example Question: "Can you tell me something about World War II?"

Specificity Score: 1
Reasoning: This question is vague, asking for general information about a broad topic.

Example Question: "How do I calculate the final score?"

Specificity Score: 0
Reasoning: This question is very ambiguous as it does not specify the context (e.g., sport, game, test).

Return output in a $JSON_BLOB, as shown:\n\n\
{{\n\
  "action": "Relevance Score",\n\
  "score": "Final Score between [0, 5] (MUST BE OF FLOAT DATA TYPE ONLY)"\n\
}}\n\n\
Reminder to ALWAYS respond with a valid $JSON_BLOB of a single action and nothing else.
"""
# Create a score for 3

specificity_human_template = """Please evaluate the following question and provide a specificity score along with a brief explanation for each score:

{question}

"""

specificity_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],
                                                      template=specificity_system_template)),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['question'],
                                                      template=specificity_human_template)),
])

# Setting up the vector store
pdf_directory = "./data"

loader = PyPDFDirectoryLoader(pdf_directory)
data = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(data)

model = OpenAIEmbeddings(api_key=open_ai_key, model="text-embedding-3-large")
db = Chroma.from_documents(documents, model)

# Setting up the agents
# os.environ["TAVILY_API_KEY"] = "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"
# search_tool = TavilySearchResults()
# tools = [search_tool]

# react_prompt = hub.pull("hwchase17/react-chat-json")
# search_agent = create_json_chat_agent(llm=llm, tools=tools, prompt=react_prompt)
# search_agent_chain = AgentExecutor(agent=search_agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)

# Creating the chains
llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo-0125", api_key=open_ai_key)
evaluator_agent_chain = (evaluator_prompt | llm)
answer_agent_chain = (answer_prompt | llm)
query_changer_chain = (query_prompt | llm)
specificity_chain = (specificity_prompt | llm)
choice_chain = (choice_prompt | llm)
# evaluator_agent = create_json_chat_agent(llm=llm, tools=[], prompt=evaluator_prompt)
# evaluator_agent = evaluator_agent.bind(stop=[])
# evaluator_agent_chain = AgentExecutor(agent=evaluator_agent, tools=[], verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)

# Scoring function
def score_func(contextQuestionPair, embedding_model):
    separator = "##"
    arr = contextQuestionPair.split(separator)
    context = arr[0]
    question = arr[1]
    query_emb = embedding_model.encode(question, convert_to_tensor=True)
    doc_emb = embedding_model.encode(context, convert_to_tensor=True)
    cosine_score = util.cos_sim(query_emb, doc_emb)[0].item()
    return cosine_score

# Setting up the Retrieval Evaluator
def retrieval_evaluator(question, context_list, agent):
    score_dict = {}

    for i, context in enumerate(context_list):
        print(f"On Index {i} \n-----------------------------")
        context_str = context.page_content
        contextQuestionPair = context_str + "##" + question
        manual_score = score_func(contextQuestionPair, embedding_model)
        
        response = agent.invoke({"question": question, "context": context_str, "eval": manual_score})
        response_dict = json.loads(response.content)
        
        # Extract LLM score from response_dict
        score = float(response_dict["score"])
        
        # Validate score range
        if score > 1:
            score = 1
        elif score < 0:
            score = 0
        
        # Print scores for debugging
        print(f"This is LLM score: {score}")
        print(f"This is manual score {manual_score}\n\n")
        
        # Calculate final score
        final_score = (manual_score + score) / 2
        score_dict[context_str] = final_score

    print("scoring over")
    ctr = 0
    for k in score_dict.keys():
        ctr += 1
        print(ctr, score_dict[k])
    return score_dict

# Defining the confidence
def confidence_rating(question, context_list, agent, specificity_score):
    incorrects = 0  # Counter to see how many have a low confidence score
    score_dict = retrieval_evaluator(question, context_list, agent)
    relevance_dict = {}
    correct_flag = False
    for context in context_list:
        context_str = context.page_content
        if score_dict[context_str] > (0.53 * (1 + (specificity_score / 10))):
            correct_flag = True
            relevance_dict[context_str] = 1
            score_dict["Confidence"] = "CORRECT"
        elif score_dict[context_str] < 0.2:
            relevance_dict[context_str] = -1
            incorrects += 1
        else:
            relevance_dict[context_str] = 0
    if incorrects == len(context_list):
        score_dict["Confidence"] = "INCORRECT"
        return score_dict, relevance_dict
    elif correct_flag:
        return score_dict, relevance_dict
    else:
        score_dict["Confidence"] = "AMBIGUOUS"
        return score_dict, relevance_dict


# Defining the action trigger
def action_trigger(question, context_list, agent, specificity_score):
    score_dict, relevance_dict = confidence_rating(question, context_list, agent, specificity_score)
    if score_dict["Confidence"] == "CORRECT":
        print("This is without web search")
        return [knowledge_refine(relevance_dict)]
    elif score_dict["Confidence"] == "INCORRECT":
        print("This is with web search")
        return [web_scraper(question, specificity_score, agent)]
    elif score_dict["Confidence"] == "AMBIGUOUS":
        print("This is with ambiguous search")
        return [web_scraper(question, specificity_score, agent), knowledge_refine(relevance_dict)]

# Defining the knowledge refine function
def knowledge_refine(relevance_dict):
    keys_with_value_1 = [key for key, value in relevance_dict.items() if value == 1 or value == 0]
    keys_with_value_1_str = ', '.join(keys_with_value_1)
    print(f"\nDATABASE INFO\n\n {keys_with_value_1_str} \n\n")
    return keys_with_value_1_str

# Defining the web search function
# def web_search(question, search_agent):
#     prompt = f"""You are a web searcher, and are to reword a question into a web search query using keywords and return the answer that you find through a web search. Here is the question: {question}"""
#     output = search_agent.invoke({"input": prompt})
    
#     search_info = output["output"]

#     print(f"\nWEB SEARCH INFO\n\n{search_info}\n\n")
#     return output["output"]

def specificity_scoring(llm, question):
    specificity_score = llm.invoke({"question": question})
    score_dict = json.loads(specificity_score.content)
    score = float({"output": score_dict["score"]}["output"])
    print(f"The specificity score is {score}")
    return score

def extract_visible_text(html_content):
    """Extracts visible text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(lambda text: not text.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]'], texts)
    return u" ".join(t.strip() for t in visible_texts)

def web_scraper(query, specificity_score, agent):
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "q": query
    })
    headers = {
        "X-API-KEY": search_api_key,
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read().decode('utf-8')  # Decode bytes to string
    json_data = json.loads(data)  # Convert string to JSON object

    links = []

    # Extract link from answerBox
    if 'answerBox' in json_data and 'link' in json_data['answerBox']:
        answer_link = json_data['answerBox']['link']
        links.append(answer_link)

    # Extract links from organic results
    if 'organic' in json_data:
        for item in json_data['organic']:
            organic_link = item['link']
            links.append(organic_link)

    # Extract links from sitelinks in organic results (if available)
    for item in json_data.get('organic', []):
        if 'sitelinks' in item:
            for sitelink in item['sitelinks']:
                sitelink_link = sitelink['link']
                links.append(sitelink_link)

    # Extract links from related searches (if available)
    for item in json_data.get('relatedSearches', []):
        if 'link' in item:
            query_link = item['link']
            links.append(query_link)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each link and scrape visible text into a text file
    for index, link in enumerate(links, start=1):
        try:
            # Fetch the webpage content
            response = requests.get(link)
            if response.status_code == 200:
                html_content = response.content.decode('utf-8')
                visible_text = extract_visible_text(html_content)
            else:
                print(f"Failed to fetch {link}: {response.status_code}")
                continue  
            
            txt_path = os.path.join(output_dir, f"result_{index}.txt")

            # Save visible text as a text file
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(visible_text)

            print(f"Successfully saved visible text file for {link}")
        
        except Exception as e:
            print(f"Error processing {link}: {e}")
            continue  # Skip to the next link

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    data = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                file_content = file.read()
                document = Document(page_content=file_content) 
                data.append(document)
                
    documents = text_splitter.split_documents(data)
    model = OpenAIEmbeddings(api_key=open_ai_key, model="text-embedding-3-large")
    database = Chroma.from_documents(documents, model)

    i = 0
    while i < len(documents):
        doc = documents[i]
        if len(doc.page_content) > 1000:
            print("Initiating manual text splitting")
            page_content = doc.page_content
            metadata = doc.metadata
            counter = 0
            while len(page_content) > 1000:
                documents.insert(i + counter, Document(page_content=page_content[:1000], metadata=metadata))
                page_content = page_content[1000:]
                counter += 1
            # Insert the final chunk
            documents.insert(i + counter, Document(page_content=page_content, metadata=metadata))
            # Remove the original oversized chunk
            documents.pop(i + counter + 1)
        i += 1

    # Update the database with the new chunks
    database = Chroma.from_documents(documents, model)

    print(database)
    
    scraped_query_embedding = model.embed_query(query)
    scraped_context_list = database.similarity_search_by_vector(scraped_query_embedding, 5)
    scraped_score_dict, scraped_relevance_dict = confidence_rating(query, scraped_context_list, agent, specificity_score)
    return knowledge_refine(scraped_relevance_dict)


query = input("Enter a question: ")
question_message = query_changer_chain.invoke({"question": query})
question_json = json.loads(question_message.content) 
reworded_question = question_json["query"]  
print(reworded_question)
embedding_reworded_query = model.embed_query(reworded_question)
embedding_query = model.embed_query(query)
context_list = db.similarity_search_by_vector(embedding_query, 10)
context_list_reworded = db.similarity_search_by_vector(embedding_reworded_query, 10)

reworded_specificity_score = specificity_scoring(specificity_chain, reworded_question)
specificity_score = specificity_scoring(specificity_chain, query)
print(f"This is the specificity score {specificity_score} \n\n and this, the reworded specificity score {reworded_specificity_score} \n")
relevant_context = action_trigger(query, context_list, evaluator_agent_chain, specificity_score)
relevant_context1 = action_trigger(reworded_question, context_list_reworded, evaluator_agent_chain, reworded_specificity_score)

relevant_context.append(relevant_context1)

print("Relevant search has been achieved!\n\n")

relevant_context_str = ""

for item in relevant_context:
    if isinstance(item, list):
        relevant_context_str += ", ".join(item) + "\n"
    else:
        relevant_context_str += item + "\n"

prompt = f"Based on the following context, answer the question.\ncontext:{relevant_context_str}\nquestion:{query}"

if len(relevant_context_str) > 10000:
    print("Entered cutting phase")
    cut_context_list = []
    while(True):
        if len(relevant_context_str) > 10000:
            cut_context_list.append(relevant_context_str[0:10000])
            relevant_context_str = relevant_context_str[10000:]
        else:
            cut_context_list.append(relevant_context_str)
            break
    answer_list = []
    for context in cut_context_list:
        answer = answer_agent_chain.invoke({"context":context, "question":query}).content
        answer_list.append(answer)
    choice_response = choice_chain.invoke({"question": query, "answers": answer_list})
    choice_dict = json.loads(choice_response.content)
    choice = {"output": choice_dict["answer"]}["output"]
    print(choice)

else:        
    print(answer_agent_chain.invoke({"context":relevant_context_str, "question":query}).content)
