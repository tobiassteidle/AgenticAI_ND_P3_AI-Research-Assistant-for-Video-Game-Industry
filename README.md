## Project Scenario
You’ve been hired as an AI Engineer at a gaming analytics company developing an assistant called UdaPlay. Executives, analysts, and gamers want to ask natural language questions like:

- “Who developed FIFA 21?”
- “When was God of War Ragnarok released?”
- “What platform was Pokémon Red launched on?”
- “What is Rockstar Games working on right now?”
- 
### Your agent should:
Attempt to answer the question from internal knowledge (about a pre-loaded list of companies and games)
If the information is not found or confidence is low, search the web
Parse and persist the information in long-term memory
Generate a clean, structured answer/report
Project Specifications
In this project, you will build an AI Research Agent called UdaPlay designed to answer questions about video games. The agent will be capable of:

#### Answering user questions about games, including:

- Game titles and their details
- Release dates and platforms
- Game descriptions and genres
- Publisher information

#### Using a two-tier information retrieval system:

- Primary: RAG (Retrieval Augmented Generation) over a local dataset of games
- Secondary: Web search using the Tavily API when internal knowledge is insufficient


#### Implementing a robust evaluation system:

- Assessing the quality of retrieved information
- Determining when to fall back to web search
- Providing confidence levels in answers

#### Generating clear, well-structured responses that:

- Cite information sources
- Combine information from multiple sources when needed
- Present information in a natural, readable format


### Part 1 - RAG Pipeline
- Set up a ChromaDB vector database
- Process and embed game data from JSON files
- Implement semantic search functionality
- Create a reusable vector store manager

### Part 2 - Agent Implementation
Build an agent with three core tools:
- retrieve_game: Search the vector database
- evaluate_retrieval: Assess answer quality
- game_web_search: Fall back to web search
- Implement a state machine for agent workflow
- Create a reporting system for clear output
