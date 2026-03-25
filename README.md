PersonaChat

A customizable AI assistant platform that lets users create, manage, and interact with dynamic personas while maintaining conversational memory.

Features
Persona-Based Conversations
Choose from preset personas (e.g., Socratic Tutor, Debugging Specialist, First Principles Thinker) or create your own custom assistants.
Dynamic Persona Management
Add, delete, and switch between custom personas in real time—no code changes required.
Conversational Memory (LangGraph)
Maintains context across messages using persistent memory for more natural interactions.
Real-Time Prompt Engineering
System prompts are dynamically generated based on selected persona and user-defined instructions.
Interactive UI (Streamlit)
Clean, responsive interface with sidebar controls for personalization.
Example Personas
Socratic Tutor → Guides thinking through questions
Debugging Specialist → Finds and fixes bugs efficiently
First Principles Thinker → Breaks problems down to fundamentals
Clinical Reasoning Coach → Walks through diagnostic logic step-by-step
No-BS Advisor → Direct, high-signal advice
Tech Stack
Frontend/UI: Streamlit
LLM Orchestration: LangChain
State + Memory: LangGraph
Model Provider: Groq
Language: Python
How It Works
User selects or creates a persona
A dynamic system prompt is generated
Chat history + persona are passed into the model
LangGraph manages memory across turns
Responses adapt based on persona + context
