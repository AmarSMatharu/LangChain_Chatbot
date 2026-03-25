# PersonaChat

A customizable AI assistant platform that lets users create, manage, and interact with dynamic personas while maintaining conversational memory.

---

## Features

-  Persona-Based Conversations  
  Choose from preset personas (e.g., Socrates, Research Translator, Motivational Coach) or create your own custom assistants.

-  Dynamic Persona Management  
  Add, delete, and switch between custom personas in real time — no code changes required.

-  Conversational Memory  
  Maintains context across messages using LangGraph for more natural, stateful interactions.

-  Real-Time Prompt Engineering  
  System prompts are dynamically generated based on the selected persona and user-defined instructions.

-  Interactive UI  
  Built with Streamlit for a clean and responsive user experience.

---

##  Example Personas

- General Assistant → Helpful, intelligent, and conversational AI assistant  
- Socrates → Guides thinking through questions
- Research Translator → Simplifies complex research into clear, digestible explanations  
- Motivational Coach → An encouraging assistant who helps the user stay disciplined, focused, and motivated
- Strategic Advisor → A high-level strategy consultant that focuses on clarity, priorities, and outcomes  

---

##  Tech Stack

- Streamlit  
- LangChain  
- LangGraph  
- Groq API  
- Python  

---

##  How It Works

1. User selects or creates a persona  
2. A dynamic system prompt is generated  
3. Chat history + persona are passed into the model  
4. LangGraph manages conversational memory  
5. Responses adapt based on persona and context  

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/personachat.git
cd personachat
