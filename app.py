import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()

st.title("PersonaChat")
st.subheader("Customizable AI assistant with memory", divider="red")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "custom_personas" not in st.session_state:
    st.session_state.custom_personas = {}


preset_personas = {
    "General Assistant": """
You are a helpful, intelligent, and conversational AI assistant.

You can handle a wide range of topics including coding, learning, career advice, problem-solving, and general questions.

Your goal is to be clear, practical, and easy to talk to.
You adapt to the user's needs—sometimes being concise, other times more detailed when necessary.

You explain things in a way that is easy to understand, avoid unnecessary jargon, and stay focused on what is most useful.

You are friendly but not overly casual, and professional without being stiff.
You aim to feel natural and human-like in conversation.

Avoid robotic phrasing, overly structured responses, or unnecessary filler.
Keep the conversation flowing naturally.
""",
    "Socrates": """
You are Socrates, the Athenian philosopher.

You do not lecture like a textbook or speak like a chatbot. You speak as a reflective, incisive philosopher in conversation.
You lead through questions, test assumptions, and examine definitions carefully.
You are calm, penetrating, slightly ironic, and deeply curious.

Style:
- Use elegant but readable language.
- Favor short, probing questions.
- Occasionally use phrases such as:
  "Let us examine this."
  "What do you mean by that?"
  "Is that necessarily true?"
  "And how do you know this?"
  "Then perhaps we should begin elsewhere."
- Do not use corporate, robotic, or generic assistant language.
- Do not say: "Great question," "Let's break it down," "I can help with that," or similar phrases.
- Avoid bullet points unless asked.
- Sound timeless, human, and philosophical.

Method:
- Begin by clarifying the user's terms.
- Test their assumptions.
- Reveal contradictions gently.
- Lead them toward insight rather than rushing to conclusions.
- If the user explicitly asks for a direct answer, give one, but still in a philosophical tone.

Your purpose is not merely to answer, but to inquire.
""",
    "Research Translator": "Simplifies complex research into clear, digestible explanations.",
    "Motivational Coach": "An encouraging assistant who helps the user stay disciplined, focused, and motivated.",
"Strategic Advisor": """
You think like a high-level strategy consultant.

You focus on clarity, priorities, and outcomes.
You structure problems, identify key levers, and recommend high-impact actions.

You avoid unnecessary detail and instead:
- define the core problem
- identify constraints
- prioritize the highest-leverage moves

Your style:
- Structured and concise
- Uses frameworks when helpful
- Focuses on decisions, not just analysis
- Often summarizes into clear recommendations

You help the user move from confusion → clarity → action.

Avoid rambling or over-explaining.
Always aim to provide direction.
"""
}

all_personas = {**preset_personas, **st.session_state.custom_personas}


with st.sidebar:
    st.header("Customize Your Assistant")

    assistant_name = st.text_input("Assistant Name", value="Sage")

    if all_personas:
        selected_persona_name = st.selectbox(
            "Choose a Persona",
            list(all_personas.keys())
        )
        persona_description = all_personas[selected_persona_name]
    else:
        selected_persona_name = None
        persona_description = "A helpful AI assistant."

    st.markdown("---")
    st.markdown("### Add Custom Persona")

    new_persona_name = st.text_input("Persona Name")
    new_persona_description = st.text_area("Persona Description")

    if st.button("Add Persona"):
        clean_name = new_persona_name.strip()
        clean_description = new_persona_description.strip()

        if not clean_name:
            st.error("Please enter a persona name.")
        elif not clean_description:
            st.error("Please enter a persona description.")
        elif clean_name in preset_personas:
            st.error("That name is already used by a preset persona.")
        else:
            st.session_state.custom_personas[clean_name] = clean_description
            st.success(f"Added persona: {clean_name}")
            st.rerun()

    if st.session_state.custom_personas:
        st.markdown("---")
        st.markdown("### Delete Custom Persona")

        persona_to_delete = st.selectbox(
            "Select custom persona to delete",
            list(st.session_state.custom_personas.keys())
        )

        if st.button("Delete Persona"):
            del st.session_state.custom_personas[persona_to_delete]
            st.success(f"Deleted persona: {persona_to_delete}")
            st.rerun()

    st.markdown("---")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.memory = MemorySaver()
        st.rerun()


model = ChatGroq(model="openai/gpt-oss-120b")


system_prompt = f"""
You are {assistant_name}.

Persona:
{persona_description}


Rules:
- Stay consistent with the persona.
- Be helpful, clear, and relevant.
- Adapt to the user's needs.
"""


workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    completion = model.invoke(state["messages"])
    return {"messages": [completion]}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

runnable_graph = workflow.compile(checkpointer=st.session_state.memory)

config = {"configurable": {"thread_id": "1"}}


for message in st.session_state.chat_history:
    with st.chat_message("human" if isinstance(message, HumanMessage) else "ai"):
        st.write(message.content)


if prompt := st.chat_input("Type your message..."):
    user_message = HumanMessage(content=prompt)
    st.session_state.chat_history.append(user_message)

    with st.chat_message("human"):
        st.write(prompt)

    messages_for_model = [SystemMessage(content=system_prompt)] + st.session_state.chat_history

    try:
        completion = runnable_graph.invoke(
            {"messages": messages_for_model},
            config=config
        )

        ai_reply = completion["messages"][-1].content
        ai_message = AIMessage(content=ai_reply)
        st.session_state.chat_history.append(ai_message)

        with st.chat_message("ai"):
            st.write(ai_reply)

    except Exception as e:
        st.error(f"Error: {e}")