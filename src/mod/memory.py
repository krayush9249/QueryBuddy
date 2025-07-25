from langchain.memory import ConversationBufferMemory

def get_conversation_context(memory: ConversationBufferMemory) -> str:
    """Get formatted conversation context from memory"""
    if not memory.chat_memory.messages:
        return "No previous conversation."
    
    context_parts = []
    for message in memory.chat_memory.messages[-5:]:  # Last 5 messages
        if hasattr(message, 'content'):
            role = "User" if message.type == "human" else "Assistant"
            context_parts.append(f"{role}: {message.content}")
    
    return "\n".join(context_parts)