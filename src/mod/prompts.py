from langchain.prompts import PromptTemplate
from typing import List

class NL2SQLPrompts:
    """Collection of prompts for NL2SQL processing"""
    
    @staticmethod
    def get_table_selection_prompt() -> PromptTemplate:
        """
        Prompt for selecting relevant tables based on user question
        """
        template = """
        You are a database expert analyzing which tables are needed to answer a user's question.

        Database Schema:
        {schema}

        User Question: {question}

        Analyze the question and identify the most relevant tables needed to answer it.
        Consider:
        1. Table names that relate to the question topic
        2. Column names within tables that might contain relevant data
        3. Relationships between tables that might be needed for joins
        4. Foreign key relationships

        Return only the table names as a comma-separated list, nothing else.
        If no tables seem relevant, return "NO_TABLES_FOUND".

        Example format: table1, table2, table3

        Relevant Tables:"""
        
        return PromptTemplate(
            input_variables=["question", "schema"],
            template=template
        )
    
    @staticmethod
    def get_sql_generation_prompt() -> PromptTemplate:
        """
        Prompt for generating SQL query from natural language question
        """
        template = """
        You are an expert SQL query generator. Create a precise SQL SELECT query based on the user's question.

        Database Schema:
        {schema}

        Relevant Tables: {tables}

        User Question: {question}

        Previous Conversation Context (if any):
        {context}

        IMPORTANT RULES:
        1. Generate ONLY SELECT queries - no INSERT, UPDATE, DELETE, or ALTER statements
        2. Use proper SQL syntax and formatting
        3. Include appropriate JOINs if multiple tables are needed
        4. Use meaningful table aliases for readability
        5. Consider proper WHERE clauses for filtering
        6. Add GROUP BY, ORDER BY, LIMIT clauses as needed
        7. Use appropriate aggregate functions (COUNT, SUM, AVG, etc.) when needed
        8. Ensure column names and table names exist in the schema
        9. Handle case sensitivity appropriately for the database type
        10. Return ONLY the SQL query without any explanation or markdown formatting

        SQL Query:"""
        
        return PromptTemplate(
            input_variables=["question", "schema", "tables", "context"],
            template=template
        )
    
    @staticmethod
    def get_query_explanation_prompt() -> PromptTemplate:
        """
        Prompt for explaining the generated SQL query
        """
        template = """
        Explain the following SQL query in clear, simple terms for someone who may not know SQL.

        Original Question: {question}

        SQL Query:
        {sql_query}

        Database Schema Context:
        {schema}

        Provide a comprehensive explanation that includes:
        1. What the query is trying to find/calculate
        2. Which tables and columns it uses
        3. Any joins between tables and why they're needed
        4. Filters or conditions applied (WHERE clauses)
        5. Any grouping, sorting, or limiting applied
        6. How the query answers the original question
        7. Any assumptions made in the query logic

        Keep the explanation conversational and easy to understand.

        Explanation:"""
        
        return PromptTemplate(
            input_variables=["question", "sql_query", "schema", "results_count"],
            template=template
        )
    
    @staticmethod
    def get_query_validation_prompt() -> PromptTemplate:
        """
        Prompt for validating generated SQL query
        """
        template = """
        Review the following SQL query for correctness and safety.

        Database Schema:
        {schema}

        Generated SQL Query:
        {sql_query}

        Original Question: {question}

        Check for:
        1. Syntax correctness
        2. Table and column name validity against schema
        3. Proper JOIN conditions
        4. Logical correctness for answering the question
        5. Security issues (only SELECT allowed)
        6. Performance considerations

        Respond with:
        - "VALID" if the query is correct and safe
        - "INVALID: [reason]" if there are issues

        Validation Result:"""
        
        return PromptTemplate(
            input_variables=["schema", "sql_query", "question"],
            template=template
        )
    
    @staticmethod
    def get_result_formatting_prompt() -> PromptTemplate:
        """
        Prompt for formatting query results for better presentation
        """
        template = """
        Format the following query results in a clear, readable way for the user.

        Original Question: {question}

        SQL Query: {sql_query}

        Raw Results:
        {raw_results}

        Create a well-formatted response that:
        1. Provides a brief summary of what was found
        2. Presents the data in an organized, readable format
        3. Highlights key insights or patterns if relevant
        4. Mentions the total number of results
        5. Uses appropriate formatting (tables, lists, etc.)

        If there are no results, provide a helpful message explaining why and suggest potential alternatives.

        Formatted Results:"""
        
        return PromptTemplate(
            input_variables=["question", "sql_query", "raw_results"],
            template=template
        )
    
    @staticmethod
    def get_error_analysis_prompt() -> PromptTemplate:
        """
        Prompt for analyzing SQL execution errors and providing suggestions
        """
        template = """
        Analyze the following SQL error and provide helpful guidance.

        Original Question: {question}

        Generated SQL Query:
        {sql_query}

        Error Message:
        {error_message}

        Database Schema:
        {schema}

        Analyze the error and provide:
        1. What caused the error
        2. Specific issues in the SQL query
        3. Suggestions to fix the query
        4. Alternative approaches to answer the question

        Keep the explanation user-friendly and actionable.

        Error Analysis:"""
        
        return PromptTemplate(
            input_variables=["question", "sql_query", "error_message", "schema"],
            template=template
        )
    
    @staticmethod
    def get_context_aware_prompt() -> PromptTemplate:
        """
        Prompt that considers conversation history for better context
        """
        template = """
        You are helping with a natural language to SQL conversion. Consider the conversation history to provide better responses.

        Conversation History:
        {conversation_history}

        Current Database Schema:
        {schema}

        Current Question: {question}

        Based on the conversation context and current question:
        1. Identify if this question relates to previous queries
        2. Consider any clarifications or refinements from the conversation
        3. Use context to better understand what the user is looking for
        4. Generate an appropriate SQL query that considers the full context

        Generate a SQL SELECT query that answers the current question while considering the conversation context.

        SQL Query:"""
        
        return PromptTemplate(
            input_variables=["conversation_history", "schema", "question"],
            template=template
        )


class PromptManager:
    """Manages and provides access to all NL2SQL prompts"""
    
    def __init__(self):
        self.prompts = NL2SQLPrompts()
    
    def get_prompt(self, prompt_type: str) -> PromptTemplate:
        """
        Get a specific prompt by type
        
        Args:
            prompt_type: Type of prompt to retrieve
                       Options: 'table_selection', 'sql_generation', 'query_explanation',
                               'query_validation', 'result_formatting', 'error_analysis',
                               'context_aware'
        
        Returns:
            PromptTemplate object
        """
        prompt_methods = {
            'table_selection': self.prompts.get_table_selection_prompt,
            'sql_generation': self.prompts.get_sql_generation_prompt,
            'query_explanation': self.prompts.get_query_explanation_prompt,
            'query_validation': self.prompts.get_query_validation_prompt,
            'result_formatting': self.prompts.get_result_formatting_prompt,
            'error_analysis': self.prompts.get_error_analysis_prompt,
            'context_aware': self.prompts.get_context_aware_prompt
        }
        
        if prompt_type not in prompt_methods:
            raise ValueError(f"Unknown prompt type: {prompt_type}. Available types: {list(prompt_methods.keys())}")
        
        return prompt_methods[prompt_type]()
    
    def list_available_prompts(self) -> List[str]:
        """Get list of available prompt types"""
        return [
            'table_selection',
            'sql_generation', 
            'query_explanation',
            'query_validation',
            'result_formatting',
            'error_analysis',
            'context_aware'
        ]
    
    def customize_prompt(self, prompt_type: str, custom_template: str, input_variables: List[str]) -> PromptTemplate:
        """
        Create a custom prompt template
        
        Args:
            prompt_type: Identifier for the custom prompt
            custom_template: The template string
            input_variables: List of input variable names
            
        Returns:
            PromptTemplate object
        """
        return PromptTemplate(
            input_variables=input_variables,
            template=custom_template
        )