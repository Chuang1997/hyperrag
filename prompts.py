DECOMPOSE_HEAD = """Your task is to decompose the given question Q into sub-questions. You should based on the specific logic of the question to determine the number of sub-questions and output them sequentially. If you consider the question Q to be sufficiently simple and no further decomposition is needed, then output "End." I will provide you with a question similar to q, along with its decomposed sub-questions as examples. You can learn from the example on how to decompose such question, and then apply what you've learned to decompose Q.

IMPORTANT GUIDELINES:
- Decompose into the MINIMUM number of sub-questions necessary to answer the original question
- Each sub-question should be as SIMPLE and DIRECT as possible
- Avoid redundant or overlapping sub-questions
- If a question can be answered directly without decomposition, output "End."
- Focus on essential logical components only
"""

DECOMPOSE_INSTRUCTION = """Please note that: If Q can be decomposed, you should output multiple sub-questions as shown in the above Logic Heuristics. Otherwise please output "End". 

Remember to:
- Keep sub-questions minimal and non-redundant
- Make each sub-question as simple as possible
- Only decompose when absolutely necessary
- Output sub-questions in logical order

The decomposed sub-questions for Q is:
"""

CHAIN_REASONING_HEAD = """You are a factual question answering assistant. Answer the sub-question directly and concisely.

CRITICAL RULES:
1. You MUST provide a direct factual answer - never say the context doesn't provide information
2. Use the context if relevant, otherwise use your own knowledge
3. Never say "unknown", "not available", "context doesn't provide", or similar phrases
4. Always give your best answer based on available information or general knowledge"""

CHAIN_REASONING_INSTRUCTION = """Output ONLY the direct answer - no explanations, no meta-commentary about the context. Just the answer itself."""

