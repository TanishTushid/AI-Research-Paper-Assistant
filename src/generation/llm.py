
from pathlib import Path
from llama_cpp import Llama


class LLMModel:
    """
    Load a local GGUF language model and generate responses.
    """

    def __init__(
        self,
        model_path=None,
        context_size=4096,
        n_threads=4
    ):

        # Resolve project root directory
        project_root = Path(__file__).resolve().parents[2]

        # Default model location
        if model_path is None:
            model_path = (
                project_root
                / "models"
                / "llm"
                / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
            )

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"LLM model not found: {self.model_path}"
            )

        print("Loading local LLM...")

        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=context_size,
            n_threads=n_threads,
            n_batch=256,
            verbose=False
        )

        print("LLM loaded successfully!")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI research assistant.",
        max_tokens: int = 512,
        temperature: float = 0.2
    ) -> str:
        """
        Generate a response from the LLM.
        """

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9
        )

        answer = response["choices"][0]["message"]["content"]

        return answer.strip()

    def answer_question(
        self,
        question: str,
        context: str
    ) -> str:
        """
        Answer a question using retrieved document context.
        """

        system_prompt = """
        You are an AI Research Paper Assistant.

        Answer questions using only the provided research context.

        Rules:
        1. Do not invent facts or results.
        2. If the context does not contain the answer,
           clearly say that the information is not available.
        3. Explain technical concepts clearly.
        4. Preserve important scientific terminology.
        """

        prompt = f"""
        Research Context:

        {context}

        User Question:

        {question}

        Provide a clear and accurate answer based on the context.
        """

        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )
