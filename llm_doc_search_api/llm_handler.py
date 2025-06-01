import logging
import os

from llama_cpp import Llama

from . import config

logger = logging.getLogger(__name__)

_llm_instance = None


def get_llm():

    global _llm_instance
    if _llm_instance is None:
        if not os.path.exists(config.LLM_MODEL_PATH):
            logger.error(f"LLM model file not found at: {config.LLM_MODEL_PATH}")
            raise FileNotFoundError(
                f"LLM model file not found: {config.LLM_MODEL_PATH}"
            )

        logger.info(
            f"Loading LLM from path: {config.LLM_MODEL_PATH} with n_ctx={config.LLM_N_CTX}, n_gpu_layers={config.LLM_N_GPU_LAYERS}"
        )
        try:
            _llm_instance = Llama(
                model_path=config.LLM_MODEL_PATH,
                n_ctx=config.LLM_N_CTX,  # Context window size
                n_gpu_layers=config.LLM_N_GPU_LAYERS,  # Number of layers to offload to GPU
                verbose=(
                    logger.getEffectiveLevel() == logging.DEBUG
                ),  # llama.cpp verbose output if debug
            )
            logger.info("LLM model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}", exc_info=True)
            _llm_instance = None  # Ensure it's None if loading failed
            raise RuntimeError(
                f"Failed to load LLM model from {config.LLM_MODEL_PATH}"
            ) from e
    return _llm_instance


def construct_prompt_for_qa(query: str, context_chunks: list[str]) -> str:

    context_str = "\n\n".join(context_chunks)

    prompt = f"<|user|>\nBased on the following context, please answer the question. If the context doesn't provide enough information, say so.\n\nContext:\n{context_str}\n\nQuestion: {query}\n<|assistant|>"

    # A more general prompt template could be:
    # prompt = f"Context information is provided below.\n" \
    #          f"---------------------\n" \
    #          f"{context_str}\n" \
    #          f"---------------------\n" \
    #          f"Given the context information and not prior knowledge, answer the query.\n" \
    #          f"Query: {query}\n" \
    #          f"Answer: "
    logger.debug(f"Constructed LLM Prompt:\n{prompt}")
    return prompt


def generate_llm_response(prompt: str) -> str | None:

    try:
        llm = get_llm()
        if llm is None:
            logger.error("LLM model is not available, cannot generate response.")
            return "Error: LLM model not available."

        logger.info(
            f"Generating LLM response (max_tokens={config.LLM_MAX_TOKENS}, temp={config.LLM_TEMPERATURE})..."
        )

        response = llm(
            prompt,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            stop=[
                "<|end|>",
                "<|user|>",
            ],
            echo=False,
        )

        generated_text = response["choices"][0]["text"].strip()
        logger.info(f"LLM generated response: {generated_text[:200]}...")
        return generated_text

    except Exception as ex:
        logger.error(f"Error during LLM response generation: {ex}", exc_info=True)
        return "Error: Could not generate a response from the LLM."


# Example usage for testing
if __name__ == "__main__":
    # Ensure logging is configured for standalone testing
    from logging.config import dictConfig

    dictConfig(config.LOGGING_CONFIG)

    logger.info("Testing LLM Handler...")

    # Test 1: Model Loading
    try:
        llm_model = get_llm()
        if llm_model:
            logger.info("LLM model loaded successfully for standalone test.")
        else:
            logger.error("LLM model FAILED to load for standalone test.")
            exit()  # Can't proceed without model
    except Exception as e:
        logger.error(f"Standalone test: LLM loading failed: {e}")
        exit()

    # Test 2: Prompt Construction and Generation
    test_query = "What is the capital of France?"
    test_context = [
        "France is a country in Europe.",
        "Paris is its capital city and a major global center for art, fashion, gastronomy and culture.",
    ]

    prompt = construct_prompt_for_qa(test_query, test_context)
    logger.info(f"\n--- Test Prompt ---\n{prompt}\n-------------------\n")

    response_text = generate_llm_response(prompt)
    logger.info(
        f"\n--- Test LLM Response ---\n{response_text}\n-----------------------\n"
    )

    test_query_2 = "What is the main ingredient in bread?"
    test_context_2 = ["The sky is blue on a clear day.", "Water is essential for life."]
    prompt_2 = construct_prompt_for_qa(test_query_2, test_context_2)
    logger.info(
        f"\n--- Test Prompt 2 (irrelevant context) ---\n{prompt_2}\n-------------------\n"
    )
    response_text_2 = generate_llm_response(prompt_2)
    logger.info(
        f"\n--- Test LLM Response 2 ---\n{response_text_2}\n-----------------------\n"
    )
