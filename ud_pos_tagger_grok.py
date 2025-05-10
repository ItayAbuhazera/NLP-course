# See https://docs.x.ai/docs/guides/structured-outputs 
# --- Imports ---
import os
from openai import OpenAI

from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional

# Limit: 5 requests per second 
# Context: 131,072 tokens
# Text input: $0.30 per million
# Text output: $0.50 per million
model = 'grok-3-mini'


# --- Define Pydantic Models for Structured Output ---

# --- Define the Universal Dependencies POS Tagset (17 core tags) as an enum ---
class UDPosTag(str, Enum):
    """Universal Dependencies POS tags (17 core tags)"""
    ADJ = "ADJ"     # adjective
    ADP = "ADP"     # adposition (preposition, postposition)
    ADV = "ADV"     # adverb
    AUX = "AUX"     # auxiliary verb
    CCONJ = "CCONJ" # coordinating conjunction
    DET = "DET"     # determiner
    INTJ = "INTJ"   # interjection
    NOUN = "NOUN"   # noun
    NUM = "NUM"     # numeral
    PART = "PART"   # particle
    PRON = "PRON"   # pronoun
    PROPN = "PROPN" # proper noun
    PUNCT = "PUNCT" # punctuation
    SCONJ = "SCONJ" # subordinating conjunction
    SYM = "SYM"     # symbol
    VERB = "VERB"   # verb
    X = "X"         # other


class TokenPOS(BaseModel):
    """A token with its POS tag."""
    token: str = Field(description="The token text.")
    pos_tag: UDPosTag = Field(description="The Universal Dependencies POS tag.")

class SentencePOS(BaseModel):
    """A sentence with its tokenized and POS-tagged components."""
    tokens: List[TokenPOS] = Field(description="The list of tokens with their POS tags.")
    text: Optional[str] = Field(description="The original sentence text.", default=None)

class TaggedSentences(BaseModel):
    """Represents a list of sentences with their tagged tokens."""
    sentences: List[SentencePOS] = Field(description="A list of sentences, each containing tagged tokens.")


# --- Configure the Grok API ---
# Get a key https://console.x.ai/team 
# Use os.environ.get for production environments.
# For Colab/AI Studio, you might use userdata.get
# Example:
# from google.colab import userdata
# GROK_API_KEY = userdata.get('GROK_API_KEY')
# genai.configure(api_key=GROK_API_KEY)

# Make sure to replace "YOUR_API_KEY" with your actual key if running locally
# and not using environment variables or userdata.
try:
    # Attempt to get API key from environment variable
    api_key = os.environ.get("GROK_API_KEY")
    if not api_key:
        # Fallback or specific instruction for local setup
        # Replace with your actual key if needed, but environment variables are safer
        api_key = "xai-YOUR_API_KEY"
        if api_key == "YOUR_API_KEY":
           print("⚠️ Warning: API key not found in environment variables. Using placeholder.")
           print("   Please set the GROK_API_KEY environment variable or replace 'YOUR_API_KEY' in the code.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

except Exception as e:
    print(f"Error configuring API: {e}")
    print("Please ensure you have a valid API key set.")
    # Depending on the environment, you might want to exit here
    # import sys
    # sys.exit(1)


# --- Function to Perform POS Tagging ---

def tag_sentences_ud(text_to_tag: str) -> Optional[TaggedSentences]:
    """
    Performs POS tagging on the input list of sentences using the Grok API and
    returns the result structured according to the TaggedSentences Pydantic model.

    Args:
        text_to_tag: The sentence or text to be tagged.

    Returns:
        A SentencePOS object containing the tagged tokens, or None if an error occurs.
    """
    # Construct the prompt
    prompt = f"""You are an expert linguistic annotator specializing in Part-of-Speech (POS) tagging according to Universal Dependencies (UD) guidelines.
    TASK: Analyze the given text by:
    1. Splitting it into sentences
    2. Tokenizing each sentence according to UD tokenization guidelines
    3. Assigning the correct UD POS tag to each token

    TOKENIZATION GUIDELINES:
    - Split contractions: "don't" → "do n't", "I'll" → "I 'll"
    - Split punctuation from words: "word." → "word ."
    - Split hyphens in most cases: "full-fledged" → "full - fledged"
    - Keep special entities intact (emails, URLs, etc.)

    UNIVERSAL DEPENDENCIES POS TAGS:
    - ADJ: adjective (e.g., big, old, green)
    - ADP: adposition (e.g., in, to, during)
    - ADV: adverb (e.g., very, tomorrow, down)
    - AUX: auxiliary verb (e.g., is, has, will)
    - CCONJ: coordinating conjunction (e.g., and, or, but)
    - DET: determiner (e.g., a, the, this)
    - INTJ: interjection (e.g., psst, ouch, wow)
    - NOUN: noun (e.g., girl, cat, tree)
    - NUM: numeral (e.g., 1, 2017, one)
    - PART: particle (e.g., 's, not, n't)
    - PRON: pronoun (e.g., I, you, he)
    - PROPN: proper noun (e.g., Mary, John, Google)
    - PUNCT: punctuation (e.g., ., (, ))
    - SCONJ: subordinating conjunction (e.g., if, while, that)
    - SYM: symbol (e.g., $, %, =)
    - VERB: verb (e.g., run, eat)
    - X: other (e.g., foreign words, abbreviations)

    FORMATTING:
    Return the result as a structured JSON object following this exact format:
    {
    "sentences": [
        {
        "text": "Original sentence text.",
        "tokens": [
            {"token": "Original", "pos_tag": "ADJ"},
            {"token": "sentence", "pos_tag": "NOUN"},
            {"token": "text", "pos_tag": "NOUN"},
            {"token": ".", "pos_tag": "PUNCT"}
        ]
        }
    ]
    }

    EXAMPLES:
    Input: "I don't want to go."
    Output: 
    {
    "sentences": [
        {
        "text": "I don't want to go.",
        "tokens": [
            {"token": "I", "pos_tag": "PRON"},
            {"token": "do", "pos_tag": "AUX"},
            {"token": "n't", "pos_tag": "PART"},
            {"token": "want", "pos_tag": "VERB"},
            {"token": "to", "pos_tag": "PART"},
            {"token": "go", "pos_tag": "VERB"},
            {"token": ".", "pos_tag": "PUNCT"}
        ]
        }
    ]
    }    
    """

    completion = client.beta.chat.completions.parse(
        model="grok-3-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Now, analyze the following text: {text_to_tag}"},
        ],
        response_format=TaggedSentences,
    )
    # print(completion)
    res = completion.choices[0].message.parsed
    return res


# --- Example Usage ---
if __name__ == "__main__":
    # example_text = "The quick brown fox jumps over the lazy dog."
    example_text = """
What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?
Google Search is a web search engine developed by Google LLC.
It does n't change the company 's intrinsic worth , and as the article notes , the company might be added to a major index once the shares get more liquid .
I 've been looking at the bose sound dock 10 i ve currently got a jvc mini hifi system , i was wondering what would be a good set of speakers .
which is the best burger chain in the chicago metro area like for example burger king portillo s white castle which one do you like the best ?
"""
    # example_text = "החתול המהיר קופץ מעל הכלב העצלן." # Example in Hebrew

    print(f"\nTagging text: \"{example_text}\"")

    tagged_result = tag_sentences_ud(example_text)

    if tagged_result:
        print("\n--- Tagging Results ---")
        for sentence in tagged_result.sentences:
            print(f"\nSentence: {sentence.text if sentence.text else 'N/A'}")
            print("-" * 50)
            for token_pos in sentence.tokens:
                token = token_pos.token
                tag = token_pos.pos_tag
                # Handle potential None for pos_tag if model couldn't assign one
                ctag = tag if tag is not None else "UNKNOWN"
                print(f"Token: {token:<15} {str(ctag)}")
            print("-" * 50)
    else:
        print("\nFailed to get POS tagging results.")
