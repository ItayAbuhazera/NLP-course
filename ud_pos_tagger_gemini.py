# --- Imports ---
import os
from google import genai
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from enum import Enum
#
gemini_model = 'gemini-2.0-flash-lite'

# --- Define Pydantic Models for Structured Output ---
class SegmentedTokenOutput(BaseModel):
    """Represents a single tokenized sentence."""
    original_text: str = Field(description="The original input sentence text.")
    tokens: List[str] = Field(description="The list of segmented tokens according to UD guidelines.")

class SegmenterLLMResponse(BaseModel):
    """The structured response from the LLM for segmentation."""
    segmented_sentences: List[SegmentedTokenOutput] = Field(description="A list of segmented sentences.")

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

# Token class
class TokenPOS(BaseModel):
    """A token with its POS tag."""
    token: str = Field(description="The token text.")
    pos_tag: UDPosTag = Field(description="The Universal Dependencies POS tag.")

# Sentence class
class SentencePOS(BaseModel):
    """A sentence with its tokenized and POS-tagged components."""
    tokens: List[TokenPOS] = Field(description="The list of tokens with their POS tags.")
    text: Optional[str] = Field(description="The original sentence text.", default=None)

# Tagged sentences class
class TaggedSentences(BaseModel):
    """Represents a list of sentences with their tagged tokens."""
    sentences: List[SentencePOS] = Field(description="A list of sentences, each containing tagged tokens.")

# --- Configure the Gemini API ---
# Get a key https://aistudio.google.com/plan_information
# Use os.environ.get for production environments.
# For Colab/AI Studio, you might use userdata.get
# Example:
# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)

# Make sure to replace "YOUR_API_KEY" with your actual key if running locally
# and not using environment variables or userdata.
try:
    # Attempt to get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Fallback or specific instruction for local setup
        # Replace with your actual key if needed, but environment variables are safer
        api_key = "YOUR_API_KEY"
        if api_key == "YOUR_API_KEY":
            print("⚠️ Warning: API key not found in environment variables. Using placeholder.")
            print("   Please set the GOOGLE_API_KEY environment variable or replace 'YOUR_API_KEY' in the code.")

    genai.configure(api_key=api_key)

except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have a valid API key set.")
    # Depending on the environment, you might want to exit here
    # import sys
    # sys.exit(1)


# --- Function to Perform POS Tagging ---

def tag_sentences_ud(text_to_tag: str) -> Optional[TaggedSentences]:
    """
    Performs POS tagging on the input text using the Gemini API and
    returns the result structured according to the SentencePOS Pydantic model.

    Args:
        text_to_tag: The sentence or text to be tagged.

    Returns:
        A TaggedSentences object containing the tagged tokens, or None if an error occurs.
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
    {{
      "sentences": [
        {{
          "text": "Original sentence text.",
          "tokens": [
            {{"token": "Original", "pos_tag": "ADJ"}},
            {{"token": "sentence", "pos_tag": "NOUN"}},
            {{"token": "text", "pos_tag": "NOUN"}},
            {{"token": ".", "pos_tag": "PUNCT"}}
          ]
        }}
      ]
    }}

    EXAMPLES:
    Input: "I don't want to go."
    Output: 
    {{
      "sentences": [
        {{
          "text": "I don't want to go.",
          "tokens": [
            {{"token": "I", "pos_tag": "PRON"}},
            {{"token": "do", "pos_tag": "AUX"}},
            {{"token": "n't", "pos_tag": "PART"}},
            {{"token": "want", "pos_tag": "VERB"}},
            {{"token": "to", "pos_tag": "PART"}},
            {{"token": "go", "pos_tag": "VERB"}},
            {{"token": ".", "pos_tag": "PUNCT"}}
          ]
        }}
      ]
    }}

    Now, analyze the following text:
    {text_to_tag}
    """

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': TaggedSentences,
        },
    )
    # Use the response as a JSON string.
    print(response.text)

    # Use instantiated objects.
    res: TaggedSentences = response.parsed
    return res

def segment_sentences_with_llm_api(
    sentences_to_segment: List[str],
    few_shot_examples: List[dict],
    system_instruction: str,
    api_key_val: str,
    model_name_val: str
) -> Optional[SegmenterLLMResponse]:
    if not sentences_to_segment:
        return SegmenterLLMResponse(segmented_sentences=[])

    # Create prompt with examples
    user_prompt_parts = ["Here are some examples of tokenization according to UD guidelines:"]
    for ex in few_shot_examples:
        user_prompt_parts.append(f"Original: {ex['original']}")
        user_prompt_parts.append(f"Tokenized: {ex['tokenized']}")
        user_prompt_parts.append("")

    user_prompt_parts.append("Please tokenize the following sentences according to the same guidelines. For each sentence, provide its original text and the list of segmented tokens.")
    
    # Prepare sentences for processing
    text_for_llm_processing = ""
    for sent_text in sentences_to_segment:
        text_for_llm_processing += f"Original: {sent_text}\n"

    # Full prompt with clear JSON instructions
    prompt_to_llm = f"""{system_instruction}
                    {'\n'.join(user_prompt_parts)}
                    
                    You must provide the output as a single JSON object strictly following this schema:
                    {{
                    "segmented_sentences": [
                        {{
                        "original_text": "The first sentence you processed.",
                        "tokens": ["The", "first", "sentence", "you", "processed", "."]
                        }},
                        {{
                        "original_text": "Another example sentence.",
                        "tokens": ["Another", "example", "sentence", "."]
                        }}
                    ]
                    }}
                    
                    Now, process the following text block and produce the JSON output:
                    {text_for_llm_processing}
                    """
    try:
        client = genai.Client(api_key=api_key_val)
        response = client.models.generate_content(
            model=model_name_val,
            contents=prompt_to_llm,
            generation_config=genai.types.GenerationConfig(
                response_mime_type='application/json',
                response_schema=SegmenterLLMResponse,
            )
        )
        
        # For debugging - add this line to see raw response
        print("Raw response from LLM:", response.text[:200], "...")
        
        if response.parsed:
            return response.parsed
        else:
            # Manual parsing fallback if structured parsing fails
            try:
                text = response.text
                # Handle if response is wrapped in markdown code blocks
                if "```json" in text:
                    import re
                    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
                    if json_match:
                        text = json_match.group(1)
                elif "```" in text:
                    import re
                    json_match = re.search(r"```\s*([\s\S]*?)\s*```", text)
                    if json_match:
                        text = json_match.group(1)
                
                import json
                parsed_json = json.loads(text)
                
                # Create a proper SegmenterLLMResponse object
                segmented_sentences = []
                for sent in parsed_json.get("segmented_sentences", []):
                    segmented_sentences.append(
                        SegmentedTokenOutput(
                            original_text=sent.get("original_text", ""),
                            tokens=sent.get("tokens", [])
                        )
                    )
                
                return SegmenterLLMResponse(segmented_sentences=segmented_sentences)
            except Exception as e:
                print(f"Manual parsing also failed: {e}")
                print(f"Response text sample: {response.text[:200]}")
                return None

    except Exception as e:
        print(f"Error during LLM segmentation API call: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_segmenter_prompt():
    """Create a system prompt for the segmenter"""
    return """You are an expert computational linguistics segmenter specializing in tokenization for the Universal Dependencies framework.

Your task is to tokenize sentences according to the Universal Dependencies guidelines for English:

1. Split punctuation from words: "word." → "word ."
2. Split contractions: "don't" → "do n't", "I'll" → "I 'll", "she's" → "she 's"
3. Split hyphens in some cases:
   a. Split modifiers: "full-fledged" → "full - fledged"
   b. Split bracketed expressions: "(and now)" → "( and now )"
4. Keep some compound terms intact:
   a. Keep emails and URLs as single tokens
   b. Keep numbers with decimal points together: "3.14"
   c. Keep dates together: "2022-03-15"

You must respond with a JSON object containing the tokenized version of each input sentence.
The JSON MUST follow this exact format with no additional text:
{
  "segmented_sentences": [
    {
      "original_text": "Original sentence 1.",
      "tokens": ["Original", "sentence", "1", "."]
    },
    {
      "original_text": "Original sentence 2.",
      "tokens": ["Original", "sentence", "2", "."]
    }
  ]
}
"""
# Update your `call_segmenter` to use this
def call_segmenter(sentences: List[str], examples: List[dict], api_key_val: str, model_name_val: str) -> Optional[SegmenterLLMResponse]:
    # First validate the API credentials
    if not api_key_val or api_key_val == "YOUR_API_KEY":
        print("Error: Invalid API key provided")
        return None
        
    if not model_name_val:
        print("Error: Invalid model name provided")
        return None
    
    system_prompt = create_segmenter_prompt()
    
    try:
        segmented_data = segment_sentences_with_llm_api(
            sentences, examples, system_prompt, api_key_val, model_name_val
        )
        
        if not segmented_data or not segmented_data.segmented_sentences:
            print("Warning: No valid segmentation data returned. Using fallback tokenization.")
            # Simple fallback tokenization as a last resort
            import re
            segmented_sentences = []
            for sentence in sentences:
                # Basic tokenizer as fallback
                tokens = re.findall(r'\b\w+\b|[^\w\s]', sentence)
                segmented_sentences.append(
                    SegmentedTokenOutput(
                        original_text=sentence,
                        tokens=tokens
                    )
                )
            return SegmenterLLMResponse(segmented_sentences=segmented_sentences)
        
        return segmented_data
    except Exception as e:
        print(f"Error in call_segmenter: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    example_text = "The quick brown fox jumps 'over' the lazy dog."
    #example_text = "What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?"
    example_text2 = "Google Search is a web search engine developed by Google LLC."
    # example_text = "החתול המהיר קופץ מעל הכלב העצלן." # Example in Hebrew

    print(f"\nTagging text: \"{example_text}\"")

    tagged_result = tag_sentences_ud(example_text + example_text2)

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
