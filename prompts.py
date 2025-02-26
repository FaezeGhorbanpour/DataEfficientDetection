import random
import json

translations = json.load(open("translations.json"))

LANGUAGES = {
    'es': 'Spanish',
    'en': 'English',
    'fr': 'French',
    'it': 'Italian',
    'pt': 'Portuguese',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'ar': 'Arabic',
    'de': 'German'
}
def get_random_yes_no(yes='yes', no='no'):
    return (yes, no) if random.choice([True, False]) else (no, yes)

def general_prompt(text, translate_to="en"):
    yn1, yn2 = get_random_yes_no(yes=translations.get(translate_to, translations["en"])["yes"], 
                                 no=translations.get(translate_to, translations["en"])["no"])
    return translations.get(translate_to, translations["en"])["general"].format(text=text, yn1=yn1, yn2=yn2)

def classification_prompt(text, translate_to="en"):
    yn1, yn2 = get_random_yes_no(yes=translations.get(translate_to, translations["en"])["yes"], 
                                 no=translations.get(translate_to, translations["en"])["no"])
    return translations.get(translate_to, translations["en"])["classification"].format(text=text, yn1=yn1, yn2=yn2)

def definition_prompt(text, translate_to="en"):
    yn1, yn2 = get_random_yes_no(yes=translations.get(translate_to, translations["en"])["yes"], 
                                 no=translations.get(translate_to, translations["en"])["no"])
    return translations.get(translate_to, translations["en"])["definition"].format(text=text, yn1=yn1, yn2=yn2)

def chain_of_thought_prompt(text, translate_to="en"):
    yn1, yn2 = get_random_yes_no(yes=translations.get(translate_to, translations["en"])["yes"], 
                                 no=translations.get(translate_to, translations["en"])["no"])
    return translations.get(translate_to, translations["en"])["cot"].format(text=text, yn1=yn1, yn2=yn2)

def few_shot_prompt(text, examples, translate_to="en"):
    yn1, yn2 = get_random_yes_no(yes=translations.get(translate_to, translations["en"])["yes"], 
                                 no=translations.get(translate_to, translations["en"])["no"])
    formatted_examples = "\n".join([f"Text: \"{ex['text']}\"{ex['label']}" for ex in examples])
    return translations.get(translate_to, translations["en"])["few_shot"].format(examples=formatted_examples,
                                                                                 text=text, yn1=yn1, yn2=yn2)

def multilingual_prompt(text, language, translate_to="en"):
    yn1, yn2 = get_random_yes_no(yes=translations.get(translate_to, translations["en"])["yes"], 
                                 no=translations.get(translate_to, translations["en"])["no"])
    return translations.get(translate_to, translations["en"])["multilingual"].format(language=LANGUAGES[language],
                                                                                     text=text, yn1=yn1, yn2=yn2)

def nli_prompt(text, translate_to="en"):
    yn1, yn2 = get_random_yes_no(yes=translations.get(translate_to, translations["en"])["yes"], 
                                 no=translations.get(translate_to, translations["en"])["no"])
    return translations.get(translate_to, translations["en"])["nli"].format(text=text, yn1=yn1, yn2=yn2)

def multilingual_chain_of_thought_prompt(text, language, translate_to="en"):
    yn1, yn2 = get_random_yes_no(yes=translations.get(translate_to, translations["en"])["yes"], 
                                 no=translations.get(translate_to, translations["en"])["no"])
    return translations.get(translate_to, translations["en"])["multilingual_cot"].format(language=LANGUAGES[language],
                                                                                         text=text, yn1=yn1, yn2=yn2)

def role_play_prompt(text, translate_to="en"):
    yn1, yn2 = get_random_yes_no(yes=translations.get(translate_to, translations["en"])["yes"], 
                                 no=translations.get(translate_to, translations["en"])["no"])
    return translations.get(translate_to, translations["en"])["role_play"].format(text=text, yn1=yn1, yn2=yn2)


def clean_output(input_text):
    """
    Clean and normalize output text by removing various phrases and formatting.

    Args:
        input_text (str): The text to be cleaned

    Returns:
        str: The cleaned text
    """
    # Convert to lowercase and strip whitespace
    output = input_text.strip().lower()

    # Define phrases to remove, grouped by category
    phrases_to_remove = {
        # English phrases
        "based on", "given", "provided", "[/inst]", "human", "comment", "[inst]", "premise", "context", "information",
        "assistant:", "nohuman:", "analysis:", "note:", "context:", "content:", "alone",
        "role:", "system", "user", "the post is", "the comment is", "it", "it is",
        "in this case, the comment is", "the text is", "the text", "the", "text", "comment", "post",

        # Spanish phrases
        "la publicación es", "el comentario es", "en este caso, el comentario es",
        "el texto es", "el texto",

        # Portuguese phrases
        "a postagem é", "o comentário é", "neste caso, o comentário é",
        "o texto é", "o texto",

        # Italian phrases
        "il post è", "il commento è", "in questo caso, il commento è",
        "il testo è", "il testo",

        # French phrases
        "le post est", "le commentaire est", "dans ce cas, le commentaire est",
        "le texte est", "le texte",

        # German phrases
        "der Beitrag ist", "der Kommentar ist", "in diesem Fall ist der Kommentar",
        "der Text ist", "der Text",

        # Hindi phrases
        "पोस्ट है", "टिप्पणी है", "इस मामले में, टिप्पणी है",
        "पाठ है", "पाठ",

        # Arabic phrases
        "المنشور هو", "التعليق هو", "في هذه الحالة، التعليق هو",
        "النص هو", "النص",

        # Turkish phrases
        "gönderi şudur", "yorum şudur", "bu durumda, yorum şudur",
        "metin şudur", "metin",

        # Additional language-specific comment phrases
        "el comentario es", "der Kommentar ist", "o comentário é",
        "yorum şu şekildedir", "le commentaire est", "il commento è",
        "टिप्पणी है", "التعليق هو"
    }

    # Word replacements mapping
    word_replacements = {
        "does not contain": "no",
        "contains": "yes",
        "maybe": "yes",
        "tal vez": "yes",  # Spanish
        "talvez": "yes",  # Portuguese
        "forse": "yes",  # Italian
        "peut-être": "yes",  # French
        "vielleicht": "yes",  # German
        "शायद": "yes",  # Hindi
        "ربما": "yes",  # Arabic
        "belki": "yes",  # Turkish

        # Language-specific content indicators
        "no contiene": "no",  # Spanish
        "contiene": "sí",  # Spanish
        "não contém": "não",  # Portuguese
        "contém": "sim",  # Portuguese
        "non contiene": "no",  # Italian
        "contiene": "sì",  # Italian
        "ne contient pas": "non",  # French
        "contient": "oui",  # French
        "enthält nicht": "nein",  # German
        "enthält": "ja",  # German
        "शामिल नहीं है": "नहीं",  # Hindi
        "शामिल है": "हां",  # Hindi
        "لا يحتوي على": "لا",  # Arabic
        "يحتوي على": "نعم",  # Arabic
        "içermez": "hayır",  # Turkish
        "içerir": "evet"  # Turkish
    }

    # Remove phrases
    for phrase in phrases_to_remove:
        output = output.replace(phrase, '')

    # Apply word replacements
    for old, new in word_replacements.items():
        output = output.replace(old, new)

    # Clean up formatting
    output = output.strip()

    # Remove leading punctuation
    if len(output) > 1 and output[0] in ',.:;':
        output = output[1:].strip()

    # Remove special characters and formatting
    chars_to_remove = "**:\"'"
    for char in chars_to_remove:
        output = output.replace(char, '')

    # Extract only the first part of text before certain delimiters
    delimiters = ["</s>", ",", ".", "।", "|", "\n", " "]
    for delimiter in delimiters:
        if delimiter in output:
            output = output.split(delimiter)[0]

    return output.strip()

def map_output(response, translate_to="en"):
    """Maps model response to binary labels (0 = non-hate, 1 = hate)"""
    t = translations.get(translate_to, translations["en"])
    response = response.split(t['answer'])[-1] # remove everything before "Answer:"
    response = response.split(t['answer'].lower())[-1] # remove everything before "Answer:"
    response = response.replace(t["non_hateful"], t["no"])
    response = response.replace(t["hateful"], t["yes"])

    response_cleaned = clean_output(response)
    response_lower = response_cleaned.strip().lower()

    if response_lower == t["yes"].lower() or response_lower == "yes":
        return 1
    elif response_lower == t["no"].lower() or response_lower == "no":
        return 0
    else:
        print("invalid prediction:", response, response_lower)
        return -1  # Handle uncertain cases

# Example Usage
# text_sample = "This is a test comment"
# print(general_prompt(text_sample, "es"))  # Spanish
# print(chain_of_thought_prompt(text_sample, "es"))  # Spanish
#
# Example of mapping output

text = ''' [/INST] Based on the given text, it does not contain based'''
print(map_output(text, "en"))  # Should return 1
# print(map_output("no odioso", "es"))  # Should return 0



# def context_aware_prompt(text, context):
#     prompt = (
#         f"Given the following context: \"{context}\", analyze the subsequent text to determine if it contains hate speech.\n\n"
#         f"Text: \"{text}\"\n\n"
#         f"Does this text contain hate speech? Answer 'Yes' or 'No'."
#     )
#     return prompt

#
# def comparative_prompt(text, non_hate_example):
#     prompt = (
#         f"Compare the following two texts and determine which one contains hate speech.\n\n"
#         f"Text A: \"{text}\"\n"
#         f"Text B: \"{non_hate_example}\"\n\n"
#         f"Which text contains hate speech? Answer 'A', 'B', or 'Neither'."
#     )
#     return prompt

