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
LABEL_MAP={
    1: 'yes',
    0: 'no'
}


def get_random_yes_no(yes='yes', no='no'):
    """Return yes/no options in random order."""
    return (yes, no) if random.choice([True, False]) else (no, yes)


def standard_prompt(text, variant, translate_to="en"):
    """Handle standard prompts with no additional parameters beyond the text."""
    yn1, yn2 = get_random_yes_no(
        yes=translations.get(translate_to, translations["en"])["yes"],
        no=translations.get(translate_to, translations["en"])["no"]
    )
    template = translations.get(translate_to, translations["en"])[variant]

    return template.format(text=text, yn1=yn1, yn2=yn2)


def language_aware_prompt(text, language, variant, translate_to="en"):
    """Handle prompts that need language information."""
    yn1, yn2 = get_random_yes_no(
        yes=translations.get(translate_to, translations["en"])["yes"],
        no=translations.get(translate_to, translations["en"])["no"]
    )

    # Choose the appropriate template based on variant
    template = translations.get(translate_to, translations["en"])[variant]

    return template.format(language=LANGUAGES[language], text=text, yn1=yn1, yn2=yn2)


def example_based_prompt(text, examples, variant, translate_to="en"):
    """Handle prompts that require examples for few-shot learning."""
    yn1, yn2 = get_random_yes_no(
        yes=translations.get(translate_to, translations["en"])["yes"],
        no=translations.get(translate_to, translations["en"])["no"]
    )

    formatted_examples = "\n".join([f"Comment: \"{ex['text']}\t\"Answer: {LABEL_MAP[ex['label']]}" for ex in examples])
    template = translations.get(translate_to, translations["en"])[variant]

    return template.format(examples=formatted_examples, text=text, yn1=yn1, yn2=yn2)

def example_and_language_based_prompt(text, examples, language, variant, translate_to="en"):
    """Handle prompts that require examples for few-shot learning."""
    yn1, yn2 = get_random_yes_no(
        yes=translations.get(translate_to, translations["en"])["yes"],
        no=translations.get(translate_to, translations["en"])["no"]
    )

    formatted_examples = "\n".join([f"Comment: \"{ex['text']}\t\"Answer: {LABEL_MAP[ex['label']]}" for ex in examples])
    template = translations.get(translate_to, translations["en"])[variant]

    return template.format(examples=formatted_examples, language=LANGUAGES[language], text=text, yn1=yn1, yn2=yn2)

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
        "assistant", "nohuman", "analysis", "note", "context:", "content:", "alone", "the comment is",
        "role", "system", "user", "the post is", "the comment is", " it ", "it is", " is ", "**", 'the final answer is',
        "in this case,", "the text is", "the text", "the", "text", "comment", 'answer', "post", 'inst',
        'the answer is', ' ai ', 'final', 'this', 'case', ' in ',

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
        "not yes": "no",
        "not hate speech": "no",
        "contains": "yes",
        "maybe": "yes",
        "hate speech": "yes",
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
        "si": "sí",  # Spanish
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

    for i in ',.:;':
        output = output.replace(i, '')

    # Remove special characters and formatting
    chars_to_remove = "**:\"')([]\\/"
    for char in chars_to_remove:
        output = output.replace(char, '')

    output = output.strip()
    # Extract only the first part of text before certain delimiters
    delimiters = ["</s>", ",", ".", "।", "|", "\n", " "]
    delimiters += ['target', 'explanation']
    for delimiter in delimiters:
        if delimiter in output:
            output = output.split(delimiter)[0]

    return output.strip()

def map_output(response, lang='en', translate_prompt=False):
    """Maps model response to binary labels (0 = non-hate, 1 = hate)"""
    en = translations["en"]
    response = response.split(en['answer'])[-1]  # remove everything before "Answer:"
    response = response.split(en['answer'].lower())[-1]  # remove everything before "Answer:"
    response = response.replace(en["non_hateful"], en["no"])
    response = response.replace(en["hateful"], en["yes"])

    lang = translations.get(lang, translations["en"])
    response = response.split(lang['answer'])[-1]  # remove everything before "Answer:"
    response = response.split(lang['answer'].lower())[-1]  # remove everything before "Answer:"
    response = response.replace(lang["non_hateful"], lang["no"])
    response = response.replace(lang["hateful"], lang["yes"])

    response_cleaned = clean_output(response)
    response_lower = response_cleaned.strip().lower()

    if response_lower == lang["yes"].lower() or response_lower == en["yes"].lower() or response_lower == "yes":
        return 1
    elif response_lower == lang["no"].lower() or response_lower == en["yes"].lower() or response_lower == "no":
        return 0
    else:
        print("invalid prediction:", response, response_lower)
        return -1  # Handle uncertain cases


text = '''[/INST] yes[/ yes/'''
print(map_output(text, "es"))  # Should return 1
# print(map_output("no odioso", "es"))  # Should return 0

