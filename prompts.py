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

def clean_output(input):
    output = input.strip().lower()
    output = output.replace('Based on the given text, it', '')
    output = output.split("</s>")[0] # remove </s> and everything after
    output = output.split(',')[0] # remove quotes, and stuff like "I hope this answer helped!"
    output = output.split('.')[0]
    output = output.split('।')[0]
    output = output.split('|')[0]
    output = output.split('human')[0]
    output = output.split('comment')[0]
    output = output.replace('[/inst]', '')
    output = output.replace('[inst]', '')
    output = output.replace('assistant:', '')
    output = output.replace('nohuman:', '')
    output = output.replace('analysis:', '')
    output = output.replace('note:', '')
    output = output.replace('context:', '')
    output = output.replace('content:', '')
    output = output.replace('role:', '')
    output = output.replace('system', '')
    output = output.replace('user', '')
    output = output.replace('the post is', '')
    output = output.replace('the comment is', '')
    output = output.replace('el comentario es', '')
    output = output.replace('der Kommentar ist', '')
    output = output.replace('o comentário é', '')
    output = output.replace('yorum şu şekildedir', '')
    output = output.replace('le commentaire est', '')
    output = output.replace('il commento è', '')
    output = output.replace('टिप्पणी है', '')
    output = output.replace('التعليق هو', '')
    output = output.replace('the text is', '')
    output = output.replace('the text', '')
    output = output.replace('the', '')
    output = output.replace('does not contain', 'no')
    output = output.replace('contains', 'yes')

    # Spanish
    output = output.replace('el texto es', '')
    output = output.replace('el texto', '')
    output = output.replace('no contiene', 'no')
    output = output.replace('contiene', 'sí')

    # Portuguese
    output = output.replace('o texto é', '')
    output = output.replace('o texto', '')
    output = output.replace('não contém', 'não')
    output = output.replace('contém', 'sim')

    # Italian
    output = output.replace('il testo è', '')
    output = output.replace('il testo', '')
    output = output.replace('non contiene', 'no')
    output = output.replace('contiene', 'sì')

    # French
    output = output.replace('le texte est', '')
    output = output.replace('le texte', '')
    output = output.replace('ne contient pas', 'non')
    output = output.replace('contient', 'oui')

    # German
    output = output.replace('der Text ist', '')
    output = output.replace('der Text', '')
    output = output.replace('enthält nicht', 'nein')
    output = output.replace('enthält', 'ja')

    # Hindi
    output = output.replace('पाठ है', '')
    output = output.replace('पाठ', '')
    output = output.replace('शामिल नहीं है', 'नहीं')
    output = output.replace('शामिल है', 'हां')

    # Arabic
    output = output.replace('النص هو', '')
    output = output.replace('النص', '')
    output = output.replace('لا يحتوي على', 'لا')
    output = output.replace('يحتوي على', 'نعم')

    # Turkish
    output = output.replace('metin şudur', '')
    output = output.replace('metin', '')
    output = output.replace('içermez', 'hayır')
    output = output.replace('içerir', 'evet')

    output = output.replace('**', '')
    output = output.replace(':', '')
    output = output.replace('"', '')
    output = output.strip()
    output = output.split("\n")[0]
    output = output.split(" ")[0]
    return output

def map_output(response, translate_to="en"):
    """Maps model response to binary labels (0 = non-hate, 1 = hate)"""
    t = translations.get(translate_to, translations["en"])
    response = response.split(t['answer'])[-1] # remove everything before "Answer:"
    response = response.split(t['answer'].lower())[-1] # remove everything before "Answer:"
    response = response.replace(t["non_hateful"], t["no"])
    response = response.replace(t["hateful"], t["yes"])

    response_cleaned = clean_output(response)
    response_lower = response_cleaned.strip().lower()

    if response_lower == t["yes"] or response_lower == "yes":
        return 1
    elif response_lower == t["no"] or response_lower == "no":
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

text = ''' 


**Analysis:**

The text contains hate speech'''
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

