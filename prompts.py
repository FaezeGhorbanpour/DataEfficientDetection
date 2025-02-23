import random

translations = {
    "en": {
        "comment": "Comment",
        "yes": "yes",
        "hateful": "hateful",
        "no": "no",
        "non_hateful": "non-hateful",
        "general": "Is this comment hateful? (don't explain and only answer with {yn1} or {yn2})\nComment: \"{text}\"\nAnswer:",
        "classification": "Here is a social media post: {text}. Classify this post as “yes” if it is hateful or “no” if it is not hateful. (don't explain and only answer with {yn1} or {yn2})\nAnswer:",
        "definition": "Hate speech is speech that attacks a person or group based on attributes such as race, religion, ethnic origin, national origin, sex, disability, sexual orientation, or gender identity.\n\n"
                      "Is this comment hateful? (don't explain and only answer with {yn1} or {yn2})\nComment: \"{text}\"\nAnswer:",
        "cot": "Analyze the following comment to determine if it contains hate speech.\n\n"
               "Comment: \"{text}\"\n\n"
               "Step 1: Identify any offensive or harmful language targeting a person or group.\n"
               "Step 2: Determine if the language attacks based on attributes such as race, religion, ethnic origin, national origin, sex, disability, sexual orientation, or gender identity.\n"
               "Step 3: Conclude whether it is hate speech.\n"
               "(don't explain and only answer with {yn1} or {yn2}).\nAnswer:",
        "few_shot": "Classify the following text as hate speech or not hate speech. Here are some examples:\n\n{examples}\n"
                    "Now, classify the following text:\n\nText: \"{text}\"\nAnswer:",
        "multilingual": "Analyze the following text written in {language} and determine if it contains hate speech. "
                        "Consider cultural nuances and context specific to {language}-speaking communities.\n\n"
                        "Text: \"{text}\"\n\n"
                        "Does this text contain hate speech? (don't explain and only answer with {yn1} or {yn2}).\nAnswer:",
        "nli": "Premise: \"{text}\"\nHypothesis: \"This statement expresses hate speech.\"\nDoes the premise entail the hypothesis? (Only don't explain and only answer with {yn1} or {yn2})\nAnswer:",
        "multilingual_cot": "Analyze the following text written in {language} to determine if it contains hate speech. "
                            "Consider cultural nuances and context specific to {language}-speaking communities.\n\n"
                            "Text: \"{text}\"\n\n"
                            "Step 1: Identify any offensive or harmful language targeting a person or group.\n"
                            "Step 2: Determine if the language attacks based on attributes such as race, religion, ethnic origin, national origin, sex, disability, sexual orientation, or gender identity.\n"
                            "Step 3: Conclude whether it is hate speech.\n\n"
                            "(don't explain and only answer with {yn1} or {yn2})\nAnswer:",
        "role_play": "Imagine you are a moderator for an online community. Your task is to ensure that all content adheres to community guidelines, "
                     "which prohibit hate speech. Analyze the following text and determine if it violates these guidelines.\n\n"
                     "Text: \"{text}\"\n\n"
                     "Does this text violate the community guidelines on hate speech? (don't explain and only answer with {yn1} or {yn2}).\nAnswer:"
    },

}

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
def get_random_yes_no():
    return ("yes", "no") if random.choice([True, False]) else ("no", "yes")

def general_prompt(text, translate_to="en"):
    yn1, yn2 = get_random_yes_no()
    return translations.get(translate_to, translations["en"])["general"].format(text=text, yn1=yn1, yn2=yn2)

def classification_prompt(text, translate_to="en"):
    yn1, yn2 = get_random_yes_no()
    return translations.get(translate_to, translations["en"])["classification"].format(text=text, yn1=yn1, yn2=yn2)

def definition_prompt(text, translate_to="en"):
    yn1, yn2 = get_random_yes_no()
    return translations.get(translate_to, translations["en"])["definition"].format(text=text, yn1=yn1, yn2=yn2)

def chain_of_thought_prompt(text, translate_to="en"):
    yn1, yn2 = get_random_yes_no()
    return translations.get(translate_to, translations["en"])["cot"].format(text=text, yn1=yn1, yn2=yn2)

def few_shot_prompt(text, examples, translate_to="en"):
    yn1, yn2 = get_random_yes_no()
    formatted_examples = "\n".join([f"Texto: \"{ex['text']}\"{ex['label']}" for ex in examples])
    return translations.get(translate_to, translations["en"])["few_shot"].format(examples=formatted_examples, text=text, yn1=yn1, yn2=yn2)

def multilingual_prompt(text, language, translate_to="en"):
    yn1, yn2 = get_random_yes_no()
    return translations.get(translate_to, translations["en"])["multilingual"].format(language=LANGUAGES[language], text=text, yn1=yn1, yn2=yn2)

def nli_prompt(text, translate_to="en"):
    yn1, yn2 = get_random_yes_no()
    return translations.get(translate_to, translations["en"])["nli"].format(text=text, yn1=yn1, yn2=yn2)

def multilingual_chain_of_thought_prompt(text, language, translate_to="en"):
    yn1, yn2 = get_random_yes_no()
    return translations.get(translate_to, translations["en"])["multilingual_cot"].format(language=LANGUAGES[language], text=text, yn1=yn1, yn2=yn2)

def role_play_prompt(text, translate_to="en"):
    yn1, yn2 = get_random_yes_no()
    return translations.get(translate_to, translations["en"])["role_play"].format(text=text, yn1=yn1, yn2=yn2)

def clean_output(input):
    output = input.strip().lower()
    output = output.split("Answer:")[-1] # remove everything before "Answer:"
    output = output.split("answer:")[-1] # remove everything before "Answer:"
    output = output.split("</s>")[0] # remove </s> and everything after
    output = output.split(',')[0] # remove quotes, and stuff like "I hope this answer helped!"
    output = output.split('.')[0]
    output = output.split('the')[0]
    output = output.split('human')[0]
    output = output.split('comment')[0]
    output = output.replace('[/inst]', '')
    output = output.replace('[inst]', '')
    output = output.replace('assistant:', '')
    output = output.replace('analysis:', '')
    output = output.replace('note:', '')
    output = output.replace('context:', '')
    output = output.replace('**', '')
    output = output.replace('non-hateful', 'no')
    output = output.replace('hateful', 'yes')
    output = output.replace('"', '')
    output = output.strip()
    output = output.split("\n")[0]
    output = output.split(" ")[0]
    return output

def map_output(response, translate_to="en"):
    """Maps model response to binary labels (0 = non-hate, 1 = hate)"""
    t = translations.get(translate_to, translations["en"])
    response_cleaned = clean_output(response)
    response_lower = response_cleaned.strip().lower()
    if response_lower == t["yes"] or response_lower == t["hateful"]:
        return 1
    elif response_lower == t["no"] or response_lower == t["non_hateful"]:
        return 0
    else:
        print("invalid prediction:", response)
        return -1  # Handle uncertain cases

# Example Usage
# text_sample = "This is a test comment"
# print(general_prompt(text_sample, "es"))  # Spanish
# print(chain_of_thought_prompt(text_sample, "es"))  # Spanish
#
# # Example of mapping output
# print(map_output("odioso", "es"))  # Should return 1
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

