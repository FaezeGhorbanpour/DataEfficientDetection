
def general_prompt(text):
    prompt = (
        f"Is this comment hate or non-hate? (answer with hate or non-hate)\n"
        f"Comment: \"{text}\"\n"
    )
    return prompt

def definition_prompt(text):
    prompt = (
        f"Hate speech is speech that attacks a person or group based on attributes such as race, religion, ethnic origin, national origin, sex, disability, sexual orientation, or gender identity.\n\n"
        f"Is this comment hate or non-hate? (answer with hate or non-hate)\n"
        f"Comment: \"{text}\"\n"
    )
    return prompt

def chain_of_thought_prompt(text):
    prompt = (
        f"Analyze the following comment to determine if it contains hate speech.\n\n"
        f"Comment: \"{text}\"\n\n"
        f"Step 1: Identify any offensive or harmful language targeting a person or group.\n"
        f"Step 2: Determine if the language attacks based on attributes such as race, religion, ethnic origin, national origin, sex, disability, sexual orientation, or gender identity.\n"
        f"Step 3: Conclude whether it is hate speech.\n\n"
        f"Answer (hate or non-hate):\n"
    )
    return prompt


def few_shot_prompt(text, examples):
    prompt = (
        f"Classify the following text as hate speech or not hate speech. Here are some examples:\n\n"
    )
    for example in examples:
        prompt += f"Text: \"{example['text']}\"\nClassification: {example['label']}\n\n"
    prompt += f"Now, analyze the following text:\n\nText: \"{text}\"\nClassification:"
    return prompt

def multilingual_prompt(text, language):
    prompt = (
        f"Analyze the following text written in {language} and determine if it contains hate speech. "
        f"Consider cultural nuances and context specific to {language}-speaking communities.\n\n"
        f"Text: \"{text}\"\n\n"
        f"Does this text contain hate speech? Answer 'Yes' or 'No'."
    )
    return prompt

def zero_shot_nli_prompt(text):
    prompt = (
        f"Determine if the following statement is an example of hate speech:\n\n"
        f"Text: \"{text}\"\n\n"
        f"Answer with hate vs non-hate\n"
        f"Select the most appropriate option."
    )
    return prompt


def multilingual_chain_of_thought_prompt(text, language):
    prompt = (
        f"Analyze the following text written in {language} to determine if it contains hate speech. "
        f"Consider cultural nuances and context specific to {language}-speaking communities.\n\n"
        f"Text: \"{text}\"\n\n"
        f"Step 1: Identify any offensive or harmful language targeting a person or group.\n"
        f"Step 2: Determine if the language attacks based on attributes such as race, religion, ethnic origin, national origin, sex, disability, sexual orientation, or gender identity.\n"
        f"Step 3: Conclude whether it is hate speech.\n\n"
        f"Answer (hate or non-hate):\n"
    )
    return prompt

def context_aware_prompt(text, context):
    prompt = (
        f"Given the following context: \"{context}\", analyze the subsequent text to determine if it contains hate speech.\n\n"
        f"Text: \"{text}\"\n\n"
        f"Does this text contain hate speech? Answer 'Yes' or 'No'."
    )
    return prompt

def role_play_prompt(text):
    prompt = (
        f"Imagine you are a moderator for an online community. Your task is to ensure that all content adheres to community guidelines, "
        f"which prohibit hate speech. Analyze the following text and determine if it violates these guidelines.\n\n"
        f"Text: \"{text}\"\n\n"
        f"Does this text violate the community guidelines on hate speech? Answer 'Yes' or 'No'."
    )
    return prompt

def comparative_prompt(text, non_hate_example):
    prompt = (
        f"Compare the following two texts and determine which one contains hate speech.\n\n"
        f"Text A: \"{text}\"\n"
        f"Text B: \"{non_hate_example}\"\n\n"
        f"Which text contains hate speech? Answer 'A', 'B', or 'Neither'."
    )
    return prompt

# Here is a social media post: [POST]. Classify this post
# as “hateful” or “non-hateful”. Only return the
# predicted class.