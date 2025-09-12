SAMPLE_GENERATION_PROMPT = """I want to train a large language model and you should help me generate training data for it. Here is the system prompt of the model that tells it what it should be able to do:

<system_prompt>
{{{{system_prompt}}}}
</system_prompt>

You should now generate three training samples for the model. Each training sample should consist of a JSON object with the field "messages", which is a list of messages alternating between user and assistant roles. The first message must always be from the user, and the last one from the assistant. Depending on the use case of the system prompt, there may be multiple user and assistant messages. The format for each training sample must strictly follow this format:

{
    "messages": [
        {
            "role": "user",
            "content": "<user_content>"
        },
        {
            "role": "assistant",
            "content": "<assistant_content>"
        }
    ]
}

It is crucial that you respond only with valid JSON. Do not include any introductions, explanations, summaries, or additional text that is not part of the JSON object. Any non-JSON content will be considered incorrect. If you encounter issues generating valid JSON, please retry or provide a default response.

Here are additional inputs to guide you:

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}

Now, generate a single training sample in the JSON format specified above. Respond only with valid JSON."""

TREE_GENERATION_PROMPT = """I want to train a large language model and I am using another, bigger large language model to generate training data for this. However, if we always ask the bigger model to generate training data with the same prompt, it will end up generating very repetitive training samples. Therefore, we will slightly modify our prompt for each sampling procedure according to some aspects. For instance, when asking the model to generate news articles, we could modify the prompt to let the model tell news articles about particular topics, such as business or politics. To further generate training data, we will do this recursively, and generate submodifications to the prompt. For instance, within the domain of business, we could adapt the prompt to generate news about the stock market or business scandals, and within politics, we could ask the model to generate articles for subtopics like elections or climate policy. We do this recursively, and therefore, we get a tree-like structure of topics.
Your job is the following: I will give you a path of nodes down the topic tree - you should then come up with a list of new subtopics for this given node and return it as a python list. Here are a few examples of what your outputs should look like, related to the news example I just gave you:

Example 1:
node path: "News Topics" -> "Sports" -> "Football"
desired number of subtopics: 5
subtopics: ["college football", "football stadiums", "health consequences football", "Seattle Seahawks", "football sponsorships"]


Example 2:
node path: "News Topics" -> "Entertainment" -> "Movies" -> "Star Portraits"
desired number of subtopics: 8
subtopics: ["Tom Hanks", "Meryl Streep", "Leonardo DiCaprio", "Jennifer Lawrence", "Denzel Washington", "Charlize Theron", "Robert Downey Jr.", "Emma Stone"]


Here are three new examples, this time for generating smalltalk topics for a friendly chat assistant:

Example 1:
node path: "Small Talk Topics"
desired number of subtopics: 7
subtopics: ["weather", "weekend plans", "hobbies", "family", "books", "food", "music"]

Example 2:
node path: "Small Talk Topics" -> "Family"
desired number of subtopics: 5
subtopics: ["parents", "grandparents", "siblings", "family traditions", "family vacations"]

Example 3:
node path: "Small Talk Topics" -> "Hobbies" -> "Cooking"
desired number of subtopics: 6
subtopics: ["recipes", "asian food", "favourite dishes", "cookbooks", "kitchen gadgets", "vegan cooking"]


Here is a description / the system prompt for the model we want to train:

<system_prompt>
{{{{system_prompt}}}}
</system_prompt>


Here is your topic input. When generating subtopics, remain somewhat vague. Things can only be tangentially related and they don't have to be interpreted in a single way. Importantly, make sure that the subtopics fit the system prompt, if one was supplied:
node path: {{{{subtopics_list}}}}
desired number of subtopics: {{{{num_subtopics}}}}

Now return the subtopics as a python list, and return it in just one line, not multiple ones. Don't return anything else."""

TREE_JSON_INSTRUCTIONS = """When listing subtopics, format your response as a valid JSON array of strings.
Example: ["topic 1", "topic 2", "topic 3"]
1. Use double quotes for strings
2. Use square brackets for the array
3. Separate items with commas
4. Do not include any text before or after the JSON array
5. Ensure all JSON syntax is valid
"""

OLD_ENGINE_JSON_INSTRUCTIONS = """Your response **must be valid JSON** that can be parsed by `json.loads()`. Follow these rules precisely:

1. **Double Quotes Only**: Use double quotes (`"`) around all string values, including keys.
2. **No Extra Text**: Do not include any text before or after the JSON block. Ensure the output is **only JSON**.
3. **Valid Syntax**: Check that all JSON syntax is correct:
   - Every key-value pair should be separated by a colon.
   - Separate each item in an array or object with a comma, except for the last item.
4. **No Trailing Commas**: Ensure there are no trailing commas in arrays or objects.
5. **Number Formatting**: Ensure numbers are formatted correctly (e.g., no leading zeroes unless the number is decimal).
6. **Boolean & Null Values**: Use lowercase `true`, `false`, and `null` as valid JSON values.
7. **Final Validation**: Your response will be parsed as JSON. Any syntax errors will cause a failure, so check carefully.

**Important**: The entire response must be **valid JSON**, with no explanations, comments, or text outside of the JSON structure.
"""

ENGINE_JSON_INSTRUCTIONS = """You are an expert JSON builder designed to assist with a wide range of tasks.

Your response **must be valid JSON** that can be parsed by `json.loads()`. Follow these rules precisely:

1. **Double Quotes Only**: Use double quotes (`"`) around all string values, including keys.
2. **No Extra Text**: Do not include any text before or after the JSON block. Ensure the output is **only JSON**.
3. **Valid Syntax**: Check that all JSON syntax is correct:
   - Every key-value pair should be separated by a colon.
   - Separate each item in an array or object with a comma, except for the last item.
4. **No Trailing Commas**: Ensure there are no trailing commas in arrays or objects.
5. **Number Formatting**: Ensure numbers are formatted correctly (e.g., no leading zeroes unless the number is decimal).
6. **Boolean & Null Values**: Use lowercase `true`, `false`, and `null` as valid JSON values.
7. **Final Validation**: Your response will be parsed as JSON. Any syntax errors will cause a failure, so check carefully.

**Important**: The entire response must be **valid JSON**, with no explanations, comments, or text outside of the JSON structure.

**JSON Structure**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "<user_message>"
    },
    {
      "role": "assistant",
      "content": "<assistant_response>"
    }
  ]
}
```

**JSON Examples**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hey, how are you today?"
    },
    {
      "role": "assistant",
      "content": "I'm good thanks, how are you?"
    }
  ]
},
{
  "messages": [
    {
      "role": "user",
      "content": "What color is the sky?"
    },
    {
      "role": "assistant",
      "content": "The sky is blue."
    }
  ]
}
```

All of Assistant's communication is performed using this JSON format.
"""
