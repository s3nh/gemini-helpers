from google.cloud import aiplatform
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Assume you have your intents defined in a list called 'intents'

def recognize_intent(user_input, intents, embedding_model_name="textembedding-gecko@001", project_id="your-project-id", location="us-central1"):
    """Recognizes the intent of a user input using semantic similarity."""

    aiplatform.init(project=project_id, location=location)
    embedding_model = aiplatform.TextEmbeddingModel.from_pretrained(embedding_model_name)

    user_input_embedding = embedding_model.get_embeddings([user_input])[0].values

    best_intent = None
    best_similarity = -1

    for intent in intents:
        for training_phrase in intent.training_phrases:
            training_phrase_embedding = embedding_model.get_embeddings([training_phrase])[0].values
            similarity = cosine_similarity([user_input_embedding], [training_phrase_embedding])[0][0]

            if similarity > best_similarity:
                best_similarity = similarity
                best_intent = intent

    if best_intent is None or best_similarity < 0.7:  # Adjust the threshold as needed
        return None, {}  # No intent matched

    return best_intent, {} # Return the best intent and parameters

def extract_parameters_with_llm(user_input, intent, project_id="your-project-id", location="us-central1", llm_model_name="text-bison@001"):
    """Extracts parameters from user input using a Large Language Model."""

    aiplatform.init(project=project_id, location=location)
    llm_model = aiplatform.TextGenerationModel.from_pretrained(llm_model_name)

    # Craft a prompt to instruct the LLM to extract the parameters
    parameter_names = ", ".join(intent.parameters.keys())
    prompt = f"""Extract the following parameters from the user input: {parameter_names}.

    User Input: {user_input}
    Parameters:""" # Consider adding examples for few-shot learning

    response = llm_model.predict(prompt=prompt)

    # Parse the LLM's response to extract the parameter values
    # (This will depend on the format of the LLM's response)
    parameters = {}
    # Add your parsing logic here

    return parameters

def fulfill_intent(intent, parameters):
  """Fulfills the given intent with the extracted parameters."""
  return intent.fulfillment_function(**parameters)


# Example Intent Definitions
def get_weather(city):
    # Call a weather API or perform some other action
    return f"The weather in {city} is sunny."

def order_pizza(pizza_type, size):
    # Place a pizza order
    return f"You have ordered a {size} {pizza_type} pizza."


intents = [
    IntentDefinition(
        name="GetWeather",
        training_phrases=["What's the weather like in {city}?", "Tell me the weather in {city}"],
        parameters={"city": "string"},
        fulfillment_function=get_weather,
        response_template="The weather is {weather} in {city}."
    ),
    IntentDefinition(
        name="OrderPizza",
        training_phrases=["I want to order a pizza", "Order a {pizza_type} pizza", "I'd like a {size} pizza"],
        parameters={"pizza_type": "string", "size": "string"},
        fulfillment_function=order_pizza,
        response_template="OK. I ordered {size} {pizza_type} pizza for you."
    )
]

# Example Usage
user_input = "What's the weather like in London?"
recognized_intent, _ = recognize_intent(user_input, intents)

if recognized_intent:
  parameters = extract_parameters_with_llm(user_input, recognized_intent)
  fulfillment_result = fulfill_intent(recognized_intent, parameters)
  print(fulfillment_result)
else:
  print("Sorry, I didn't understand your request.")
