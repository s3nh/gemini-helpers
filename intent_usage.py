from google.cloud import aiplatform

def classify_intent(
    project_id: str,
    location: str,
    endpoint_id: str,
    prompt: str,
):
    client = aiplatform.EndpointServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )
    endpoint = client.endpoint_path(
        project=project_id, location=location, endpoint=endpoint_id
    )

    intent_options = ["GetWeather", "BookFlight", "OrderPizza", "Unknown"]

    prompt = f"""Classify the following user input into one of the following intents: {", ".join(intent_options)}.

    User Input: {prompt}
    Intent:"""

    response = client.predict(
        endpoint=endpoint,
        instances=[{"content": prompt}],
        parameters={
            "temperature": 0.2,
            "max_output_tokens": 30,
            "top_p": 0.8,
            "top_k": 40
        },
    )
    predictions = response.predictions
    predicted_intent = predictions[0]["content"].strip()

    if predicted_intent not in intent_options:
        predicted_intent = "Unknown" # Handle cases where the model hallucinates

    return predicted_intent



project_id = "your-project-id"
location = "us-central1"
endpoint_id = "your-llm-endpoint-id"
user_input = "What's the weather like in London?"

intent = classify_intent(project_id, location, endpoint_id, user_input)
print(f"Detected intent: {intent}")
