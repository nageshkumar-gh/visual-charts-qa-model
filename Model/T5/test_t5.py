from transformers import T5ForConditionalGeneration, T5Tokenizer

# Path where the model and tokenizer are saved
model_path = "Model/T5/Final"

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Example question
input_text = "question: ? table: Characteristic: League of Legends, Number of hours watched: 512.3; Characteristic: Fortnite, Number of hours watched: 465.0; Characteristic: Just Chatting, Number of hours watched: 372.5; Characteristic: Grand Theft Auto V, Number of hours watched: 269.1; Characteristic: Dota 2, Number of hours watched: 237.1; Characteristic: Apex Legends, Number of hours watched: 181.4; Characteristic: Counter-Strike: Global Offensive, Number of hours watched: 178.0; Characteristic: Overwatch, Number of hours watched: 127.3; Characteristic: Hearthstone, Number of hours watched: 120.7; Characteristic: World of Warcraft, Number of hours watched: 118.5; chart_type: h_bar title:  x_axis_title: ['League of Legends', 'Fortnite', 'Just Chatting', 'Grand Theft Auto V', 'Dota 2', 'Apex Legends', 'Counter-Strike: Global Offensive', 'Overwatch', 'Hearthstone', 'World of Warcraft'] y_axis_title: Number of hours watched"




# Tokenize input
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate output
outputs = model.generate(input_ids, max_length=50)

# Decode and print the output
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Answer:", answer)
