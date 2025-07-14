import json
import csv

# Your two JSON file paths
json_files = ["Data/test/test_augmented.json", "Data/test/test_human.json"]

# Define keyword lists
math_keywords = ["sum", "min", "max", "total", "difference", "average", "mean", "increase", "decrease", 
                 "more than", "less than", "range", "how many more"]
visual_keywords = ["color", "height", "tallest", "shortest", "length", "bar", "line", 
                   "slice", "segment"]

# Containers for filtered questions
math_questions = []
visual_questions = []

# Process each JSON file
for filename in json_files:
    with open(filename, "r") as f:
        data = json.load(f)

    for item in data:
        query = item["query"].lower()
        label = item["label"]
        imgname = item["imgname"]

        # Check math reasoning: at least two math keywords
        math_count = sum(kw in query for kw in math_keywords)
        if math_count >= 2:
            math_questions.append((query, label, imgname))

        # Check visual reasoning
        if any(kw in query for kw in visual_keywords):
            visual_questions.append((query, label, imgname))

# Save math questions to CSV
with open("math_questions.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Input", "Output", "ImageName"])
    writer.writerows(math_questions)

# Save visual questions to CSV
with open("visual_questions.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Input", "Output", "ImageName"])
    writer.writerows(visual_questions)
