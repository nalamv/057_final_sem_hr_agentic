import json
import openai  # or your preferred LLM provider
from tika import parser

def generate_golden_dataset(context_description, num_questions=100):
    dataset = []
    # We generate in batches to maintain high quality and avoid context limits
    categories = ["Happy Path", "Edge Case", "Negative/Safety", "Ambiguous"]

    for category in categories:
        prompt = f"""
        You are a QA Engineer. Based on this business: {context_description}
        Generate {num_questions // 4} unique Question & Answer pairs for the category: {category}.

        Format as a JSON list:
        [
          {{"question": "string", "expected_answer": "string", "category": "{category}"}}
        ]
        """

        print(prompt)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        batch = json.loads(response.choices[0].message.content)
        dataset.append(batch)

    with open("RAG_QA_Golden_Dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    return dataset


#start tika sever http:localhost:9998
#
tika_server='http://localhost:9998'
file_pdf= "C:/Users/yugan/Desktop/VNIT Mtech/GDrive/SEM-3/NLP/FinalProject/testdata/data/pdf/Leave-and-Holiday-Policy.pdf"
text=parser.from_file(file_pdf,tika_server)
generate_golden_dataset(text["content"],20)

# Example Usage:
# generate_golden_dataset("A fintech app for international wire transfers.")