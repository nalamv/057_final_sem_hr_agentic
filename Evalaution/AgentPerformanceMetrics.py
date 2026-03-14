import json
import openai
from src import PolicyAgent


def evaluate_agent():
    with open("RAG_QA_Golden_Dataset.json", "r") as f:
        goldens = json.load(f)

    results = []

    for item in goldens:
        # 1. Get the Agent's real answer
        print(f"Running policy Agent to verify question '{item['question']}'")
        actual_output = PolicyAgent.policy_agent001(item['question'])

        # 2. Ask the "Judge" to grade it
        grade_prompt = f"""
        Question: {item['question']}
        Expected Answer: {item['expected_answer']}
        Actual Agent Answer: {actual_output.content}

        Score the Agent from 1 to 5:
        1: Completely wrong or hallucinated.
        3: Mostly correct but missing details.
        5: Perfect match in facts and tone.
        Return ONLY the number.
        """

        score_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": grade_prompt}]
        )
        score = int(score_response.choices[0].message.content.strip())
        print(f'Score:{score}')
        results.append({
            "question": item['question'],
            "score": score,
            "pass": score >= 4
        })

    # Calculate final success metric
    success_rate = (sum(1 for r in results if r['pass']) / len(results)) * 100
    print(f"Agent Evaluation Complete! Success Rate: {success_rate}%")
    return results

print(evaluate_agent())

# def my_ai_agent(q): ... your agent logic here ...