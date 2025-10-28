import json
import re

def extract_number(text):
    """
    Trích xuất số cuối cùng trong đoạn văn bản.
    Ví dụ: "#### 18" -> 18
    """
    if not text:
        return None
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    return float(numbers[-1]) if numbers else None

def evaluate_accuracy(file_path):
    total, correct = 0, 0
    wrong_cases = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            model_ans = extract_number(data.get("model_answer", ""))
            ground_truth = extract_number(data.get("ground_truth_answer", ""))

            if ground_truth is not None and model_ans is not None:
                total += 1
                if abs(model_ans - ground_truth) < 1e-6:  # So sánh số học
                    correct += 1
                else:
                    wrong_cases.append({
                        "question": data["question"],
                        "predicted": model_ans,
                        "expected": ground_truth
                    })

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Total questions: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

# Gọi hàm với file của mày
evaluate_accuracy("base_model_results.jsonl")
