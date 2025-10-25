import json
import re
from tqdm import tqdm

RESULTS_FILE = "dpo_model_results.jsonl"  # File chứa kết quả thô của bạn


def extract_last_number(text):
    """Trích xuất số cuối cùng trong một chuỗi văn bản."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if nums:
        return nums[-1]
    return None


total_count = 0
correct_count = 0
error_count = 0  # Đếm các trường hợp không trích xuất được số

print(f">>> Bắt đầu phân tích file: {RESULTS_FILE}")


with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Calculating metrics"):
        # Đọc từng dòng JSON
        data = json.loads(line)
        total_count += 1

        # Lấy câu trả lời của model và đáp án đúng
        model_answer = data["model_answer"]
        ground_truth_answer = data["ground_truth_answer"]

        # Trích xuất số từ cả hai câu trả lời
        model_result = extract_last_number(model_answer)
        gold_result = extract_last_number(ground_truth_answer)

        if model_result is not None and gold_result is not None:
            # So sánh kết quả sau khi chuyển sang kiểu float
            try:
                if float(model_result) == float(gold_result):
                    correct_count += 1
            except ValueError:
                # Xử lý trường hợp chuỗi trích xuất không phải là số hợp lệ
                error_count += 1
        else:
            error_count += 1

if total_count > 0:
    accuracy = (correct_count / total_count) * 100
else:
    accuracy = 0

print("\n---KẾT QUẢ ĐÁNH GIÁ ---")
print(f"Tổng số mẫu đã xử lý: {total_count}")
print(f"Số câu trả lời đúng: {correct_count}")
print(f"Số câu không thể trích xuất số: {error_count}")
print(f"Độ chính xác (Accuracy): {accuracy:.2f}%")