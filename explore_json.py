# explore_json.py
import json
import os
from pprint import pprint

# 파일 경로
json_path = './data/commercial_law/commercial-law.json'

# 파일 존재 확인
if not os.path.exists(json_path):
    print(f"파일을 찾을 수 없습니다: {json_path}")
    exit(1)

print(f"파일 경로: {json_path}")

# 파일 로드
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print("JSON 파일 로드 성공")
except Exception as e:
    print(f"파일 로드 오류: {str(e)}")
    exit(1)

# 상위 레벨 구조 확인
print(f"\n데이터 타입: {type(data)}")
if isinstance(data, list):
    print(f"리스트 길이: {len(data)}")
    print("\n첫 번째 항목 키:")
    if len(data) > 0 and isinstance(data[0], dict):
        for key in data[0].keys():
            print(f"  - {key}")
        
        # 첫 번째 항목 샘플 데이터 출력
        print("\n첫 번째 항목 샘플:")
        pprint(data[0], depth=2, compact=True)
elif isinstance(data, dict):
    print("\n최상위 키:")
    for key in data.keys():
        print(f"  - {key}")
    
    # 중첩 구조 확인
    for key in data.keys():
        value = data[key]
        if isinstance(value, dict):
            print(f"\n'{key}' 하위 키:")
            for subkey in value.keys():
                print(f"  - {subkey}")
        elif isinstance(value, list):
            print(f"\n'{key}' 리스트 길이: {len(value)}")
            if len(value) > 0:
                if isinstance(value[0], dict):
                    print(f"  첫 항목 키: {list(value[0].keys())}")
                else:
                    print(f"  첫 항목 타입: {type(value[0])}")

# 텍스트 필드 찾기
def find_text_fields(obj, path=""):
    """객체 내 가능한 텍스트 필드를 재귀적으로 찾기"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            
            # 텍스트 같은 필드 출력
            if isinstance(value, str) and len(value) > 50:
                print(f"텍스트 필드 후보: {new_path} (첫 50자: {value[:50]}...)")
            
            find_text_fields(value, new_path)
    elif isinstance(obj, list) and len(obj) > 0:
        # 첫 항목만 확인
        find_text_fields(obj[0], f"{path}[0]")

print("\n\n텍스트 필드 후보 찾기:")
if isinstance(data, list) and len(data) > 0:
    find_text_fields(data[0])
else:
    find_text_fields(data)