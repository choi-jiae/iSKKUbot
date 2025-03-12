import http.client
import json

def evaluate_answer_generator(query):
    conn = http.client.HTTPSConnection("ngrok-api-address/chat")

    # 요청 본문 구성
    payload = json.dumps({"chat": query})
    
    # 요청 헤더 구성
    headers = {
        "Content-Type": "application/json"
    }
    
    # POST 요청 보내기
    conn.request("POST", "/chat", body=payload, headers=headers)
    
    # 응답 받기
    response = conn.getresponse()
    
    # 스트리밍 데이터를 읽고 결합
    if response.status == 200:  # HTTP 상태 코드 확인
        chunks = []
        while True:
            chunk = response.read(1024)  # 1024바이트씩 읽기
            if not chunk:
                break
            chunks.append(chunk)  # 바이트 그대로 저장
        conn.close()
        
        # 모든 바이트 데이터를 병합한 후 디코딩
        try:
            return b''.join(chunks).decode("utf-8")  # UTF-8로 한 번에 디코딩
        except UnicodeDecodeError:
            return b''.join(chunks).decode("iso-8859-1")  # 다른 인코딩 시도
    else:
        conn.close()
        return f"Error: {response.status} {response.reason}"
