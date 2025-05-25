import React, { useState } from 'react';
import './App.css'; // 기본 CSS 파일 (수정하지 않아도 됨)

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault(); // 폼 기본 제출 동작 방지
    if (!question.trim()) {
      setError('질문을 입력해주세요.');
      return;
    }

    setIsLoading(true);
    setAnswer('');
    setError('');

    try {
      // FastAPI 백엔드의 /search 엔드포인트로 요청
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: {
          // FastAPI에서 Body(..., media_type="text/plain")으로 설정했으므로
          // Content-Type을 text/plain으로 설정합니다.
          // 만약 FastAPI에서 JSON을 기대한다면 'application/json'으로 변경하고,
          // body도 JSON.stringify({ question: question }) 형태로 보내야 합니다.
          'Content-Type': 'text/plain',
        },
        body: question, // 질문 텍스트를 직접 body로 전송
      });

      if (!response.ok) {
        // 서버에서 에러 응답을 보낸 경우 (예: 4xx, 5xx 상태 코드)
        const errorData = await response.json(); // FastAPI가 에러를 JSON으로 반환한다고 가정
        throw new Error(errorData.detail || `서버 오류: ${response.status}`);
      }

      const data = await response.json(); // FastAPI가 응답을 JSON으로 반환한다고 가정
      setAnswer(data.answer);

    } catch (err) {
      console.error("API 요청 중 오류 발생:", err);
      setError(err.message || '답변을 가져오는 중 오류가 발생했습니다.');
      setAnswer(''); // 오류 발생 시 이전 답변 초기화
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>청년 정책 알리미</h1>
      </header>
      <main>
        <form onSubmit={handleSubmit}>
          <div>
            <label htmlFor="questionInput">질문 입력:</label>
            <textarea
              id="questionInput"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="청년 정책에 대해 질문해주세요..."
              rows="4"
              cols="50"
              disabled={isLoading}
            />
          </div>
          <button type="submit" disabled={isLoading}>
            {isLoading ? '질문 중...' : '질문하기'}
          </button>
        </form>

        {error && (
          <div style={{ color: 'red', marginTop: '20px' }}>
            <p>오류: {error}</p>
          </div>
        )}

        {answer && (
          <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '10px' }}>
            <h2>답변:</h2>
            <p style={{ whiteSpace: 'pre-wrap' }}>{answer}</p> {/* 줄바꿈 유지를 위해 pre-wrap 사용 */}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;