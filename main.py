import asyncio
import os
import queue
import threading
from typing import Optional

from dotenv import load_dotenv
from fastapi import Body, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from srt_processor import SrtProcessor
from sse_starlette.sse import EventSourceResponse

# --- 초기 설정 ---
load_dotenv()
API_KEY_EXISTS = bool(os.environ.get("GOOGLE_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(BASE_DIR, "srt_tool_workspace")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(os.path.join(WORKSPACE_DIR, "original_english_SRTs"), exist_ok=True)
os.makedirs(os.path.join(WORKSPACE_DIR, "videos"), exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATES_DIR)
log_queue = queue.Queue()
processor = SrtProcessor(workspace_dir=WORKSPACE_DIR, prompts_dir=PROMPTS_DIR)
processor.set_log_queue(log_queue)


def run_in_thread(target_func, *args, **kwargs):
    thread = threading.Thread(target=target_func, args=args, kwargs=kwargs, daemon=True)
    thread.start()


# --- API 엔드포인트 ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "api_key_exists": API_KEY_EXISTS,
        },
    )


async def log_generator():
    while True:
        try:
            level, message = log_queue.get_nowait()
            yield f"data: [{level}] {message}\n\n"
        except queue.Empty:
            await asyncio.sleep(0.1)


@app.get("/stream-logs")
async def stream_logs():
    return EventSourceResponse(log_generator())


# --- 작업 실행 API ---
@app.post("/start-labeling")
async def start_labeling_process(payload: dict = Body(...)):
    episodes_str = payload.get("episodes")
    skip_review = payload.get("skip_review", False)
    if not episodes_str:
        return JSONResponse(
            status_code=400, content={"message": "에피소드 범위를 입력하세요."}
        )

    # [핵심 수정] run_labeling_process -> run_full_process 호출로 변경
    run_in_thread(
        processor.run_full_process, episodes_str=episodes_str, skip_review=skip_review
    )
    return JSONResponse(content={"message": "1단계(라벨링) 작업을 시작합니다."})


@app.post("/start-translation")
async def start_translation_process(payload: dict = Body(...)):
    episodes_str = payload.get("episodes")
    if not episodes_str:
        return JSONResponse(
            status_code=400, content={"message": "에피소드 범위를 입력하세요."}
        )

    run_in_thread(
        processor.run_translation_process,
        episodes_str=episodes_str,
        batch_folder_name=None,
    )
    return JSONResponse(content={"message": "2단계(번역) 작업을 시작합니다."})


# --- 도구 API ---
@app.post("/run-time-shift")
async def run_time_shift(payload: dict = Body(...)):
    target_dir = payload.get("target_dir")
    start_num = payload.get("start_num")
    offset = payload.get("offset")
    if target_dir is None or start_num is None or offset is None:
        return JSONResponse(
            status_code=400, content={"message": "모든 필드를 입력하세요."}
        )
    run_in_thread(
        processor.run_time_shift,
        target_dir_str=target_dir,
        start_num=start_num,
        offset=offset,
    )
    return JSONResponse(content={"message": "시간 일괄 조절 작업을 시작합니다."})


@app.get("/get-videos")
async def get_videos():
    try:
        video_files = processor.get_video_files()
        return JSONResponse(content=video_files)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/run-vad")
async def run_vad(payload: dict = Body(...)):
    videos = payload.get("videos")
    if not videos:
        return JSONResponse(
            status_code=400, content={"message": "VAD를 실행할 비디오를 선택하세요."}
        )
    run_in_thread(processor.run_vad_for_videos, selected_videos=videos)
    return JSONResponse(content={"message": "VAD 자막 생성을 시작합니다."})


@app.get("/get-capcut-projects")
async def get_capcut_projects(path: Optional[str] = None):
    try:
        if not path:
            path = r"C:\Users\a7182\AppData\Local\CapCut\User Data\Projects\com.lveditor.draft"
        projects = processor.get_capcut_projects(base_path=path)
        return JSONResponse(content=projects)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/run-capcut-extraction")
async def run_capcut_extraction(payload: dict = Body(...)):
    projects = payload.get("projects")
    base_path = payload.get("base_path")
    if not projects or not base_path:
        return JSONResponse(
            status_code=400,
            content={"message": "추출할 CapCut 프로젝트와 경로를 확인하세요."},
        )
    run_in_thread(
        processor.run_capcut_extraction, projects_map=projects, base_path=base_path
    )
    return JSONResponse(content={"message": "CapCut 자막 추출을 시작합니다."})


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("🚀 SRT Tool 서버를 시작합니다. (v5.3)")
    print("🌐 웹 브라우저에서 http://127.0.0.1:8000 에 접속하세요.")
    print("=" * 50 + "\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
