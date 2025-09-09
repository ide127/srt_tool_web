import json
import logging
import os
import platform
import queue
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# --- CONFIGURATION (설정) ---
GEMINI_MODEL_NAME = "gemini-2.5-pro"
MAX_TRANSLATION_ATTEMPTS = 5
JUNK_MESSAGES_FROM_CLI = ["Loaded cached credentials.", "Data collection is disabled."]


class SrtProcessor:
    def __init__(self, workspace_dir: str, prompts_dir: str):
        self.workspace_dir = workspace_dir
        self.prompts_dir = prompts_dir
        self.log_queue = None

        # Logger 초기화 시 FileHandler는 추가하지 않음 (작업별로 동적 추가)
        self.logger = logging.getLogger("SrtProcessorLogger")
        self.logger.setLevel(logging.INFO)
        # 콘솔 출력을 위한 핸들러 (서버 터미널에서도 로그 확인 가능)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())

        self.current_job_file_handler = None
        self.debug_prompts_dir = None  # 작업별로 동적으로 설정

    def set_log_queue(self, log_queue: queue.Queue):
        self.log_queue = log_queue

    def _log(self, message: str, level: str = "INFO"):
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
        if self.log_queue:
            self.log_queue.put((level, message))

    # ====================================
    # 작업별 로깅 설정/해제 헬퍼
    # ====================================
    def _setup_job_logging(self, batch_folder_path: str):
        # 기존 핸들러가 있다면 제거
        self._remove_job_logging()

        log_folder_path = os.path.join(batch_folder_path, "logs")
        prompt_folder_path = os.path.join(batch_folder_path, "debug_prompts")
        os.makedirs(log_folder_path, exist_ok=True)
        os.makedirs(prompt_folder_path, exist_ok=True)
        self.debug_prompts_dir = prompt_folder_path

        log_filepath = os.path.join(
            log_folder_path,
            f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )

        # 새 파일 핸들러 생성 및 추가
        self.current_job_file_handler = logging.FileHandler(
            log_filepath,
            encoding="utf-8",
        )
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        self.current_job_file_handler.setFormatter(formatter)
        self.logger.addHandler(self.current_job_file_handler)
        self._log(f"작업 로그 파일이 다음 경로에 생성됩니다: {log_filepath}", "INFO")

    def _remove_job_logging(self):
        if self.current_job_file_handler:
            self.logger.removeHandler(self.current_job_file_handler)
            self.current_job_file_handler.close()
            self.current_job_file_handler = None

    # ====================================
    # SRT 파싱 및 병합 로직
    # ====================================
    @staticmethod
    def parse_srt_content(content: str) -> List[Dict[str, str]]:
        blocks = []
        time_pattern = re.compile(
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
        )

        content = content.replace("\r\n", "\n").replace("\r", "\n")
        content_chunks = content.strip().split("\n\n")

        block_counter = 1
        for chunk in content_chunks:
            lines = chunk.strip().split("\n")
            if not lines:
                continue

            time_line, time_line_index = None, -1
            for i, line in enumerate(lines):
                if "-->" in line and time_pattern.search(line):
                    time_line, time_line_index = line, i
                    break

            if time_line:
                number_part = lines[:time_line_index]
                text_part = lines[time_line_index + 1 :]

                number_str = "".join(number_part).strip()
                if not number_str.isdigit():
                    number_str = str(block_counter)

                blocks.append(
                    {
                        "number": number_str,
                        "time": time_line.strip().replace(".", ","),
                        "text": "\n".join(text_part).strip(),
                    },
                )
                block_counter += 1
        return blocks

    def _merge_to_srt(self, time_content: str, sentence_content: str) -> str:
        time_chunks = time_content.strip().split("\n\n")
        time_map = {}
        for t_chunk in time_chunks:
            t_lines = t_chunk.strip().split("\n")
            if len(t_lines) >= 2 and t_lines[0].isdigit():
                original_number, time_line = t_lines[0], t_lines[1]
                time_map[original_number] = time_line

        sentence_chunks = sentence_content.strip().split("\n\n")
        sentence_map = {}
        for s_chunk in sentence_chunks:
            s_lines = s_chunk.strip().split("\n")
            if s_lines and s_lines[0].isdigit():
                number = s_lines[0]
                text_lines = [
                    re.sub(r"^\s*[^:]+:\s*", "", line).strip() for line in s_lines[1:]
                ]
                valid_text_lines = [line for line in text_lines if line]
                if valid_text_lines:
                    sentence_map[number] = valid_text_lines

        srt_output = []
        new_block_counter = 1

        sorted_keys = sorted(time_map.keys(), key=int)

        for original_number in sorted_keys:
            if original_number in sentence_map:
                time_line = time_map[original_number]
                text_lines = sentence_map[original_number]

                if len(text_lines) == 1 or all(
                    line.strip().startswith("-") for line in text_lines
                ):
                    srt_output.append(
                        f"{new_block_counter}\n{time_line}\n"
                        + "\n".join(text_lines)
                        + "\n",
                    )
                    new_block_counter += 1
                else:
                    try:
                        start_time_str, end_time_str = time_line.split(" --> ")
                        start_td, end_td = (
                            self._srt_time_to_td(start_time_str),
                            self._srt_time_to_td(end_time_str),
                        )
                        total_duration_td = end_td - start_td
                        if total_duration_td.total_seconds() < 0:
                            total_duration_td = timedelta(seconds=0)

                        total_len = sum(len(line) for line in text_lines) or 1
                        current_start_td = start_td

                        for i, line in enumerate(text_lines):
                            proportion = len(line) / total_len
                            line_end_td = current_start_td + (
                                total_duration_td * proportion
                            )
                            if i == len(text_lines) - 1:
                                line_end_td = end_td

                            new_start_str, new_end_str = (
                                self._td_to_srt_time(current_start_td),
                                self._td_to_srt_time(line_end_td),
                            )

                            srt_output.append(
                                f"{new_block_counter}\n{new_start_str} --> {new_end_str}\n{line}\n",
                            )
                            new_block_counter += 1
                            current_start_td = line_end_td
                    except Exception as e:
                        self._log(
                            f"시간 분배 중 오류 (블록 #{original_number}): {e}",
                            "ERROR",
                        )
                        srt_output.append(
                            f"{new_block_counter}\n{time_line}\n"
                            + "\n".join(text_lines)
                            + "\n",
                        )
                        new_block_counter += 1
        return "\n".join(srt_output)

    # ====================================
    # Gemini CLI 호출 및 워크플로우
    # ====================================
    def _call_gemini_cli(self, prompt: str) -> Tuple[bool, str, str, int]:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+",
                delete=False,
                encoding="utf-8",
                suffix=".txt",
            ) as temp_prompt_file:
                temp_prompt_file.write(prompt)
                temp_filepath = temp_prompt_file.name

            command = ["gemini", "--debug", "-m", GEMINI_MODEL_NAME]
            self._log(f"실행할 명령어: {' '.join(command)} < {temp_filepath}", "INFO")

            with open(temp_filepath, encoding="utf-8") as infile:
                process = subprocess.Popen(
                    command,
                    stdin=infile,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                )
                stdout, stderr = process.communicate()

            os.remove(temp_filepath)

            if process.returncode != 0:
                return False, stdout, stderr, process.returncode

            clean_stdout = "\n".join(
                [
                    line
                    for line in stdout.split("\n")
                    if not any(junk in line for junk in JUNK_MESSAGES_FROM_CLI)
                ],
            ).strip()
            return True, clean_stdout, "", process.returncode
        except FileNotFoundError:
            return False, "", "'gemini' 명령을 찾을 수 없습니다.", -1
        except Exception as e:
            if "temp_filepath" in locals() and os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return False, "", str(e), -1

    def _cross_checked_gemini_call(
        self,
        ep_num: int,
        content_to_process: str,
        prompt_template: str,
        context: str,
    ) -> Optional[str]:
        full_prompt = f"{prompt_template}\n\n[문맥 정보]\n{context if context else '없음'}\n\n[처리해야 할 내용]\n{content_to_process}"
        results_cache = []

        for attempt in range(MAX_TRANSLATION_ATTEMPTS):
            if attempt == 0 and self.debug_prompts_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prompt_filename = f"prompt_{timestamp}_ep{ep_num}.txt"
                prompt_filepath = os.path.join(self.debug_prompts_dir, prompt_filename)
                try:
                    with open(prompt_filepath, "w", encoding="utf-8") as f:
                        f.write(full_prompt)
                    self._log(
                        f"디버그용 프롬프트를 다음 파일에 저장했습니다: {prompt_filepath}",
                        "INFO",
                    )
                except Exception as e:
                    self._log(f"프롬프트 파일 저장 실패: {e}", "ERROR")

            self._log(
                f"Gemini API 호출 시도 #{attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}...",
                "INFO",
            )
            success, result_text, error_msg, exit_code = self._call_gemini_cli(
                full_prompt,
            )

            if not success:
                error_details = f"종료 코드: {exit_code}\n[Stderr]:\n{error_msg.strip()}\n[Stdout]:\n{result_text.strip()}"
                self._log(f"API 호출 실패:\n{error_details}", "ERROR")
                self._log(
                    "자세한 내용은 작업 폴더 내의 로그 파일을 확인하세요.",
                    "ERROR",
                )
                continue

            if success and not result_text:
                self._log(
                    "API 호출은 성공(종료 코드 0)했으나 출력이 없습니다. CLI 환경이나 인증을 확인하세요.",
                    "WARNING",
                )
                continue

            line_map = self.create_line_count_map(result_text)
            results_cache.append({"text": result_text, "map": line_map})

            if len(results_cache) >= 2:
                latest, prev = results_cache[-1], results_cache[-2]
                if latest["map"] == prev["map"]:
                    self._log(
                        f"✅ 구조 일치 발견! (시도 #{len(results_cache) - 1}과 #{len(results_cache)})",
                        "INFO",
                    )
                    return latest["text"]
                self._log(" L 구조 불일치 발견. 상세 비교:", "WARNING")
                diff_keys = set(prev["map"].keys()) ^ set(latest["map"].keys()) | {
                    k
                    for k, v in prev["map"].items()
                    if k in latest["map"] and v != latest["map"][k]
                }
                log_detail = []
                for key in sorted(diff_keys, key=int):
                    prev_text = self.get_text_block_for_number(
                        prev["text"],
                        key,
                    ).replace("\n", " ")
                    latest_text = self.get_text_block_for_number(
                        latest["text"],
                        key,
                    ).replace("\n", " ")
                    log_detail.append(
                        f"  - #{key}: [이전] L:{prev['map'].get(key, 'N')}, C: {prev_text} / [최신] L:{latest['map'].get(key, 'N')}, C: {latest_text}",
                    )
                if log_detail:
                    self._log("\n".join(log_detail), "CONTEXT")
            else:
                self._log(
                    f"시도 #{attempt + 1} 완료. 구조 일치를 위해 추가 시도합니다.",
                    "WARNING",
                )

        self._log(
            f"최대 시도({MAX_TRANSLATION_ATTEMPTS}회) 이후에도 안정적인 구조를 얻지 못했습니다.",
            "ERROR",
        )
        return None

    def run_full_process(self, episodes_str: str, skip_review: bool):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name_base = episodes_str.replace(",", "_").replace("-", "to")
        output_folder_name = f"{folder_name_base}_{timestamp}"
        batch_dir = os.path.join(self.workspace_dir, output_folder_name)

        try:
            self._setup_job_logging(batch_dir)

            # --- 1단계 ---
            self._log(
                f"===== 1단계: 분리 및 라벨링 시작 (결과 폴더: {output_folder_name}) =====",
                "INFO",
            )
            labeling_success, total_count, success_count = self._run_labeling_stage(
                episodes_str,
                output_folder_name,
            )

            # --- 1단계 요약 ---
            summary_step1 = [
                "--- 1단계 작업 요약 ---",
                f"요청 에피소드: {total_count}개",
                f"성공: {success_count}개",
                f"실패: {total_count - success_count}개",
                f"결과 폴더: {batch_dir}",
                "-----------------------",
            ]
            self._log("\n".join(summary_step1), "INFO")

            # --- 2단계 (조건부 실행) ---
            if labeling_success and skip_review:
                self._log(
                    "'즉시 번역' 옵션이 활성화되어 자동으로 2단계를 시작합니다.",
                    "INFO",
                )
                self.run_translation_process(episodes_str, output_folder_name)
            elif not skip_review and labeling_success:
                self._log(
                    "라벨링 파일을 검토한 후 [2단계: 번역 및 병합] 버튼을 누르세요.",
                    "INFO",
                )
            elif not labeling_success:
                self._log("1단계 작업 중 오류가 발생하여 중단합니다.", "ERROR")

        finally:
            self._remove_job_logging()

    def _run_labeling_stage(
        self,
        episodes_str: str,
        output_folder_name: str,
    ) -> Tuple[bool, int, int]:
        batch_dir = os.path.join(self.workspace_dir, output_folder_name)
        time_dir = os.path.join(batch_dir, "txtWithTime")
        labeling_dir = os.path.join(batch_dir, "txtWithLabeling")
        os.makedirs(time_dir, exist_ok=True)
        os.makedirs(labeling_dir, exist_ok=True)

        srt_map = self._get_episode_file_map("original_english_SRTs")
        episode_nums = self._parse_episode_range(episodes_str)

        success_count = 0
        for ep_num in episode_nums:
            if self._split_and_label_episode(ep_num, srt_map, time_dir, labeling_dir):
                success_count += 1

        return success_count == len(episode_nums), len(episode_nums), success_count

    def _split_and_label_episode(
        self,
        ep_num: int,
        srt_map: Dict[int, str],
        time_dir: str,
        labeling_dir: str,
    ) -> bool:
        base_name = os.path.splitext(
            os.path.basename(srt_map.get(ep_num, f"ep{ep_num}.srt")),
        )[0]
        self._log(f"--- 에피소드 {ep_num} ({base_name}) 처리 시작 ---", "INFO")
        srt_path = srt_map.get(ep_num)
        if not srt_path:
            self._log(
                f"에피소드 {ep_num}의 원본 SRT 파일을 찾을 수 없습니다.",
                "WARNING",
            )
            return False

        try:
            with open(srt_path, encoding="utf-8-sig") as f:
                srt_content = f.read()
            blocks = self.parse_srt_content(srt_content)

            time_content = "\n\n".join(f"{b['number']}\n{b['time']}" for b in blocks)
            sentence_content = "\n\n".join(
                f"{b['number']}\n{b['text']}" for b in blocks
            )

            with open(
                os.path.join(time_dir, f"{base_name}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(time_content)
        except Exception as e:
            self._log(f"파일 분리 실패 ({base_name}): {e}", "ERROR")
            return False

        context = self._get_context_dialogue(ep_num, srt_map)
        prompt_template = open(
            os.path.join(self.prompts_dir, "labeling_prompt.txt"),
            encoding="utf-8",
        ).read()

        labeled_text = self._cross_checked_gemini_call(
            ep_num,
            sentence_content,
            prompt_template,
            context,
        )

        if (
            labeled_text
            and self._validate_format(labeled_text)
            and self._validate_sequential_numbering(labeled_text)
        ):
            with open(
                os.path.join(labeling_dir, f"{base_name}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(labeled_text)
            self._log(f"라벨링 완료: {base_name}.txt", "INFO")
            return True
        self._log(f"라벨링 실패 또는 결과 검증 실패: {base_name}.txt", "ERROR")
        return False

    def run_translation_process(
        self,
        episodes_str: str,
        batch_folder_name: Optional[str],
    ):
        if not batch_folder_name:
            folder_name_base = episodes_str.replace(",", "_").replace("-", "to")
            try:
                candidate_dirs = [
                    d
                    for d in os.listdir(self.workspace_dir)
                    if d.startswith(folder_name_base)
                ]
                if not candidate_dirs:
                    self._log(
                        f"'{folder_name_base}'로 시작하는 작업 폴더를 찾을 수 없습니다. 1단계를 먼저 실행하세요.",
                        "ERROR",
                    )
                    return
                batch_folder_name = sorted(candidate_dirs)[-1]
                self._log(
                    f"가장 최근 작업 폴더를 대상으로 번역을 시작합니다: {batch_folder_name}",
                    "INFO",
                )
            except Exception as e:
                self._log(f"작업 폴더 검색 중 오류: {e}", "ERROR")
                return

        batch_dir = os.path.join(self.workspace_dir, batch_folder_name)
        try:
            self._setup_job_logging(batch_dir)
            self._log(
                f"===== 2단계: 번역 및 병합 시작 (대상 폴더: {batch_folder_name}) =====",
                "INFO",
            )

            os.makedirs(os.path.join(batch_dir, "updatedSrt"), exist_ok=True)
            os.makedirs(os.path.join(batch_dir, "txtWithSentence"), exist_ok=True)

            srt_map = self._get_episode_file_map("original_english_SRTs")
            episode_nums = self._parse_episode_range(episodes_str)

            success_count = 0
            for ep_num in episode_nums:
                if self._translate_and_merge_episode(ep_num, srt_map, batch_dir):
                    success_count += 1

            summary_step2 = [
                "--- 2단계 작업 요약 ---",
                f"요청 에피소드: {len(episode_nums)}개",
                f"성공: {success_count}개",
                f"실패: {len(episode_nums) - success_count}개",
                f"최종 SRT 저장 폴더: {os.path.join(batch_dir, 'updatedSrt')}",
                "-----------------------",
            ]
            self._log("\n".join(summary_step2), "INFO")
        finally:
            self._remove_job_logging()

    def _translate_and_merge_episode(
        self,
        ep_num: int,
        srt_map: Dict[int, str],
        batch_dir: str,
    ) -> bool:
        base_name = os.path.splitext(
            os.path.basename(srt_map.get(ep_num, f"ep{ep_num}.srt")),
        )[0]
        self._log(f"--- 에피소드 {ep_num} ({base_name}) 번역 시작 ---", "INFO")

        labeled_txt_path = os.path.join(
            batch_dir,
            "txtWithLabeling",
            f"{base_name}.txt",
        )
        time_txt_path = os.path.join(batch_dir, "txtWithTime", f"{base_name}.txt")
        if not (os.path.exists(labeled_txt_path) and os.path.exists(time_txt_path)):
            self._log(
                f"필수 파일(라벨링, 시간)이 없어 건너뜁니다: {base_name}",
                "ERROR",
            )
            return False

        with open(labeled_txt_path, encoding="utf-8") as f:
            labeled_content = f.read()

        context = self._get_context_dialogue(ep_num, srt_map)
        prompt_template = open(
            os.path.join(self.prompts_dir, "translation_prompt.txt"),
            encoding="utf-8",
        ).read()
        translated_text = self._cross_checked_gemini_call(
            ep_num,
            labeled_content,
            prompt_template,
            context,
        )

        if (
            translated_text
            and self._validate_format(translated_text)
            and self._validate_sequential_numbering(translated_text)
        ):
            translated_path = os.path.join(
                batch_dir,
                "txtWithSentence",
                f"{base_name}.txt",
            )
            with open(translated_path, "w", encoding="utf-8") as f:
                f.write(translated_text)
            self._log(
                f"번역 결과 저장 완료: {os.path.basename(translated_path)}",
                "INFO",
            )

            with open(time_txt_path, encoding="utf-8") as f:
                time_content = f.read()
            merged_srt = self._merge_to_srt(time_content, translated_text)

            output_srt_path = os.path.join(
                batch_dir,
                "updatedSrt",
                f"{base_name}_updated.srt",
            )
            with open(output_srt_path, "w", encoding="utf-8") as f:
                f.write(merged_srt)
            self._log(f"SRT 병합 완료: {os.path.basename(output_srt_path)}", "INFO")
            return True
        self._log(f"번역 실패 또는 결과 검증 실패: {base_name}", "ERROR")
        return False

    # ====================================
    # 도구 모음 (시간조절, VAD, CapCut)
    # ====================================
    def run_time_shift(self, target_dir_str: str, start_num: int, offset: float):
        self._log(f"===== 시간 일괄 조절 시작 ({target_dir_str}) =====", "INFO")
        target_dir = os.path.join(self.workspace_dir, target_dir_str)
        if not os.path.isdir(target_dir):
            self._log(f"대상 폴더를 찾을 수 없습니다: {target_dir}", "ERROR")
            return

        output_dir = os.path.join(target_dir, "timeShiftedSrt")
        os.makedirs(output_dir, exist_ok=True)

        srt_files = [f for f in os.listdir(target_dir) if f.lower().endswith(".srt")]
        for filename in srt_files:
            try:
                with open(
                    os.path.join(target_dir, filename),
                    encoding="utf-8-sig",
                ) as f:
                    content = f.read()
                blocks = self.parse_srt_content(content)
                new_blocks = []
                for block in blocks:
                    if int(block["number"]) >= start_num:
                        start_str, end_str = block["time"].split(" --> ")
                        new_start = self._shift_time_str(start_str, offset)
                        new_end = self._shift_time_str(end_str, offset)
                        block["time"] = f"{new_start} --> {new_end}"
                    new_blocks.append(
                        f"{block['number']}\n{block['time']}\n{block['text']}\n",
                    )

                with open(
                    os.path.join(output_dir, filename),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("\n".join(new_blocks))
                self._log(f"시간 조절 완료: {filename}", "INFO")
            except Exception as e:
                self._log(f"시간 조절 실패 ({filename}): {e}", "ERROR")
        self._log(f"✅ 시간 조절 완료. 결과 폴더: {output_dir}", "INFO")

    def get_video_files(self) -> List[str]:
        video_dir = os.path.join(self.workspace_dir, "videos")
        if not os.path.isdir(video_dir):
            return []
        ext = (".mp4", ".mkv", ".mov", ".avi", ".flv")
        return [f for f in os.listdir(video_dir) if f.lower().endswith(ext)]

    def run_vad_for_videos(self, selected_videos: List[str]):
        self._log("===== VAD 자막 생성 시작 =====", "INFO")
        video_dir = os.path.join(self.workspace_dir, "videos")
        output_dir = os.path.join(self.workspace_dir, "vad_generated_srts")
        os.makedirs(output_dir, exist_ok=True)

        if not shutil.which("ffmpeg"):
            self._log(
                "FFmpeg가 설치되어 있지 않거나 시스템 경로에 없습니다. FFmpeg를 설치해주세요.",
                "ERROR",
            )
            return

        try:
            from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

            model = load_silero_vad()
        except Exception as e:
            self._log(
                f"Silero VAD 모델 로딩 실패: {e}. 'pip install torch torchaudio soundfile silero-vad'를 실행하세요.",
                "ERROR",
            )
            return

        for video_name in selected_videos:
            input_file = os.path.join(video_dir, video_name)
            output_srt = os.path.join(
                output_dir,
                f"{os.path.splitext(video_name)[0]}_vad.srt",
            )
            self._log(f"처리 중: {video_name}", "INFO")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name

            try:
                cmd = [
                    "ffmpeg",
                    "-i",
                    input_file,
                    "-vn",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    "-y",
                    temp_wav_path,
                ]
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                wav = read_audio(temp_wav_path, sampling_rate=16000)
                segments = get_speech_timestamps(
                    wav,
                    model,
                    sampling_rate=16000,
                    return_seconds=True,
                )

                with open(output_srt, "w", encoding="utf-8") as fh:
                    for i, seg in enumerate(segments, start=1):
                        start = timedelta(seconds=seg["start"])
                        end = timedelta(seconds=seg["end"])
                        fh.write(
                            f"{i}\n{self._td_to_srt_time(start)} --> {self._td_to_srt_time(end)}\n[speech]\n\n",
                        )
                self._log(f"성공: {os.path.basename(output_srt)} 생성 완료", "INFO")
            except Exception as e:
                self._log(f"VAD 처리 오류 ({video_name}): {e}", "ERROR")
            finally:
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
        self._log(f"✅ VAD 작업 완료. 결과 폴더: {output_dir}", "INFO")

    def get_capcut_projects(self, base_path: str) -> Dict[str, str]:
        wsl_safe_path = self._convert_windows_path_to_wsl(base_path)
        projects = {}
        if not os.path.isdir(wsl_safe_path):
            self._log(
                f"CapCut 프로젝트 폴더를 찾을 수 없습니다: {wsl_safe_path}",
                "WARNING",
            )
            return {}
        for item in os.listdir(wsl_safe_path):
            proj_dir = os.path.join(wsl_safe_path, item)
            draft_path = os.path.join(proj_dir, "draft_info.json")
            if os.path.isdir(proj_dir) and os.path.exists(draft_path):
                name = f"Unnamed ({item})"
                try:
                    with open(draft_path, encoding="utf-8") as f:
                        info = json.load(f)
                    if info.get("draft_name"):
                        name = f"{info['draft_name']} ({item})"
                except:
                    pass
                projects[name] = proj_dir
        return projects

    def run_capcut_extraction(self, projects_map: Dict[str, str], base_path: str):
        self._log("===== CapCut 자막 추출 시작 =====", "INFO")
        output_dir = os.path.join(self.workspace_dir, "capcut_extracted_srts")
        os.makedirs(output_dir, exist_ok=True)

        for name, path in projects_map.items():
            self._log(f"추출 중: {name}", "INFO")
            try:
                with open(os.path.join(path, "draft_info.json"), encoding="utf-8") as f:
                    data = json.load(f)
                subs = []
                text_map = {
                    t["id"]: t for t in data.get("materials", {}).get("texts", [])
                }
                for track in data.get("tracks", []):
                    if track.get("type") == "text":
                        for seg in track.get("segments", []):
                            mat_id = seg.get("material_id")
                            if mat_id in text_map:
                                text_info = text_map[mat_id]
                                timerange = seg.get("target_timerange", {})
                                start_us, dur_us = (
                                    timerange.get("start", 0),
                                    timerange.get("duration", 0),
                                )
                                content = (
                                    json.loads(text_info.get("content", "{}"))
                                    .get("text", "")
                                    .strip()
                                )
                                if content:
                                    subs.append(
                                        {
                                            "start": start_us,
                                            "end": start_us + dur_us,
                                            "text": content,
                                        },
                                    )
                if not subs:
                    self._log(f"정보: '{name}' 프로젝트에 자막 없음", "WARNING")
                    continue
                subs.sort(key=lambda s: s["start"])
                safe_name = "".join(
                    c
                    for c in data.get("draft_name", os.path.basename(path))
                    if c.isalnum() or c in " _-"
                )
                output_path = os.path.join(output_dir, f"{safe_name}.srt")
                with open(output_path, "w", encoding="utf-8-sig") as f:
                    for i, s in enumerate(subs, 1):
                        start = self._td_to_srt_time(timedelta(microseconds=s["start"]))
                        end = self._td_to_srt_time(timedelta(microseconds=s["end"]))
                        f.write(f"{i}\n{start} --> {end}\n{s['text']}\n\n")
                self._log(f"성공: '{os.path.basename(output_path)}' 저장 완료", "INFO")
            except Exception as e:
                self._log(f"오류 ({name}): {e}", "ERROR")
        self._log(f"✅ CapCut 추출 완료. 결과 폴더: {output_dir}", "INFO")

    # ====================================
    # 헬퍼(Helper) 메서드들
    # ====================================
    def _convert_windows_path_to_wsl(self, path: str) -> str:
        if platform.system() == "Linux" and (
            "microsoft" in platform.release().lower()
            or "wsl" in platform.release().lower()
        ):
            self._log("WSL 환경 감지됨. Windows 경로 자동 변환 시도...", "INFO")
            match = re.match(r"([A-Za-z]):[\\/]", path)
            if match:
                drive_letter = match.group(1).lower()
                new_path = f"/mnt/{drive_letter}/{path[3:].replace('\\', '/')}"
                self._log(f"경로 변환: {path} -> {new_path}", "INFO")
                return new_path
        return path

    @staticmethod
    def get_text_block_for_number(content: str, number: str) -> str:
        chunks = content.strip().split("\n\n")
        for chunk in chunks:
            lines = chunk.strip().split("\n")
            if lines and lines[0].strip() == number:
                return "\n".join(lines[1:]).strip()
        return "[내용 없음]"

    @staticmethod
    def _validate_format(content: str) -> bool:
        lines = content.strip().split("\n")
        if len(lines) <= 1:
            return True
        for i in range(1, len(lines)):
            if lines[i].strip().isdigit() and lines[i - 1].strip() != "":
                return False
        return True

    @staticmethod
    def _validate_sequential_numbering(content: str) -> bool:
        numbers = [int(n) for n in re.findall(r"^(\d+)\s*$", content, re.MULTILINE)]
        if not numbers:
            return True
        is_sequential = all(
            numbers[i] == numbers[i - 1] + 1 for i in range(1, len(numbers))
        )
        starts_at_one = numbers[0] == 1 if numbers else True
        return is_sequential and starts_at_one

    @staticmethod
    def srt_to_number_plus_text(srt_text: str) -> str:
        blocks = SrtProcessor.parse_srt_content(srt_text)
        return "\n\n".join(f"{b['number']}\n{b['text']}" for b in blocks)

    @staticmethod
    def create_line_count_map(content: str) -> Dict[str, int]:
        line_map = {}
        chunks = content.strip().split("\n\n")
        for chunk in chunks:
            lines = chunk.strip().split("\n")
            if lines and lines[0].strip().isdigit():
                number = lines[0].strip()
                line_map[number] = len([line for line in lines[1:] if line.strip()])
        return line_map

    def _parse_episode_range(self, episodes_str: str) -> List[int]:
        episodes = set()
        for part in episodes_str.replace(" ", "").split(","):
            if "-" in part:
                try:
                    start, end = map(int, part.split("-"))
                    episodes.update(range(start, end + 1))
                except ValueError:
                    continue
            elif part.isdigit():
                episodes.add(int(part))
        return sorted(list(episodes))

    def _get_episode_file_map(self, folder: str) -> Dict[int, str]:
        file_map = {}
        source_dir = os.path.join(self.workspace_dir, folder)
        if not os.path.isdir(source_dir):
            return {}

        for filename in os.listdir(source_dir):
            match = re.search(r"(\d{4})", filename)
            if match:
                episode_num = int(match.group(1))
                file_map[episode_num] = os.path.join(source_dir, filename)
        return file_map

    def _get_context_dialogue(
        self,
        current_episode: int,
        episode_map: Dict[int, str],
    ) -> str:
        context_episodes = [
            current_episode - 2,
            current_episode - 1,
            current_episode + 1,
            current_episode + 2,
        ]
        context_str = []
        for ep_num in context_episodes:
            srt_path = episode_map.get(ep_num)
            if srt_path:
                try:
                    with open(srt_path, encoding="utf-8-sig") as f:
                        content = f.read()
                    dialogue = self.srt_to_number_plus_text(content)
                    context_str.append(
                        f"--- 에피소드 {ep_num} 문맥 시작 ---\n{dialogue}\n--- 에피소드 {ep_num} 문맥 끝 ---",
                    )
                except Exception as e:
                    self._log(f"문맥 파일({srt_path}) 읽기 오류: {e}", "WARNING")
        return "\n\n".join(context_str)

    @staticmethod
    def _shift_time_str(time_str, offset_seconds):
        is_comma = "," in time_str
        dt_obj = datetime.strptime(time_str.strip().replace(",", "."), "%H:%M:%S.%f")
        new_dt = dt_obj + timedelta(seconds=offset_seconds)
        if new_dt.year < 1900:
            new_dt = datetime.strptime("00:00:00.000", "%H:%M:%S.%f")
        new_time_str = new_dt.strftime("%H:%M:%S.%f")[:-3]
        return new_time_str.replace(".", ",") if is_comma else new_time_str

    @staticmethod
    def _td_to_srt_time(td: timedelta) -> str:
        total_seconds = td.total_seconds()
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{td.microseconds // 1000:03d}"

    @staticmethod
    def _srt_time_to_td(time_str: str) -> timedelta:
        dt = datetime.strptime(time_str.strip().replace(",", "."), "%H:%M:%S.%f")
        return timedelta(
            hours=dt.hour,
            minutes=dt.minute,
            seconds=dt.second,
            microseconds=dt.microsecond,
        )
