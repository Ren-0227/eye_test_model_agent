"""
Orchestrator: glue code to connect text, image, vision test, memory, and LLM.
Returns structured responses with optional actions.
"""
import os
import uuid
from typing import Any, Dict, Optional

from backend.tools.local_qwen_api import LocalQwenAPI
from backend.tools.image_processing import analyze_image, load_error as img_load_error, reload_model
from backend.tools.memory_manager import get_user_memory, update_user_memory
from backend.tools.reporting import generate_report
from backend.tools.symptom_extractor import extract_structured
from backend.tools.followup_questioner import generate_followups
from backend.tools.risk_assessor import assess
from backend.tools.image_quality import evaluate_image_quality


class Orchestrator:
    def __init__(self, uploads_dir: str = "uploads"):
        self.llm = LocalQwenAPI()
        self.uploads_dir = uploads_dir
        os.makedirs(self.uploads_dir, exist_ok=True)

    def _needs_vision_test(self, text: Optional[str]) -> bool:
        if not text:
            return False
        keywords = ['模糊', '近视', '远视', '看不清', '视力下降', '眯眼']
        return any(k in text for k in keywords)

    def _should_offer_vision_test(self, text: Optional[str], struct: Dict[str, Any], risk_level: str, risk_reason: str, user_mem: Dict[str, Any]) -> bool:
        """
        综合判断是否需要触发视力检测：
        - 触发关键词（模糊/看不清等）
        - 风险为中/高时尚未有视力结果
        - 症状过短或提到视力但未提供结果
        """
        if user_mem.get("vision_test"):
            return False
        if self._needs_vision_test(text):
            return True
        if risk_level in ("medium", "high"):
            return True
        if text and len(text.strip()) < 10:
            return True
        if "视力" in (risk_reason or ""):
            return True
        return False

    def _risk_level(self, text: Optional[str]) -> str:
        if not text:
            return "low"
        high_kw = ['剧烈疼痛', '突然失明', '视网膜脱离', '大量飞蚊', '闪光感', '视野缺损']
        medium_kw = ['视力下降', '持续模糊', '复视', '眼压高', '黄斑病变']
        if any(k in text for k in high_kw):
            return "high"
        if any(k in text for k in medium_kw):
            return "medium"
        return "low"
    
    def _generate_fallback_response(self, text: Optional[str], struct: Dict[str, Any], risk: str, risk_reason: str, user_mem: Dict[str, Any]) -> str:
        """当LLM不可用时，生成基础回复"""
        if not text:
            return "请描述您的眼部症状，我会尽力为您提供建议。"
        
        # 根据风险级别和症状生成基础回复
        response_parts = []
        
        # 风险提示
        if risk == "high":
            response_parts.append("⚠️ 根据您的描述，症状较为严重，建议尽快就医。")
        elif risk == "medium":
            response_parts.append("⚠️ 建议您尽快咨询专业眼科医生。")
        
        # 基础建议
        if "模糊" in text or "看不清" in text or "视力" in text:
            response_parts.append("关于视力问题，建议：\n1. 避免长时间用眼\n2. 保持适当距离看屏幕\n3. 定期进行视力检查")
            if not user_mem.get("vision_test"):
                response_parts.append("建议进行视力检测以获取准确数据。")
        
        if "干涩" in text or "疲劳" in text:
            response_parts.append("关于眼部干涩/疲劳，建议：\n1. 多眨眼，保持眼部湿润\n2. 使用人工泪液\n3. 每20分钟看远处20秒（20-20-20法则）")
        
        if "疼痛" in text or "痛" in text:
            response_parts.append("眼部疼痛需要重视，建议：\n1. 立即停止用眼\n2. 如疼痛持续或加剧，请尽快就医\n3. 避免揉眼睛")
        
        if "飞蚊" in text or "黑影" in text:
            response_parts.append("关于飞蚊症，建议：\n1. 如突然出现大量飞蚊或伴随闪光，需立即就医\n2. 定期检查眼底\n3. 避免剧烈运动")
        
        # 通用建议
        if not response_parts:
            response_parts.append("根据您的描述，建议：\n1. 注意休息，避免过度用眼\n2. 保持良好用眼习惯\n3. 如症状持续或加重，请咨询专业医生")
        
        # 添加模型状态提示
        response_parts.append("\n\n[提示] AI模型暂时不可用，以上为基础建议。如需更详细的诊断，请修复模型加载问题。")
        
        return "\n".join(response_parts)

    def _tool_status(self) -> Dict[str, Any]:
        """Returns current availability of core tools."""
        return {
            "llm_ready": self.llm.load_error is None,
            "llm_error": self.llm.load_error,
            "oct_ready": img_load_error is None,
            "oct_error": img_load_error,
        }

    def tool_status(self) -> Dict[str, Any]:
        """Public endpoint to check tool status."""
        return self._tool_status()

    def chat(self, text: str, user_id: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        # Check tools are ready
        status = self._tool_status()
        if not status["llm_ready"]:
            return {
                "error": f"LLM not ready: {status['llm_error']}",
                "need_train_model": True,
                "actions": [],
            }
        if image_path and not status["oct_ready"]:
            return {
                "error": f"OCT model not ready: {status['oct_error']}",
                "need_train_model": True,
                "actions": [],
            }

        # Load memory
        memory = get_user_memory(user_id)
        history = memory.get("chat_history", [])

        # Extract structured symptom info
        symptoms = extract_structured(text)

        # Run OCT analysis if image is provided
        oct_result = None
        if image_path:
            quality = evaluate_image_quality(image_path)
            if quality["is_good_enough"]:
                try:
                    oct_result = analyze_image(image_path)
                except Exception as e:
                    return {
                        "error": f"OCT图像分析失败: {str(e)}",
                        "actions": [],
                    }
            else:
                return {
                    "answer": "您上传的眼部图像质量不佳，请在光线充足的情况下重新拍摄眼底照片。",
                    "actions": [{"type": "upload_image"}],
                }

        # Run vision test if needed
        vision_result = None
        if self._needs_vision_test(text):
            # Note: actual vision test would be triggered separately via /vision-test/start
            vision_result = "尚未进行视力检测，请点击视力检测按钮开始测试。"

        # Query LLM
        try:
            llm_resp = self.llm.get_health_advice(
                symptoms=symptoms["symptom_text"],
                vision_result=vision_result,
                oct_result=oct_result,
            )
            
            if llm_resp["status"] != "ok":
                return {
                    "error": f"LLM query failed: {llm_resp.get('message', 'Unknown error')}",
                    "actions": [],
                }
                
            answer = llm_resp["answer"]
        except Exception as e:
            return {
                "error": f"Failed to query LLM: {str(e)}",
                "actions": [],
            }

        # Risk assessment
        risk = self._risk_level(text)
        followups = generate_followups(symptoms)

        # Save to memory
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": answer})
        update_user_memory(user_id, {"chat_history": history})

        # Generate report if needed
        report_actions = []
        if risk in ("medium", "high"):
            report_path = os.path.join(self.uploads_dir, f"{user_id}_report.pdf")
            generate_report(report_path, {
                "symptoms": symptoms,
                "vision_result": vision_result,
                "oct_result": oct_result,
                "llm_response": answer,
                "risk_level": risk,
            })
            report_actions = [{
                "type": "download",
                "text": "下载诊断报告",
                "url": f"/download/report/{user_id}",
            }]

        return {
            "answer": answer,
            "risk": risk,
            "followups": followups,
            "actions": [
                {"type": "upload_image"},
                {"type": "vision_test"},
                *report_actions,
            ],
        }

    def process(self, user_id: str, text: Optional[str] = None, image_file=None, vision_tester=None) -> Dict[str, Any]:
        """
        Main entry for a request. Returns structured response:
        {
            status: "ok" | "error",
            answer: str (optional),
            action: None | "vision_test",
            data: { ... }
        }
        """
        user_mem = get_user_memory(user_id)
        tool_status = self._tool_status()
        context = {
            "current_symptoms": text,
            "medical_history": user_mem.get("history", []),
            "oct_result": user_mem.get("oct_result"),
            "vision_test": user_mem.get("vision_test"),
        }

        # No input guard
        if not text and image_file is None:
            return {
                "status": "ok",
                "answer": "请描述您的眼部症状，或上传眼部图片进行分析。",
                "action": None,
                "data": {"tool_status": tool_status}
            }

        # Handle image
        if image_file is not None:
            if tool_status["oct_error"]:
                return {
                    "status": "error",
                    "message": tool_status["oct_error"],
                    "data": {"tool_status": tool_status}
                }
            image_path = self._save_uploaded_file(image_file)
            try:
                quality = evaluate_image_quality(image_path)
                if not quality.get("ok", False):
                    return {
                        "status": "error",
                        "message": "图片质量不佳，建议重新拍摄。",
                        "data": {"quality": quality, "tool_status": tool_status}
                    }
                oct_result = analyze_image(image_path)
                user_mem["oct_result"] = oct_result
                update_user_memory(user_id, {"oct_result": oct_result})
                report_path = generate_report(user_id, "图片分析", oct_result)
                return {
                    "status": "ok",
                    "answer": f"OCT分析结果：{oct_result}",
                    "action": None,
                    "data": {"oct_result": oct_result, "report_path": report_path, "risk_level": self._risk_level(oct_result)}
                }
            except Exception as e:
                return {"status": "error", "message": f"图片处理失败: {e}"}

        # Structured symptoms + risk
        struct = extract_structured(text)
        risk, risk_reason = assess(text, struct)
        followups = generate_followups(struct)

        need_vision = self._should_offer_vision_test(text, struct, risk, risk_reason, user_mem)

        # 视力检测决策：不仅看关键词，还考虑风险/描述不足等情况
        if need_vision:
            if vision_tester is not None:
                try:
                    vr = vision_tester.run_test()
                    if vr is not None:
                        user_mem["vision_test"] = vr
                        update_user_memory(user_id, {"vision_test": vr})
                        # 将视力结果并入症状描述，提升回答准确性
                        text = f"{text}\n视力检测结果：{vr}"
                except Exception as e:
                    return {
                        "status": "ok",
                        "answer": "建议进行视力检测，请启动摄像头测试。",
                        "action": "vision_test",
                        "data": {
                            "risk_level": risk,
                            "followups": followups,
                            "reason": risk_reason,
                            "vision_error": str(e),
                            "vision_test": user_mem.get("vision_test"),
                        },
                    }
            else:
                return {
                    "status": "ok",
                    "answer": "建议进行视力检测，请启动摄像头测试。",
                    "action": "vision_test",
                    "data": {
                        "risk_level": risk,
                        "followups": followups,
                        "reason": risk_reason,
                        "vision_test": user_mem.get("vision_test"),
                    },
                }

        # Short input: ask for more detail
        if text and len(text.strip()) < 4:
            return {
                "status": "ok",
                "answer": "请再详细描述您的症状（如视力模糊、眼痛、飞蚊等），便于给出建议。",
                "action": None,
                "data": {"tool_status": tool_status, "followups": followups}
            }

        # Call LLM
        if tool_status["llm_error"]:
            # 模型加载失败时，提供基础回复而不是直接报错
            fallback_answer = self._generate_fallback_response(text, struct, risk, risk_reason, user_mem)
            return {
                "status": "ok",  # 改为ok，让前端能正常显示
                "answer": fallback_answer,
                "data": {
                    "tool_status": tool_status,
                    "risk_level": risk,
                    "risk_reason": risk_reason,
                    "followups": followups,
                    "llm_unavailable": True,  # 标记LLM不可用
                    "fallback": True  # 标记这是fallback回复
                }
            }
        llm_res = self.llm.get_health_advice(
            symptoms=str(context),
            vision_result=user_mem.get("vision_test"),
            oct_result=user_mem.get("oct_result"),
        )
        if isinstance(llm_res, dict):
            if llm_res.get("status") == "error":
                # 本地模型未就绪或加载失败，使用fallback
                fallback_answer = self._generate_fallback_response(text, struct, risk, risk_reason, user_mem)
                return {
                    "status": "ok",
                    "answer": fallback_answer,
                    "data": {
                        "tool_status": tool_status,
                        "risk_level": risk,
                        "risk_reason": risk_reason,
                        "followups": followups,
                        "llm_unavailable": True,
                        "fallback": True,
                        "llm_error": llm_res.get("message")
                    }
                }
            answer = llm_res.get("answer", "")
        else:
            answer = str(llm_res)

        risk = self._risk_level(text or answer)
        # 保存对话历史到记忆
        history = user_mem.get("chat_history", [])
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": answer})
        update_user_memory(user_id, {
            "last_response": answer,
            "chat_history": history[-20:]  # 保留最近20轮对话
        })
        report_path = generate_report(user_id, text or "咨询", answer)
        return {
            "status": "ok",
            "answer": answer,
            "action": None,
            "data": {
                "risk_level": risk,
                "followups": followups,
                "risk_reason": risk_reason,
                "report_path": report_path,
                "vision_test": user_mem.get("vision_test"),
                "oct_result": user_mem.get("oct_result"),
                "actions": ([{"type": "vision_test"}] if need_vision and not user_mem.get("vision_test") else []),
            },
        }

    def run_vision_test(self, user_id: str, tester) -> Dict[str, Any]:
        """Run vision test with provided tester instance."""
        try:
            result = tester.run_test()
            update_user_memory(user_id, {"vision_test": result})
            report_path = generate_report(user_id, "视力检测", f"视力检测结果：{result}")
            return {
                "status": "ok",
                "answer": f"视力检测结果：{result}",
                "action": None,
                "data": {"vision_test": result, "report_path": report_path, "risk_level": self._risk_level(result)}
            }
        except Exception as e:
            return {"status": "error", "message": f"视力检测失败: {e}"}

    def _save_uploaded_file(self, file):
        file_id = str(uuid.uuid4())
        filename = getattr(file, "filename", f"{file_id}.jpg")
        path = os.path.join(self.uploads_dir, filename)
        file.save(path)
        return path

