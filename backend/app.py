from flask import Flask, request, jsonify, render_template
import uuid
import os
import sys
import threading
import queue
import json
import datetime

# 调试：打印模块名，确认是否以脚本方式运行
print(f"[app] module loaded, __name__ = {__name__}", flush=True)

# 确保可以从项目根目录导入 backend 包
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

print("[app] Starting imports...", flush=True)
try:
    from backend.tools.local_qwen_api import LocalQwenAPI
    print("[app] LocalQwenAPI imported", flush=True)
except Exception as e:
    print(f"[app] Failed to import LocalQwenAPI: {e}", flush=True)
    import traceback
    traceback.print_exc()

try:
    from backend.orchestrator import Orchestrator
    print("[app] Orchestrator imported", flush=True)
except Exception as e:
    print(f"[app] Failed to import Orchestrator: {e}", flush=True)
    import traceback
    traceback.print_exc()

try:
    from backend.tools.image_processing import OCTClassifier, analyze_image  # noqa: F401 (预留)
    print("[app] image_processing imported (OCT model may load here)", flush=True)
except Exception as e:
    print(f"[app] Failed to import image_processing: {e}", flush=True)
    import traceback
    traceback.print_exc()

try:
    from backend.tools.memory_manager import get_user_memory, update_user_memory
    print("[app] memory_manager imported", flush=True)
except Exception as e:
    print(f"[app] Failed to import memory_manager: {e}", flush=True)
    import traceback
    traceback.print_exc()

# vision_test 现在使用延迟导入 mediapipe，模块级导入应该不会失败
try:
    from backend.tools.vision_test import VisionTester, check_camera_available  # noqa: F401
    print("[app] vision_test imported", flush=True)
except Exception as e:
    print(f"[app] Failed to import vision_test: {e}", flush=True)
    import traceback
    traceback.print_exc()
    # 如果导入失败，设置为 None，后续使用时会报错
    VisionTester = None
    check_camera_available = None

try:
    from backend.tools.resource_check import check_resources
    print("[app] resource_check imported", flush=True)
except Exception as e:
    print(f"[app] Failed to import resource_check: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("[app] All imports completed", flush=True)

# 修改初始化部分
print("[app] Creating Flask app...", flush=True)
# 配置模板目录指向项目根目录的 templates 文件夹
template_dir = os.path.join(BASE_DIR, 'templates')
static_dir = os.path.join(BASE_DIR, 'static') if os.path.exists(os.path.join(BASE_DIR, 'static')) else None
app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)
print(f"[app] Flask app created, template_dir: {template_dir}", flush=True)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
AVATAR_FOLDER = os.path.join(UPLOAD_FOLDER, "avatars")
os.makedirs(AVATAR_FOLDER, exist_ok=True)
DATA_STORE = os.path.join("data", "api_store")
os.makedirs(DATA_STORE, exist_ok=True)
print("[app] Directories created", flush=True)

# 添加超时设置和错误处理
VISION_TEST_TIMEOUT = 300  # 5分钟超时
CAMERA_INIT_TIMEOUT = 30   # 30秒摄像头初始化超时

# 延迟初始化 orchestrator，避免在导入时阻塞
print("[app] Initializing global variables...", flush=True)
orchestrator = None
vision_tester = None  # 延迟初始化视力检测模块
print("[app] Global variables initialized", flush=True)

def get_orchestrator():
    """延迟初始化 orchestrator，确保只在需要时加载模型"""
    global orchestrator
    if orchestrator is None:
        print("[app] Initializing orchestrator (this may take a while for model loading)...", flush=True)
        try:
            orchestrator = Orchestrator(uploads_dir=UPLOAD_FOLDER)
            print(f"[app] Orchestrator initialized. LLM load_error: {orchestrator.llm.load_error}", flush=True)
        except Exception as e:
            print(f"[app] Failed to initialize orchestrator: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
    return orchestrator

class MedicalAssistant:
    def __init__(self):
        self.symptom_history = {}
        self.vision_test_lock = threading.Lock()  # 添加线程锁
        self.vision_tester = None
        
    def _initiate_vision_test(self, user_id):
        global vision_tester
        try:
            # 延迟初始化并添加超时
            with self.vision_test_lock:
                if VisionTester is None:
                    raise RuntimeError("视力检测模块未加载，请检查依赖安装。")
                if check_camera_available is None or not check_camera_available():
                    raise RuntimeError("未检测到可用摄像头，请检查连接与权限。")
                if vision_tester is None:
                    vision_tester = VisionTester()
                    
                # 添加测试超时保护
                result_queue = queue.Queue()
                test_thread = threading.Thread(
                    target=lambda q: q.put(vision_tester.run_test()),
                    args=(result_queue,)
                )
                test_thread.start()
                test_thread.join(VISION_TEST_TIMEOUT)
                
                if test_thread.is_alive():
                    raise TimeoutError("视力检测超时")
                    
                vision_result = result_queue.get()
                update_user_memory(user_id, {'vision_test': vision_result})
                return {
                    'status': 'ok',
                    'action': None,
                    'answer': f'视力检测结果：{vision_result}',
                    'data': {'vision_test': vision_result}
                }
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def process_request(self, user_id, text_input=None, image_file=None):
        try:
            print(f"[app] Processing request for user_id={user_id}, text={text_input[:50] if text_input else None}", flush=True)
            # 确保有 vision_tester 可用（懒加载）
            tester = self.vision_tester
            if tester is None:
                with self.vision_test_lock:
                    if self.vision_tester is None:
                        try:
                            if VisionTester is None:
                                raise ImportError("VisionTester not available")
                            self.vision_tester = VisionTester()
                        except Exception as e:
                            # 如果摄像头不可用，则后续会提示前端做视力检测
                            print(f"[vision] 摄像头初始化失败: {e}")
                    tester = self.vision_tester

            orch = get_orchestrator()
            result = orch.process(user_id, text=text_input, image_file=image_file, vision_tester=tester)
            print(f"[app] Orchestrator returned: status={result.get('status')}, has_answer={bool(result.get('answer'))}", flush=True)
            return result
        except Exception as e:
            print(f"[app] Error in process_request: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"处理请求时出错: {str(e)}"
            }

    def _update_diagnosis_history(self, user_id, diagnosis):
        history = self.symptom_history.get(user_id, [])
        history.append(diagnosis)
        self.symptom_history[user_id] = history[-5:]  # 保留最近5条记录

assistant = MedicalAssistant()

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    # 优先从request.json获取user_id，其次从cookies，最后生成新的
    user_id = None
    if request.json:
        user_id = request.json.get('user_id')
    if not user_id:
        user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
    
    text_input = request.json.get('text', '') if request.json else ''
    image_file = request.files.get('image')
    
    try:
        response = assistant.process_request(user_id, text_input, image_file)
        # 确保响应格式正确
        if not isinstance(response, dict):
            response = {"status": "error", "message": "Invalid response format"}
        elif "status" not in response:
            response["status"] = "ok"
        # 在响应中设置cookie以便后续请求使用
        resp = jsonify(response)
        resp.set_cookie('user_id', user_id, max_age=365*24*60*60)  # 1年有效期
        return resp
    except Exception as e:
        print(f"[app] Error in chat_endpoint: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"处理请求时出错: {str(e)}"
        }), 500


@app.route('/action/vision-test', methods=['POST'])
def vision_test_endpoint():
    user_id = request.json.get('user_id', str(uuid.uuid4()))
    response = assistant._initiate_vision_test(user_id)
    return jsonify(response)

@app.route('/')
def index():
    """主页面 - 返回完整的ui5.html"""
    # 直接读取并返回ui5.html文件内容
    ui5_path = os.path.join(BASE_DIR, 'ui5.html')
    if os.path.exists(ui5_path):
        with open(ui5_path, 'r', encoding='utf-8') as f:
            return f.read()
    # 如果ui5.html不存在，尝试返回templates/index.html
    return render_template('index.html')

@app.route('/demo')
def demo():
    """演示页面 - 返回简化的ui演示.html"""
    ui_demo_path = os.path.join(BASE_DIR, 'ui演示.html')
    if os.path.exists(ui_demo_path):
        with open(ui_demo_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Demo page not found", 404

# 添加健康检查接口
@app.route('/health')
def health_check():
    try:
        orch = get_orchestrator()
        # 资源检查（包含摄像头/模型等）
        resources = check_resources(orch.llm.load_error, getattr(orch.llm, "model_path", None))
        # LLM 烟囱测试
        smoke = orch.llm.run_smoke_test()
        return jsonify({
            'status': 'alive',
            'components': {
                'camera_initialized': vision_tester is not None,
                **resources,
                'llm_smoke': smoke
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# === 简易持久化工具 ===
def _store_path(name: str):
    return os.path.join(DATA_STORE, f"{name}.json")


def _load_json(name: str, default):
    path = _store_path(name)
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(name: str, data):
    path = _store_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# === Profile ===
@app.route("/api/profile/info", methods=["GET"])
def api_profile_info():
    # 从cookie或request获取user_id，如果没有则使用默认
    user_id = request.cookies.get('user_id') or request.args.get('user_id') or 'default_user'
    # 尝试从用户记忆中获取个人信息
    user_mem = get_user_memory(user_id)
    data = _load_json("profile_info", {
        "name": "演示用户",
        "age": 16,
        "mode": "青少年模式",
        "avatar": None,
        "vision": user_mem.get("vision_test") and {"left": user_mem["vision_test"], "right": user_mem["vision_test"]} or {"left": 0.8, "right": 0.7}
    })
    return jsonify(data)


@app.route("/api/profile/update", methods=["PUT"])
def api_profile_update():
    payload = request.json or {}
    _save_json("profile_info", payload)
    return jsonify({"status": "ok", "data": payload})


@app.route("/api/profile/vision-trend", methods=["GET"])
def api_profile_vision_trend():
    months = int(request.args.get("months", 6))
    data = _load_json("vision_trend", [
        {"month": m, "left": 1.0 - 0.05 * (i % 3), "right": 0.9 - 0.05 * (i % 3)}
        for i, m in enumerate(range(1, months + 1))
    ])
    return jsonify(data[:months])


@app.route("/api/profile/avatar", methods=["POST"])
def api_profile_avatar():
    file = request.files.get("file")
    if not file:
        return jsonify({"status": "error", "message": "未找到上传文件"}), 400
    fname = f"{uuid.uuid4().hex}_{file.filename}"
    path = os.path.join(AVATAR_FOLDER, fname)
    file.save(path)
    info = _load_json("profile_info", {})
    info["avatar"] = f"/{path}"
    _save_json("profile_info", info)
    return jsonify({"status": "ok", "avatar": info["avatar"]})


# === Reminders ===
@app.route("/api/reminder/settings", methods=["GET", "PUT"])
def api_reminder_settings():
    if request.method == "GET":
        data = _load_json("reminder_settings", {"interval": 60, "enabled": True})
        return jsonify(data)
    payload = request.json or {}
    _save_json("reminder_settings", payload)
    return jsonify({"status": "ok", "data": payload})


@app.route("/api/reminder/history", methods=["GET"])
def api_reminder_history():
    data = _load_json("reminder_history", [])
    return jsonify(data)


# === Disease Diagnosis ===
@app.route("/api/diagnosis/disease", methods=["GET"])
def api_diagnosis_disease():
    """获取疾病信息"""
    disease_type = request.args.get("type", "")
    
    disease_info = {
        "myopia": {
            "name": "近视",
            "description": "近视（Myopia）是指眼睛在调节放松状态下，平行光线经眼球屈光系统后聚焦在视网膜之前，导致看远处物体模糊不清的眼部疾病。",
            "symptoms": ["看远处物体模糊", "需要眯眼才能看清", "眼睛疲劳", "头痛"],
            "causes": ["遗传因素", "长时间近距离用眼", "环境因素", "缺乏户外活动"],
            "treatment": ["配戴眼镜或隐形眼镜", "角膜塑形镜", "激光手术", "定期检查"],
            "prevention": ["保持正确的用眼姿势", "增加户外活动时间", "控制用眼时间", "定期检查视力"]
        },
        "hyperopia": {
            "name": "远视",
            "description": "远视（Hyperopia）是指眼睛在调节放松状态下，平行光线经眼球屈光系统后聚焦在视网膜之后，导致看近处物体模糊不清的眼部疾病。",
            "symptoms": ["看近处物体模糊", "眼睛疲劳", "头痛", "阅读困难"],
            "causes": ["眼球轴长过短", "角膜曲率过小", "晶状体屈光力不足", "遗传因素"],
            "treatment": ["配戴凸透镜", "隐形眼镜", "激光手术", "定期检查"],
            "prevention": ["定期检查视力", "保持良好用眼习惯", "及时矫正"]
        },
        "astigmatism": {
            "name": "散光",
            "description": "散光（Astigmatism）是指眼球在不同子午线上的屈光力不同，导致平行光线经过眼球屈光系统后不能形成一个焦点，而是形成两条焦线，造成视物模糊、变形。",
            "symptoms": ["视物模糊", "视物变形", "眼睛疲劳", "头痛", "夜间视力差"],
            "causes": ["角膜曲率不规则", "晶状体曲率异常", "遗传因素", "眼部手术"],
            "treatment": ["配戴柱镜", "隐形眼镜", "激光手术", "定期检查"],
            "prevention": ["定期检查视力", "保持良好用眼习惯", "避免揉眼"]
        },
        "glaucoma": {
            "name": "青光眼",
            "description": "青光眼（Glaucoma）是一组以视神经萎缩和视野缺损为共同特征的疾病，主要与眼压升高有关。如果不及时治疗，可能导致永久性视力丧失。",
            "symptoms": ["眼压升高", "视野缺损", "视力下降", "眼痛", "头痛", "恶心呕吐"],
            "causes": ["眼压升高", "遗传因素", "年龄增长", "眼部外伤", "某些药物"],
            "treatment": ["降眼压药物", "激光治疗", "手术治疗", "定期监测"],
            "prevention": ["定期检查眼压", "控制血压", "避免长时间低头", "及时治疗"]
        }
    }
    
    if disease_type and disease_type in disease_info:
        return jsonify(disease_info[disease_type])
    
    # 返回所有疾病列表
    return jsonify({
        "diseases": list(disease_info.keys()),
        "info": disease_info
    })


# === Help Center ===
@app.route("/api/help/faqs", methods=["GET"])
def api_help_faqs():
    category = request.args.get("category")
    faqs = _load_json("faqs", [
        {"q": "如何缓解眼睛干涩？", "a": "减少屏幕时间，使用人工泪液，保证休息。", "category": "dry"},
        {"q": "视力突然下降怎么办？", "a": "立即停止用眼，尽快就医检查。", "category": "blur"},
    ])
    if category:
        faqs = [f for f in faqs if f.get("category") == category]
    return jsonify(faqs)


@app.route("/api/help/search", methods=["GET"])
def api_help_search():
    keyword = request.args.get("keyword", "")
    faqs = _load_json("faqs", [])
    results = [f for f in faqs if keyword in f.get("q", "") or keyword in f.get("a", "")]
    return jsonify(results)


@app.route("/api/help/feedback", methods=["POST"])
def api_help_feedback():
    feedback = request.json or {}
    items = _load_json("feedback", [])
    feedback["timestamp"] = datetime.datetime.utcnow().isoformat()
    items.append(feedback)
    _save_json("feedback", items)
    return jsonify({"status": "ok"})


# === Vision test history ===
@app.route("/api/vision/history", methods=["GET"])
def api_vision_history():
    # 从cookie或request获取user_id
    user_id = request.cookies.get('user_id') or request.args.get('user_id')
    if user_id:
        # 从用户记忆中获取视力检测历史
        user_mem = get_user_memory(user_id)
        vision_test = user_mem.get("vision_test")
        if vision_test:
            # 转换为前端期望的格式
            data = [{
                "leftEye": vision_test if isinstance(vision_test, (int, float)) else 0.8,
                "rightEye": vision_test if isinstance(vision_test, (int, float)) else 0.7,
                "testType": "standard",
                "testTime": user_mem.get("last_vision_test_time", datetime.datetime.utcnow().isoformat())
            }]
            return jsonify(data)
    # 返回全局历史（兼容旧数据）
    data = _load_json("vision_history", [])
    return jsonify(data)


# === Health Data ===
@app.route("/api/profile/health-data", methods=["GET"])
def api_profile_health_data():
    """获取用户健康数据（当前视力、护眼时长等）"""
    user_id = request.cookies.get('user_id') or request.args.get('user_id')
    if not user_id:
        return jsonify({
            "current_vision": 0.0,
            "vision_change": 0.0,
            "eye_care_hours": 0,
            "eye_care_change": 0
        })
    
    user_mem = get_user_memory(user_id)
    vision_test = user_mem.get("vision_test")
    
    # 计算当前视力（取左右眼平均值）
    current_vision = 0.0
    if vision_test:
        if isinstance(vision_test, (int, float)):
            current_vision = float(vision_test)
        elif isinstance(vision_test, dict):
            left = vision_test.get("left", vision_test.get("leftEye", 0))
            right = vision_test.get("right", vision_test.get("rightEye", 0))
            current_vision = (float(left) + float(right)) / 2
    
    # 获取历史视力数据计算变化
    vision_history = user_mem.get("vision_history", [])
    vision_change = 0.0
    if len(vision_history) >= 2:
        # 计算与上个月的差值
        latest = vision_history[-1] if isinstance(vision_history[-1], (int, float)) else 0.8
        previous = vision_history[-2] if isinstance(vision_history[-2], (int, float)) else 0.8
        vision_change = round(float(latest) - float(previous), 1)
    elif len(vision_history) == 1:
        latest = vision_history[0] if isinstance(vision_history[0], (int, float)) else 0.8
        vision_change = round(float(current_vision) - float(latest), 1)
    
    # 护眼时长（从聊天历史估算，每次对话约5分钟）
    chat_history = user_mem.get("chat_history", [])
    # 本周的对话次数（简化计算）
    eye_care_hours = len(chat_history) * 5 / 60  # 转换为小时
    eye_care_change = 3  # 简化：固定值，实际应该计算
    
    return jsonify({
        "current_vision": round(current_vision, 1),
        "vision_change": round(vision_change, 1),
        "eye_care_hours": round(eye_care_hours, 1),
        "eye_care_change": eye_care_change
    })


@app.route("/api/vision/test", methods=["POST"])
def api_vision_test_submit():
    payload = request.json or {}
    user_id = request.cookies.get('user_id') or payload.get('user_id')
    
    # 保存到用户记忆
    if user_id:
        vision_result = {
            "leftEye": payload.get("leftEye", payload.get("vision_test", 0.8)),
            "rightEye": payload.get("rightEye", payload.get("vision_test", 0.7)),
        }
        update_user_memory(user_id, {
            "vision_test": (vision_result["leftEye"] + vision_result["rightEye"]) / 2,
            "last_vision_test_time": datetime.datetime.utcnow().isoformat()
        })
    
    # 同时保存到全局历史（兼容）
    history = _load_json("vision_history", [])
    payload["timestamp"] = datetime.datetime.utcnow().isoformat()
    history.append(payload)
    _save_json("vision_history", history)
    return jsonify({"status": "ok"})


# === AI chat history ===
@app.route("/api/ai/chat/history", methods=["GET"])
def api_ai_chat_history():
    # 从cookie或request获取user_id
    user_id = request.cookies.get('user_id') or request.args.get('user_id')
    if user_id:
        # 从用户记忆中获取聊天历史
        user_mem = get_user_memory(user_id)
        chat_history = user_mem.get("chat_history", [])
        # 转换为前端期望的格式
        data = []
        for msg in chat_history:
            if isinstance(msg, dict) and "role" in msg:
                data.append({
                    "isUser": msg["role"] == "user",
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("timestamp", datetime.datetime.utcnow().isoformat())
                })
        return jsonify(data)
    # 返回全局历史（兼容旧数据）
    data = _load_json("chat_history", [])
    return jsonify(data)


@app.route("/api/ai/chat/save", methods=["POST"])
def api_ai_chat_save():
    payload = request.json or {}
    user_id = request.cookies.get('user_id') or payload.get('user_id')
    
    # 保存到用户记忆
    if user_id:
        user_mem = get_user_memory(user_id)
        history = user_mem.get("chat_history", [])
        # 添加用户消息
        if payload.get("userMessage"):
            history.append({
                "role": "user",
                "content": payload["userMessage"],
                "timestamp": payload.get("timestamp", datetime.datetime.utcnow().isoformat())
            })
        # 添加AI回复
        if payload.get("aiResponse"):
            history.append({
                "role": "assistant",
                "content": payload["aiResponse"],
                "timestamp": payload.get("timestamp", datetime.datetime.utcnow().isoformat())
            })
        update_user_memory(user_id, {"chat_history": history[-50:]})  # 保留最近50条
    
    # 同时保存到全局历史（兼容）
    history = _load_json("chat_history", [])
    history.append(payload)
    _save_json("chat_history", history)
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    print("[app] Checking __name__ == '__main__'", flush=True)
    print(f"[app] __name__ = {__name__}", flush=True)
    try:
        print("[app] __main__ entered", flush=True)
        print(f"[app] CWD: {os.getcwd()}", flush=True)
        print("[app] Note: Orchestrator will be initialized on first request (lazy loading)", flush=True)
        if VisionTester is None:
            print("[app] Warning: Vision test module not available. Vision test features will be disabled.", flush=True)
        print("[app] Starting backend server on http://127.0.0.1:5000 ...", flush=True)
        print("[app] Server is ready! Model will load when first request arrives.", flush=True)
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[app] Server stopped by user", flush=True)
    except Exception as e:
        print(f"[app] Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("[app] Exiting", flush=True)
else:
    print(f"[app] Not running as main script, __name__ = {__name__}", flush=True)