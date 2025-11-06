from flask import Flask, request, jsonify, render_template
from api_integration import DeepseekAPI
from image_processing import OCTPredictor
from vision_test import VisionTestApp
from memory_manager import get_user_memory, update_user_memory
import uuid
import os

# ... 顶部引入部分保持不变 ...

# 修改初始化部分
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 添加超时设置和错误处理
VISION_TEST_TIMEOUT = 300  # 5分钟超时
CAMERA_INIT_TIMEOUT = 30   # 30秒摄像头初始化超时

# 修改核心模块初始化方式
api_handler = DeepseekAPI()
oct_predictor = OCTPredictor()
# 延迟初始化视力检测模块（解决摄像头初始化阻塞问题）
vision_tester = None  

class MedicalAssistant:
    def __init__(self):
        self.symptom_history = {}
        self.vision_test_lock = threading.Lock()  # 添加线程锁
        
    def _initiate_vision_test(self, user_id):
        global vision_tester
        try:
            # 延迟初始化并添加超时
            with self.vision_test_lock:
                if vision_tester is None:
                    from vision_test import VisionTester
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
                return {'action': 'vision_test', 'result': vision_result}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def process_request(self, user_id, text_input=None, image_file=None):
        # 获取用户记忆
        user_memory = get_user_memory(user_id)
        
        # 处理图片上传
        if image_file:
            image_path = self._save_uploaded_file(image_file)
            oct_result = oct_predictor.predict(image_path)
            user_memory['oct_result'] = oct_result
            update_user_memory(user_id, {'oct_result': oct_result})

        # 构建对话上下文
        context = {
            'current_symptoms': text_input,
            'medical_history': user_memory.get('history', []),
            'oct_result': user_memory.get('oct_result'),
            'vision_test': user_memory.get('vision_test')
        }

        # 判断是否需要视力检测
        if self._needs_vision_test(text_input, context):
            return self._initiate_vision_test(user_id)

        # 获取大模型诊断
        diagnosis = api_handler.get_health_advice(
            symptoms=str(context),
            vision_result=user_memory.get('vision_test')
        )
        
        # 更新历史记录
        self._update_diagnosis_history(user_id, diagnosis)
        return diagnosis

    def _needs_vision_test(self, text_input, context):
        # 通过API判断是否需要视力检测
        check_prompt = f"用户输入：{text_input}，是否需要视力检测？只需回答yes或no"
        response = api_handler._call_api(check_prompt)
        return 'yes' in response.lower()

    def _save_uploaded_file(self, file):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return file_path

    def _update_diagnosis_history(self, user_id, diagnosis):
        history = self.symptom_history.get(user_id, [])
        history.append(diagnosis)
        self.symptom_history[user_id] = history[-5:]  # 保留最近5条记录

assistant = MedicalAssistant()

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_id = request.cookies.get('user_id', str(uuid.uuid4()))
    text_input = request.json.get('text', '')
    image_file = request.files.get('image')
    
    response = assistant.process_request(user_id, text_input, image_file)
    
    if 'action' in response:
        return jsonify({'action': response['action'], 'data': response['result']})
    return jsonify(response)

@app.route('/')
def index():
    return render_template('index.html')

# 添加健康检查接口
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'alive',
        'components': {
            'camera': vision_tester is not None,
            'model_loaded': True
        }
    })

if __name__ == '__main__':
    # 添加多线程支持
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)