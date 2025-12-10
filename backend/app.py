"""
Flask Web服务，提供HTTP API接口
"""
import os
import sys
import json
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 确保能从项目根导入 backend 包
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from backend.orchestrator import Orchestrator
from backend.tools.vision_test import VisionTester, check_camera_available
from backend.tools.memory_manager import get_user_memory

app = Flask(__name__, static_folder='../uploads', static_url_path='/uploads')
CORS(app)  # 允许跨域请求

# 配置
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 懒加载：避免启动时阻塞
_orchestrator = None
_vision_tester = None


def get_orchestrator():
    """懒加载Orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def get_vision_tester():
    """懒加载VisionTester"""
    global _vision_tester
    if _vision_tester is None:
        try:
            if check_camera_available():
                _vision_tester = VisionTester()
            else:
                _vision_tester = None
        except Exception as e:
            print(f"[app] VisionTester初始化失败: {e}")
            _vision_tester = None
    return _vision_tester


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_user_id():
    """从请求中获取或生成user_id"""
    # 优先从JSON body获取
    if request.is_json and request.json:
        user_id = request.json.get('user_id')
        if user_id:
            return user_id
    
    # 从cookies获取
    user_id = request.cookies.get('user_id')
    if user_id:
        return user_id
    
    # 生成新的user_id
    return f"user_{uuid.uuid4().hex[:8]}"


@app.route('/')
def index():
    """返回前端页面（ui5.html）"""
    frontend_path = os.path.join(BASE_DIR, "ui5.html")
    if os.path.exists(frontend_path):
        return send_from_directory(BASE_DIR, "ui5.html")
    else:
        return """
        <html>
            <head><title>眼部医疗AI助手</title></head>
            <body>
                <h1>眼部医疗AI助手后端服务</h1>
                <p>请访问前端页面：<a href="/ui5.html">前端页面</a></p>
                <p>API文档：</p>
                <ul>
                    <li>POST /chat - 对话接口</li>
                    <li>POST /action/vision-test - 视力检测接口</li>
                    <li>GET /api/profile/health-data - 获取用户健康数据</li>
                    <li>GET /api/diagnosis/disease - 获取疾病详情</li>
                </ul>
            </body>
        </html>
        """


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """对话接口，处理用户文本和图像输入"""
    try:
        user_id = get_user_id()
        orch = get_orchestrator()
        
        # 获取文本输入
        text = None
        image_file = None
        
        if request.is_json:
            # JSON格式请求
            data = request.json
            text = data.get('text', '').strip() or None
        else:
            # FormData格式请求（包含文件）
            text = request.form.get('text', '').strip() or None
            if 'image' in request.files:
                image_file = request.files['image']
                if image_file.filename == '':
                    image_file = None
                elif not allowed_file(image_file.filename):
                    return jsonify({
                        "status": "error",
                        "message": "不支持的文件格式，请上传图片文件（png, jpg, jpeg, gif, bmp）"
                    }), 400
        
        # 处理图像文件 - orchestrator的_save_uploaded_file需要文件对象
        # 如果上传了图片，直接传递文件对象给orchestrator
        vision_tester = get_vision_tester()
        result = orch.process(
            user_id=user_id,
            text=text,
            image_file=image_file,  # 直接传递Flask的FileStorage对象
            vision_tester=vision_tester
        )
        
        # 设置cookie保存user_id
        response = jsonify(result)
        response.set_cookie('user_id', user_id, max_age=60*60*24*365)  # 1年有效期
        
        return response
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[app] /chat 错误: {error_detail}")
        return jsonify({
            "status": "error",
            "message": f"处理请求时出错: {str(e)}"
        }), 500


@app.route('/action/vision-test', methods=['POST'])
def vision_test_endpoint():
    """执行视力检测"""
    try:
        user_id = get_user_id()
        vision_tester = get_vision_tester()
        
        if vision_tester is None:
            return jsonify({
                "status": "error",
                "message": "摄像头不可用，无法进行视力检测。请检查摄像头连接和权限。"
            }), 400
        
        orch = get_orchestrator()
        result = orch.run_vision_test(user_id, vision_tester)
        
        response = jsonify(result)
        response.set_cookie('user_id', user_id, max_age=60*60*24*365)
        
        return response
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[app] /action/vision-test 错误: {error_detail}")
        return jsonify({
            "status": "error",
            "message": f"视力检测失败: {str(e)}"
        }), 500


@app.route('/api/profile/health-data', methods=['GET'])
def get_health_data():
    """获取用户健康数据"""
    try:
        user_id = get_user_id()
        memory = get_user_memory(user_id)
        
        vision_test = memory.get("vision_test", {})
        current_vision = None
        if isinstance(vision_test, dict):
            left = vision_test.get("left", 0)
            right = vision_test.get("right", 0)
            if left and right:
                current_vision = (left + right) / 2
        
        # 模拟数据（实际应该从历史记录计算）
        return jsonify({
            "status": "ok",
            "current_vision": current_vision or 0.7,
            "vision_change": 0.1,
            "eye_care_hours": 12,
            "eye_care_change": 3,
            "vision_test": vision_test
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"获取健康数据失败: {str(e)}"
        }), 500


@app.route('/api/diagnosis/disease', methods=['GET'])
def get_disease_info():
    """获取疾病详情"""
    try:
        disease_type = request.args.get('type', '')
        
        # 疾病知识库（简化版，实际可以从数据库或知识库加载）
        disease_db = {
            "近视": {
                "name": "近视",
                "description": "近视是指眼睛在调节放松状态下，平行光线经眼球屈光系统后聚焦在视网膜之前，导致看远处物体模糊不清的眼部疾病。",
                "symptoms": ["看远模糊", "眯眼", "视疲劳", "头痛"],
                "causes": ["遗传因素", "长时间近距离用眼", "环境因素"],
                "treatment": ["佩戴眼镜", "隐形眼镜", "激光手术", "ICL晶体植入"],
                "prevention": ["20-20-20法则", "增加户外活动", "保持正确用眼姿势", "定期检查视力"]
            },
            "远视": {
                "name": "远视",
                "description": "远视是指眼睛在调节放松状态下，平行光线经眼球屈光系统后聚焦在视网膜之后，导致看近处物体模糊不清的眼部疾病。",
                "symptoms": ["看近模糊", "视疲劳", "头痛", "眼胀"],
                "causes": ["遗传因素", "眼轴过短", "角膜曲率过小"],
                "treatment": ["佩戴眼镜", "隐形眼镜", "激光手术"],
                "prevention": ["定期检查", "注意用眼卫生", "避免过度用眼"]
            },
            "AMD": {
                "name": "年龄相关性黄斑变性（AMD）",
                "description": "AMD是一种影响黄斑的退行性眼病，是导致老年人失明的主要原因之一。",
                "symptoms": ["中心视力下降", "视物变形", "中心暗点"],
                "causes": ["年龄增长", "遗传因素", "吸烟", "高血压"],
                "treatment": ["抗VEGF治疗", "激光治疗", "营养补充"],
                "prevention": ["戒烟", "控制血压", "补充叶黄素", "定期检查"]
            },
            "CNV": {
                "name": "脉络膜新生血管（CNV）",
                "description": "CNV是异常血管在脉络膜中生长，可能导致视网膜出血和视力丧失。",
                "symptoms": ["视力突然下降", "视物变形", "中心暗点"],
                "causes": ["AMD", "高度近视", "外伤"],
                "treatment": ["抗VEGF治疗", "激光治疗", "光动力疗法"],
                "prevention": ["定期检查", "控制基础疾病"]
            }
        }
        
        if disease_type and disease_type in disease_db:
            return jsonify({
                "status": "ok",
                **disease_db[disease_type]
            })
        else:
            return jsonify({
                "status": "ok",
                "name": disease_type or "未知疾病",
                "description": "暂无该疾病的详细信息",
                "symptoms": [],
                "causes": [],
                "treatment": [],
                "prevention": []
            })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"获取疾病信息失败: {str(e)}"
        }), 500


@app.route('/api/tool-status', methods=['GET'])
def tool_status():
    """获取工具状态"""
    try:
        orch = get_orchestrator()
        status = orch.tool_status()
        return jsonify({
            "status": "ok",
            **status
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"获取工具状态失败: {str(e)}"
        }), 500


@app.route('/api/user/memory', methods=['GET'])
def get_user_memory_endpoint():
    """获取用户记忆"""
    try:
        user_id = get_user_id()
        memory = get_user_memory(user_id)
        return jsonify({
            "status": "ok",
            "user_id": user_id,
            "memory": memory
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"获取用户记忆失败: {str(e)}"
        }), 500


if __name__ == '__main__':
    print("[app] 启动Flask服务...")
    print("[app] 访问地址: http://127.0.0.1:5000")
    print("[app] API文档: http://127.0.0.1:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)

