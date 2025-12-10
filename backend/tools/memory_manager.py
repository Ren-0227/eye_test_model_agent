import json
import os

MEMORY_FILE = "user_memory.json"

def get_user_memory(user_id):
    """获取用户记忆"""
    if not os.path.exists(MEMORY_FILE):
        return {}
    
    with open(MEMORY_FILE, 'r') as f:
        try:
            data = json.load(f)
            return data.get(user_id, {})
        except:
            return {}

def update_user_memory(user_id, updates):
    """更新用户记忆"""
    data = {}
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            try:
                data = json.load(f)
            except:
                pass
    
    user_data = data.get(user_id, {})
    user_data.update(updates)
    data[user_id] = user_data
    
    with open(MEMORY_FILE, 'w') as f:
        json.dump(data, f, indent=2)