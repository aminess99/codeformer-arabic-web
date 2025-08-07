#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CodeFormer Web Server
خادم ويب لمعالجة الصور باستخدام CodeFormer
"""

import os
import sys
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import base64
import io
from PIL import Image
import numpy as np
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.archs.codeformer_arch import CodeFormer

app = Flask(__name__)
CORS(app)

# إعدادات النموذج
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
face_helper = None

def init_model():
    """تهيئة نموذج CodeFormer"""
    global model, face_helper
    
    # تحميل النموذج
    model_path = 'weights/CodeFormer/codeformer.pth'
    if not os.path.exists(model_path):
        # تحميل النموذج المدرب مسبقاً
        model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
        os.makedirs('weights/CodeFormer', exist_ok=True)
        load_file_from_url(model_url, model_dir='weights/CodeFormer', progress=True, file_name='codeformer.pth')
    
    # إنشاء النموذج
    model = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                      connect_list=['32', '64', '128', '256']).to(device)
    
    # تحميل الأوزان
    checkpoint = torch.load(model_path, map_location=device)['params_ema']
    model.load_state_dict(checkpoint)
    model.eval()
    
    # تهيئة مساعد الوجوه
    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device
    )
    
    print("تم تحميل النموذج بنجاح!")

def process_face_restoration(img, w=0.5):
    """معالجة استعادة الوجه"""
    global model, face_helper
    
    if model is None:
        init_model()
    
    # تحويل الصورة إلى RGB
    if img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 3:  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    face_helper.clean_all()
    face_helper.read_image(img)
    
    # كشف الوجوه
    face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    face_helper.align_warp_face()
    
    # معالجة كل وجه
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        # تحويل إلى tensor
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
        
        try:
            with torch.no_grad():
                output = model(cropped_face_t, w=w, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'خطأ في معالجة الوجه {idx}: {error}')
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
        
        restored_face = restored_face.astype('uint8')
        face_helper.add_restored_face(restored_face)
    
    # دمج الوجوه المستعادة مع الصورة الأصلية
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()
    
    return restored_img

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    with open('web_interface.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/api/process', methods=['POST'])
def process_image():
    """معالجة الصورة"""
    try:
        data = request.get_json()
        
        # استخراج البيانات
        image_data = data.get('image')
        w = float(data.get('w', 0.5))
        mode = data.get('mode', 'restoration')
        
        if not image_data:
            return jsonify({'error': 'لم يتم إرسال صورة'}), 400
        
        # فك تشفير الصورة
        image_data = image_data.split(',')[1]  # إزالة data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # تحويل إلى numpy array
        img_array = np.array(image)
        
        # معالجة الصورة حسب النوع
        if mode == 'restoration':
            result_img = process_face_restoration(img_array, w)
        elif mode == 'colorization':
            # TODO: تنفيذ التلوين
            result_img = process_face_restoration(img_array, w)
        elif mode == 'inpainting':
            # TODO: تنفيذ ملء الفراغات
            result_img = process_face_restoration(img_array, w)
        else:
            return jsonify({'error': 'نوع معالجة غير مدعوم'}), 400
        
        # تحويل النتيجة إلى base64
        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        result_pil.save(buffer, format='PNG')
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'result_image': f'data:image/png;base64,{result_base64}'
        })
        
    except Exception as e:
        print(f'خطأ في معالجة الصورة: {str(e)}')
        return jsonify({'error': f'خطأ في معالجة الصورة: {str(e)}'}), 500

@app.route('/api/status')
def status():
    """حالة الخادم"""
    return jsonify({
        'status': 'running',
        'device': str(device),
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("بدء تشغيل خادم CodeFormer...")
    print(f"الجهاز المستخدم: {device}")
    
    # تهيئة النموذج
    try:
        init_model()
    except Exception as e:
        print(f"تحذير: فشل في تحميل النموذج: {e}")
        print("سيتم تحميل النموذج عند أول طلب معالجة")
    
    # تشغيل الخادم
    app.run(host='0.0.0.0', port=5000, debug=True)