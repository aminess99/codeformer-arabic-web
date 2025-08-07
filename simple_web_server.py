from flask import Flask, render_template_string, request, jsonify, send_file
from flask_cors import CORS
import os
import base64
from PIL import Image
import io
import json
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.utils.registry import ARCH_REGISTRY

app = Flask(__name__)
CORS(app)

# Global variables for models
net = None
face_helper = None
bg_upsampler = None
device = None

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def initialize_models():
    global net, face_helper, bg_upsampler, device
    
    try:
        device = get_device()
        print(f"Using device: {device}")
        
        # Initialize CodeFormer network
        net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                                connect_list=['32', '64', '128', '256']).to(device)
        
        # Load checkpoint
        ckpt_path = os.path.join('weights', 'CodeFormer', 'codeformer.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)['params_ema']
            net.load_state_dict(checkpoint)
            net.eval()
            print("CodeFormer model loaded successfully")
        else:
            print(f"Model file not found: {ckpt_path}")
            return False
        
        # Initialize face helper
        face_helper = FaceRestoreHelper(
            upscale_factor=2,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=device
        )
        print("Face helper initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

def process_image_with_codeformer(image_data, fidelity_weight=0.5, enhance_background=False):
    global net, face_helper, device
    
    try:
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, "Failed to decode image"
        
        # Clean previous results
        face_helper.clean_all()
        
        # Read image and detect faces
        face_helper.read_image(img)
        num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5)
        
        if num_det_faces == 0:
            return None, "No faces detected in the image"
        
        print(f"Detected {num_det_faces} faces")
        
        # Align and warp faces
        face_helper.align_warp_face()
        
        # Process each face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # Prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
            
            try:
                with torch.no_grad():
                    output = net(cropped_face_t, w=fidelity_weight, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'Failed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
            
            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)
        
        # Get inverse affine matrix
        face_helper.get_inverse_affine(None)
        
        # Paste faces back to original image
        restored_img = face_helper.paste_faces_to_input_image(upsample_img=None, draw_box=False)
        
        # Background enhancement if requested
        if enhance_background:
            print(f"Background enhancement requested: {enhance_background}")
            try:
                # Use RealESRGAN for background enhancement
                from basicsr.utils.realesrgan_utils import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
                
                print("Creating RRDBNet model...")
                # Create RRDBNet model for RealESRGAN
                realesrgan_model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                )
                
                print("Initializing RealESRGANer...")
                upsampler = RealESRGANer(
                    scale=2,
                    model_path='weights/realesrgan/RealESRGAN_x2plus.pth',
                    model=realesrgan_model,
                    tile=400,
                    tile_pad=10,
                    pre_pad=0,
                    half=False
                )
                
                print("Enhancing image with RealESRGAN...")
                # Enhance the entire image
                enhanced_img, _ = upsampler.enhance(restored_img, outscale=2)
                restored_img = enhanced_img
                print("Background enhanced with RealESRGAN successfully!")
            except Exception as e:
                print(f"Background enhancement failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Background enhancement not requested")
        
        # Convert result to base64
        _, buffer = cv2.imencode('.png', restored_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}", "Success"
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, str(e)

# Simple HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CodeFormer - تحسين الصور</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 40px;
        }
        
        .upload-section {
            background: #f8f9fa;
            border: 3px dashed #dee2e6;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: #667eea;
            background: #f0f2ff;
        }
        
        .upload-section.dragover {
            border-color: #667eea;
            background: #e3f2fd;
            transform: scale(1.02);
        }
        
        .upload-icon {
            font-size: 4em;
            color: #667eea;
            margin-bottom: 20px;
        }
        
        .file-input {
            display: none;
        }
        
        .upload-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .control-group {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        
        .control-group label {
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
            color: #333;
        }
        
        .slider {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
        }
        
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }
        
        .select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
        }
        
        .process-btn {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 25px;
            font-size: 1.2em;
            cursor: pointer;
            width: 100%;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .process-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(40, 167, 69, 0.3);
        }
        
        .process-btn:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        
        .image-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .image-container h3 {
            text-align: center;
            margin-bottom: 15px;
            color: #333;
        }
        
        .image-container img {
            width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .download-btn {
            background: linear-gradient(45deg, #17a2b8, #138496);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            margin-top: 15px;
            width: 100%;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .download-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 5px 15px rgba(23, 162, 184, 0.3);
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .status {
            text-align: center;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            display: none;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .checkbox-container {
            display: block;
            position: relative;
            padding-left: 35px;
            margin-bottom: 12px;
            cursor: pointer;
            font-size: 16px;
            user-select: none;
        }
        
        .checkbox-container input {
            position: absolute;
            opacity: 0;
            cursor: pointer;
            height: 0;
            width: 0;
        }
        
        .checkmark {
            position: absolute;
            top: 0;
            left: 0;
            height: 20px;
            width: 20px;
            background-color: #eee;
            border-radius: 4px;
            border: 2px solid #ddd;
        }
        
        .checkbox-container:hover input ~ .checkmark {
            background-color: #ccc;
        }
        
        .checkbox-container input:checked ~ .checkmark {
            background-color: #4CAF50;
            border-color: #4CAF50;
        }
        
        .checkmark:after {
            content: "";
            position: absolute;
            display: none;
        }
        
        .checkbox-container input:checked ~ .checkmark:after {
            display: block;
        }
        
        .checkbox-container .checkmark:after {
            left: 6px;
            top: 2px;
            width: 5px;
            height: 10px;
            border: solid white;
            border-width: 0 3px 3px 0;
            transform: rotate(45deg);
        }
        
        .control-group small {
            color: #666;
            font-size: 0.85em;
            margin-top: 5px;
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 CodeFormer</h1>
            <p>تحسين وترميم الصور بالذكاء الاصطناعي</p>
        </div>
        
        <div class="main-content">
            <div class="upload-section" id="uploadSection">
                <div class="upload-icon">📸</div>
                <h3>اسحب الصورة هنا أو اضغط للاختيار</h3>
                <p>يدعم: JPG, PNG, WEBP</p>
                <input type="file" id="fileInput" class="file-input" accept="image/*">
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">اختر صورة</button>
            </div>
            
            <div class="controls" id="controls" style="display: none;">
                <div class="control-group">
                    <label for="fidelity">جودة الاستعادة: <span id="fidelityValue">0.5</span></label>
                    <input type="range" id="fidelity" class="slider" min="0" max="1" step="0.1" value="0.5">
                </div>
                
                <div class="control-group">
                    <label class="checkbox-container">
                        <input type="checkbox" id="enhanceBackground">
                        <span class="checkmark"></span>
                        تحسين الخلفية والصورة الكاملة
                    </label>
                    <small>تحسين جودة الصورة بالكامل بالإضافة إلى الوجه</small>
                </div>
                
                <div class="control-group">
                    <label for="mode">نوع المعالجة:</label>
                    <select id="mode" class="select">
                        <option value="restoration">ترميم الوجوه</option>
                        <option value="colorization">تلوين الصور</option>
                        <option value="enhancement">تحسين الجودة</option>
                    </select>
                </div>
            </div>
            
            <button class="process-btn" id="processBtn" onclick="processImage()" disabled>معالجة الصورة</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>جاري معالجة الصورة...</p>
            </div>
            
            <div class="status" id="status"></div>
            
            <div class="results" id="results"></div>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        // File input handling
        document.getElementById('fileInput').addEventListener('change', handleFileSelect);
        
        // Drag and drop
        const uploadSection = document.getElementById('uploadSection');
        uploadSection.addEventListener('dragover', handleDragOver);
        uploadSection.addEventListener('dragleave', handleDragLeave);
        uploadSection.addEventListener('drop', handleDrop);
        
        // Fidelity slider
        document.getElementById('fidelity').addEventListener('input', function() {
            document.getElementById('fidelityValue').textContent = this.value;
        });
        
        function handleDragOver(e) {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        }
        
        function handleDragLeave(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
        }
        
        function handleDrop(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        }
        
        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (file) {
                handleFile(file);
            }
        }
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showStatus('يرجى اختيار ملف صورة صالح', 'error');
                return;
            }
            
            selectedFile = file;
            document.getElementById('controls').style.display = 'grid';
            document.getElementById('processBtn').disabled = false;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                showPreview(e.target.result);
            };
            reader.readAsDataURL(file);
            
            showStatus('تم اختيار الصورة بنجاح!', 'success');
        }
        
        function showPreview(imageSrc) {
            const results = document.getElementById('results');
            results.innerHTML = `
                <div class="image-container">
                    <h3>الصورة الأصلية</h3>
                    <img src="${imageSrc}" alt="Original Image">
                </div>
            `;
        }
        
        function processImage() {
            if (!selectedFile) {
                showStatus('يرجى اختيار صورة أولاً', 'error');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('processBtn').disabled = true;
            
            const reader = new FileReader();
                reader.onload = function(e) {
                    const imageData = e.target.result;
                    const fidelity = document.getElementById('fidelity').value;
                    const enhanceBackground = document.getElementById('enhanceBackground').checked;
                    
                    // Send to backend for real processing
                    fetch('/api/process', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            image: imageData,
                            fidelity: fidelity,
                            enhance_background: enhanceBackground
                        })
                    })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('processBtn').disabled = false;
                    
                    if (data.success) {
                        showResults(imageData, data.processed_image);
                        showStatus(data.message, 'success');
                    } else {
                        showStatus('خطأ في معالجة الصورة: ' + data.error, 'error');
                    }
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('processBtn').disabled = false;
                    showStatus('خطأ في الاتصال: ' + error.message, 'error');
                });
            };
            reader.readAsDataURL(selectedFile);
        }
        
        function simulateProcessing() {
            const reader = new FileReader();
            reader.onload = function(e) {
                const originalSrc = e.target.result;
                
                // Create a simple "enhanced" version (just for demo)
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                const img = new Image();
                
                img.onload = function() {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    
                    // Apply some basic filters to simulate enhancement
                    ctx.filter = 'contrast(1.2) brightness(1.1) saturate(1.1)';
                    ctx.drawImage(img, 0, 0);
                    
                    const enhancedSrc = canvas.toDataURL('image/png');
                    
                    showResults(originalSrc, enhancedSrc);
                };
                
                img.src = originalSrc;
            };
            reader.readAsDataURL(selectedFile);
        }
        
        function showResults(originalSrc, enhancedSrc) {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('processBtn').disabled = false;
            
            const results = document.getElementById('results');
            results.innerHTML = `
                <div class="image-container">
                    <h3>الصورة الأصلية</h3>
                    <img src="${originalSrc}" alt="Original Image">
                </div>
                <div class="image-container">
                    <h3>الصورة المحسنة</h3>
                    <img src="${enhancedSrc}" alt="Enhanced Image">
                    <button class="download-btn" onclick="downloadImage('${enhancedSrc}', 'enhanced_image.png')">تحميل الصورة</button>
                </div>
            `;
            
            showStatus('تم تحسين الصورة بنجاح!', 'success');
        }
        
        function downloadImage(src, filename) {
            const link = document.createElement('a');
            link.href = src;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status ${type}`;
            status.style.display = 'block';
            
            setTimeout(() => {
                status.style.display = 'none';
            }, 5000);
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/process', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        fidelity_weight = float(data.get('fidelity', 0.5))
        enhance_background = data.get('enhance_background', False)
        
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        # Process image with CodeFormer
        processed_image, message = process_image_with_codeformer(image_data, fidelity_weight, enhance_background)
        
        if processed_image:
            return jsonify({
                'success': True,
                'message': 'تم تحسين الصورة بنجاح!' + (' مع تحسين الخلفية' if enhance_background else ''),
                'processed_image': processed_image
            })
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 بدء تشغيل خادم CodeFormer...")
    port = int(os.environ.get('PORT', 5000))
    print(f"📱 الموقع متاح على: http://localhost:{port}")
    print("⚡ اضغط Ctrl+C لإيقاف الخادم")
    
    # Initialize models
    print("🔄 تحميل النماذج...")
    if initialize_models():
        print("✅ تم تحميل النماذج بنجاح!")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("❌ فشل في تحميل النماذج. تحقق من وجود ملفات النماذج في مجلد weights.")
        print("🔄 تشغيل الخادم في الوضع التجريبي...")
        app.run(host='0.0.0.0', port=port, debug=False)