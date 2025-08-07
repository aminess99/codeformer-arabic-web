# نشر CodeFormer على Railway

## خطوات النشر

### 1. إعداد حساب Railway
- قم بإنشاء حساب على [Railway.app](https://railway.app)
- اربط حسابك مع GitHub

### 2. رفع المشروع
- ارفع المشروع إلى مستودع GitHub
- تأكد من وجود الملفات التالية:
  - `Procfile`
  - `railway.toml`
  - `requirements_railway.txt`
  - `.railwayignore`

### 3. إنشاء مشروع جديد على Railway
- اذهب إلى Railway Dashboard
- اضغط "New Project"
- اختر "Deploy from GitHub repo"
- اختر مستودع CodeFormer

### 4. تكوين المتغيرات
Railway سيقوم بتكوين المتغيرات تلقائياً من ملف `railway.toml`

### 5. النشر
- Railway سيبدأ عملية البناء والنشر تلقائياً
- انتظر حتى اكتمال النشر (قد يستغرق 10-15 دقيقة)
- ستحصل على رابط للوصول إلى التطبيق

## ملاحظات مهمة

### حجم النماذج
- النماذج الكبيرة (ملفات .pth) سيتم تحميلها تلقائياً عند أول تشغيل
- قد يستغرق التحميل الأول وقتاً أطول

### الذاكرة والمعالجة
- تأكد من اختيار خطة Railway مناسبة لحجم النماذج
- النماذج تحتاج ذاكرة كبيرة للتشغيل

### المنافذ
- التطبيق يستخدم متغير البيئة PORT تلقائياً
- Railway يوفر هذا المتغير تلقائياً

## استكشاف الأخطاء

### إذا فشل النشر
1. تحقق من سجلات البناء في Railway Dashboard
2. تأكد من صحة ملف requirements_railway.txt
3. تحقق من وجود جميع الملفات المطلوبة

### إذا فشل تحميل النماذج
- تحقق من سجلات التطبيق
- قد تحتاج لزيادة timeout في إعدادات Railway

## الملفات المضافة للنشر

- `Procfile`: يحدد أمر تشغيل التطبيق
- `railway.toml`: تكوين Railway
- `requirements_railway.txt`: جميع التبعيات المطلوبة
- `.railwayignore`: الملفات المستبعدة من النشر
- `RAILWAY_DEPLOYMENT.md`: هذا الملف

## الدعم
إذا واجهت مشاكل في النشر، تحقق من:
- [Railway Documentation](https://docs.railway.app)
- [Railway Community](https://railway.app/discord)