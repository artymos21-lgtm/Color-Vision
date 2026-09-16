from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2, numpy as np, pytesseract
from PIL import Image
from io import BytesIO

app = FastAPI(title='ColorVision')
app.mount('/static', StaticFiles(directory='static'), name='static')

@app.get('/', response_class=HTMLResponse)
def home():
    return open('templates/index.html', encoding='utf-8').read()

def variants(img):
    out=[]
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out += [gray, cv2.equalizeHist(gray)]
    lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    out += [clahe.apply(l), clahe.apply(a), clahe.apply(b)]
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ch in cv2.split(hsv): out.append(ch)
    # color-channel differences can reveal the foreground/background contrast
    B,G,R=cv2.split(img)
    for x,y in [(R,G),(R,B),(G,B)]:
        out.append(cv2.normalize(cv2.absdiff(x,y),None,0,255,cv2.NORM_MINMAX))
    return out

def recognize(img):
    h,w=img.shape[:2]
    scale=max(1.0, min(3.0, 1400/max(h,w)))
    if scale != 1: img=cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    candidates=[]
    for v in variants(img):
        for mode in range(4):
            if mode==0: p=v
            elif mode==1: p=cv2.threshold(v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            elif mode==2: p=cv2.adaptiveThreshold(v,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
            else: p=cv2.threshold(v,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            p=cv2.medianBlur(p,3)
            for psm in (6,8,10,11,13):
                txt=pytesseract.image_to_string(p,config=f'--psm {psm} -c tessedit_char_whitelist=0123456789').strip()
                digits=''.join(c for c in txt if c.isdigit())
                if digits: candidates.append(digits)
    if not candidates: return None
    # Prefer the most frequent short digit string; this is a helper, not a medical diagnosis.
    from collections import Counter
    c=Counter(candidates)
    return c.most_common(1)[0][0]

@app.post('/api/recognize')
async def api_recognize(file: UploadFile = File(...)):
    data=await file.read()
    img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if img is None: return JSONResponse({'ok':False,'error':'Не удалось прочитать изображение'},status_code=400)
    result=recognize(img)
    return {'ok':True,'digit':result,'message': ('Цифра распознана' if result else 'Цифра не распознана')}
