from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# טען את המודל ששמרת קודם
model = load_model('cat_dog_model.h5')

# קובץ התמונה (שנה את הנתיב לפי הצורך)
image_path = r'C:\Users\444\Downloads\bite.jpg'

# 1. טען את התמונה בגודל תואם
img = load_img(image_path, target_size=(64, 64))  # resize to match training

# 2. המר את התמונה למערך של מספרים
img_array = img_to_array(img)

# 3. הוסף מימד של "batch" (כמו: [תמונה אחת])
img_array = np.expand_dims(img_array, axis=0)

# 4. נרמל (כמו שעשית באימון – מ-0-255 ל-0-1)
img_array = img_array / 255.0

# 5. הרץ תחזית
prediction = model.predict(img_array)

# 6. פענוח התוצאה
confidence = prediction[0][0]
if confidence > 0.5:
    print(f"🐶 זו כנראה תמונה של **כלב** (ביטחון: {confidence:.2%})")
else:
    print(f"😺 זו כנראה תמונה של **חתול** (ביטחון: {(1 - confidence):.2%})")
                                                                                                                                             r'C:\Users\444\Downloads\BITE.jpg']
