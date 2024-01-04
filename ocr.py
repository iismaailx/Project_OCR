import cv2
import os
import re
import sys
import pytesseract
import textdistance
import numpy as np
import pandas as pd
from datetime import date
from operator import itemgetter

ROOT_PATH = os.getcwd()
# IMAGE_PATH = os.path.join(ROOT_PATH, 'Kywa.jpg')

#LINE_REC_PATH = os.path.join(ROOT_PATH, 'data/ID_CARD_KEYWORDS.csv')
LINE_REC_PATH = r'/home/rnd/Development/OCR_RASPI/data/ID_CARD_KEYWORDS.csv'
CITIES_REC_PATH = r'/home/rnd/Development/OCR_RASPI/data/KOTA.csv'
RELIGION_REC_PATH = r'/home/rnd/Development/OCR_RASPI/data/RELIGIONS.csv'
MARRIAGE_REC_PATH = r'/home/rnd/Development/OCR_RASPI/data/MARRIAGE_STATUS.csv'
JENIS_KELAMIN_REC_PATH = r'/home/rnd/Development/OCR_RASPI/data/JENIS_KELAMIN.csv'
PROVINCE_REC_PATH = r'/home/rnd/Development/OCR_RASPI/data/PROVINSI.csv'
DISTRIC_REC_PATH = r'/home/rnd/Development/OCR_RASPI/data/KECAMATAN.csv'
PEKERJAAN_REC_PATH = r'/home/rnd/Development/OCR_RASPI/data/PEKERJAAN.csv'
DESA_KEL_REC_PATH = r'/home/rnd/Development/OCR_RASPI/data/VILLAGES.csv'
NEED_COLON = [3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21]
NEXT_LINE = 9
ID_NUMBER = 3

def convertScale(img, alpha, beta):
    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    # auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

# ------------------------------------------------------------------------------

def ocr_raw(path):
    try:
        image = cv2.imread(path)
        image = np.array(image)
        image = cv2.resize(image, (50 * 16, 500))
        image = automatic_brightness_and_contrast(image)
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # img_gray = cv2.equalizeHist(img_gray)
        # img_gray = cv2.fastNlMeansDenoising(img_gray, None, 3, 7, 21)
        # cv2.imwrite("images/ocr_raw1.jpg", image)
        # cv2.imwrite("images/ocr_raw2.jpg", img_gray)
        id_number = return_id_number(image, img_gray)
        # print(id_number)
        cv2.fillPoly(img_gray, pts=[np.asarray([(540, 150), (540, 499), (798, 499), (798, 150)])], color=(255, 255, 255))
        th, threshed = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
        result_raw = pytesseract.image_to_string(threshed, lang="ind")
        return result_raw, id_number
    except Exception as e:
        print(f"Error on raw process : {e}")

def strip_op(result_raw):
    result_list = result_raw.split('\n')
    new_result_list = []

    for tmp_result in result_list:
        if tmp_result.strip(' '):
            new_result_list.append(tmp_result)
    # print("strip")
    return new_result_list


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes

def return_id_number(image, img_gray):
    img_mod = cv2.imread("/home/rnd/Development/OCR_RASPI/data/module1.png")
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, rectKernel)

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)

    threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = threshCnts
    cur_img = image.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
    copy = image.copy()

    locs = []
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)

        # ar = w / float(h)
        # if ar > 3:
        # if (w > 40 ) and (h > 10 and h < 20):
        if h > 10 and w > 100 and x < 300:
            img = cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            locs.append((x, y, w, h, w * h))

    locs = sorted(locs, key=itemgetter(1), reverse=False)
    # print(f"ini locs {locs}")
    # print(f"ini panjang locs {len(locs)}")
    # print(image.shape)
    # nik = image[locs[1][1] - 15:locs[1][1] + locs[1][3] + 15, locs[1][0] - 15:locs[1][0] + locs[1][2] + 15]
    # text = image[locs[2][1] - 10:locs[2][1] + locs[2][3] + 10, locs[2][0] - 10:locs[2][0] + locs[2][2] + 10]

    try:
        if len(locs)>=3:
            nik = image[locs[1][1] - 15:locs[1][1] + locs[1][3] + 15, locs[1][0] - 15:locs[1][0] + locs[1][2] + 15]
            # print(nik)
            
            # print(img_mod.shape)
            ref = cv2.cvtColor(img_mod, cv2.COLOR_BGR2GRAY)
            # print(type(ref))
            ref = cv2.threshold(ref, 66, 255, cv2.THRESH_BINARY_INV)[1]

            refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            refCnts = sort_contours(refCnts, method="left-to-right")[0]

            digits = {}
            for (i, c) in enumerate(refCnts):
                (x, y, w, h) = cv2.boundingRect(c)
                roi = ref[y:y + h, x:x + w]
                roi = cv2.resize(roi, (57, 88))
                digits[i] = roi

            gray_nik = cv2.cvtColor(nik, cv2.COLOR_BGR2GRAY)
            # print(gray_nik)
            group = cv2.threshold(gray_nik, 127, 255, cv2.THRESH_BINARY_INV)[1]

            digitCnts, hierarchy_nik = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            nik_r = nik.copy()
            cv2.drawContours(nik_r, digitCnts, -1, (0, 0, 255), 3)

            gX = locs[1][0]
            gY = locs[1][1]
            gW = locs[1][2]
            gH = locs[1][3]

            ctx = sort_contours(digitCnts, method="left-to-right")[0]

            locs_x = []
            for (i, c) in enumerate(ctx):
                (x, y, w, h) = cv2.boundingRect(c)
                if h > 10 and w > 10:
                    img = cv2.rectangle(nik_r, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    locs_x.append((x, y, w, h))
            output = []
            groupOutput = []
            for c in locs_x:
                (x, y, w, h) = c
                roi = group[y:y + h, x:x + w]
                roi = cv2.resize(roi, (57, 88))

                scores = []
                for (digit, digitROI) in digits.items():
                    result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                    (_, score, _, _) = cv2.minMaxLoc(result)
                    scores.append(score)

                groupOutput.append(str(np.argmax(scores)))

            cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
            cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            output.extend(groupOutput)
            # print("return id number")
            return ''.join(output)
        else:
            return ""
    except Exception as e:
        print(f"Error: {e}. Masalah terjadi saat mengakses elemen list.")
        
def parsing(data):
    json_data = {
        "provinsi":"-",
        "kotakab":"-",
        "nik":"-",
        "nama":"-",
        "ttl":"-",
        "jeniskelamin":"-",
        "goldarah":"-",
        "alamat":"-",
        "keldesa":"-",
        "kecamatan":"-",
        "agama":"-",
        "rtrw":"-",
        "statusperkawinan": "-",
        "pekerjaan": "-",
        "kewarganegaraan": "-",
        "berlakuhingga": "-",
        "readstatus":"Failed to read information from the ID card image."
        }
    
    agama_keywords=["ISLAM", "KRISTEN", "KATHOLIK", "HINDU", "BUDDHA", "KONGHUCU"]
    status_perkawinan_keywords = ["BELUM", "CERAI", "KAWIN", "HIDUP", "MATI"]
    golongan_darah_keywords=["A", "B", "C", "D"]
    jenis_kelamin_keywords=["LAKI-LAKI", "PEREMPUAN"]
    warga_negara_keywords=["WNA", "WNI"]
    try:
        # print(data)
        for item in data:
            # print(item)
            # print(item[0])
            if item != None and len(item) >= 2:
                key = item[0]
                value = ' '.join(item[2:]).replace(':', '').strip()
                # value = None
                if key == "PROVINSI":
                    json_data["provinsi"] = ' '.join(item[1:]).replace(':', '').strip()
                    
                if key == "KOTA" or key == "KABUPATEN":
                    json_data["kotakab"] = ' '.join(item[1:]).replace(':', '').strip()
                                
                if key == "Nama":
                    json_data["nama"] = ''.join([char for char in value if not char.isdigit()])
                
                if key == "NIK":
                    json_data["nik"]=value
                    
                if key == "Tempat/Tgl" or key == "Lahir":
                    json_data['ttl']=value
                
                if key == "Alamat":
                    json_data["alamat"]=value
                    
                if key == "RT/RW":
                    json_data["rtrw"]=value
                    
                if key == "Kel/Desa":
                    json_data['keldesa']=value
                    
                if key == "Kecamatan":
                    json_data["kecamatan"]=value
                
                if key == 'Pekerjaan':
                    json_data['pekerjaan']=forbidenchar(value)   
                
                if key == 'Berlaku' or key == 'Hingga':
                    json_data['berlakuhingga'] = value
                    
                if key == "Jenis" or key == "Kelamin":
                    forbidenchar(value)
                    for jenis in jenis_kelamin_keywords:
                        if jenis in value:
                            json_data["jeniskelamin"]=jenis
                                                        
                if key == "Gol." or key == "Darah":
                    forbidenchar(value)
                    for gol in golongan_darah_keywords:
                        if gol in value:
                            json_data["goldarah"]=gol 
                
                if key == "Agama":
                    forbidenchar(value)
                    for agama in agama_keywords:
                        if agama in value:
                            json_data["agama"]=agama
                    
                if key == "Status" or key == "Perkawinan":
                    forbidenchar(value)
                    word1=""
                    for i in status_perkawinan_keywords[:2]:
                        if i in value:
                            word1 = i
                    word2=""
                    for i in status_perkawinan_keywords[2:]:
                        if i in value:
                            word2 = i
                    json_data["statusperkawinan"]=word1+" "+word2
                        
                if key == "Kewarganegaraan":
                    forbidenchar(value)
                    for warga in warga_negara_keywords:
                        if warga in value:
                            json_data["kewarganegaraan"]=warga
                            
                json_data["readstatus"]="Successfully read information from the ID card image."
    except Exception as e:
        print(f"Error: {e}. Masalah terjadi saat mengakses elemen list parsing.") 
    # print("Errornya parsing")                            
    return json_data        
    
def forbidenchar(data):
    return re.sub(r'[^a-zA-Z/]', '', data)

def main(image):
    raw_df = pd.read_csv(LINE_REC_PATH, header=None)
    cities_df = pd.read_csv(CITIES_REC_PATH, header=None)
    religion_df = pd.read_csv(RELIGION_REC_PATH, header=None)
    marriage_df = pd.read_csv(MARRIAGE_REC_PATH, header=None)
    jenis_kelamin_df = pd.read_csv(JENIS_KELAMIN_REC_PATH, header=None)
    province_df = pd.read_csv(PROVINCE_REC_PATH, header=None)
    distric_df = pd.read_csv(DISTRIC_REC_PATH, header=None)
    pekerjaan_df = pd.read_csv(PEKERJAAN_REC_PATH, header=None)
    desa_kel_df = pd.read_csv(DESA_KEL_REC_PATH, header=None)
    result_raw, id_number = ocr_raw(image)
    # print(result_raw)
    # print(id_number)
    result_list = strip_op(result_raw)
    # print(result_list)
    # print(f"this result_list: {result_list}")
    # print(f"this result_raw: {result_raw}")
    # print(f"this id_number: {id_number}")
    data = []
    # print("NIK: " + str(id_number))

    loc2index = dict()
    for i, tmp_line in enumerate(result_list):
        for j, tmp_word in enumerate(tmp_line.split(' ')):
            tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word_, tmp_word.strip(':')) for tmp_word_ in raw_df[0].values]

            tmp_sim_np = np.asarray(tmp_sim_list)
            arg_max = np.argmax(tmp_sim_np)

            if tmp_sim_np[arg_max] >= 0.6:
                loc2index[(i, j)] = arg_max

    last_result_list = []
    useful_info = False
    
    for i, tmp_line in enumerate(result_list):
        tmp_list = []
        for j, tmp_word in enumerate(tmp_line.split(' ')):
            tmp_word = tmp_word.strip(':')

            if(i, j) in loc2index:
                useful_info = True
                if loc2index[(i, j)] == NEXT_LINE:
                    last_result_list.append(tmp_list)
                    tmp_list = []
                tmp_list.append(raw_df[0].values[loc2index[(i, j)]])
                if loc2index[(i, j)] in NEED_COLON:
                    tmp_list.append(':')
            elif tmp_word == ':' or tmp_word =='':
                continue
            else:
                tmp_list.append(tmp_word)

        if useful_info:
            if len(last_result_list) > 2 and ':' not in tmp_list:
                last_result_list[-1].extend(tmp_list)
            else:
                last_result_list.append(tmp_list)
    # print(last_result_list)
    
    for tmp_data in last_result_list:
        if '—' in tmp_data:
            tmp_data.remove('—')

        if 'PROVINSI' in tmp_data:
            if len(tmp_data) >= 1:
                for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                    if tmp_word:
                        tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in province_df[0].values]

                        tmp_sim_np = np.asarray(tmp_sim_list)
                        arg_max = np.argmax(tmp_sim_np)
                        if tmp_sim_np[arg_max] >= 0.6:
                            tmp_data[tmp_index + 1] = province_df[0].values[arg_max]
                    # print("prov done")

        if 'KABUPATEN' in tmp_data or 'KOTA' in tmp_data:
            if len(tmp_data) >= 1:
                for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                    if tmp_word:
                        tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in cities_df[0].values]

                        tmp_sim_np = np.asarray(tmp_sim_list)
                        arg_max = np.argmax(tmp_sim_np)
                        if tmp_sim_np[arg_max] >= 0.6:
                            tmp_data[tmp_index + 1] = cities_df[0].values[arg_max]
                # print("kab done")  
                #print(test)
        if 'Nama' in tmp_data:
            if len(tmp_data) >= 2:
                nama = ' '.join(tmp_data[2:])
                nama = re.sub('[^A-Z. ]', '', nama)
                if len(nama.split()) == 1:
                    nama = re.sub('[^A-Z.]', '', nama)
            # print("nama  done")

        if 'NIK' in tmp_data:
            if len(id_number) >= 16 and len(tmp_data) >= 2:
                # id_number = tmp_data[2]
                if "D" in id_number:
                    id_number = id_number.replace("D", "0")
                if "?" in id_number:
                    id_number = id_number.replace("?", "7")
                if "L" in id_number:
                    id_number = id_number.replace("L", "1")
                if "I" in id_number:
                    id_number = id_number.replace("I", "1")
                if "R" in id_number: 
                    id_number = id_number.replace("R", "2")
                if "O" in id_number:
                    id_number = id_number.replace("O", "0")   
                if "o" in id_number:
                    id_number = id_number.replace("o", "0")
                if "S" in id_number:
                    id_number = id_number.replace("S", "5")
                if "G" in id_number:
                    id_number = id_number.replace("G", "6")

                while len(tmp_data) > 2:
                    tmp_data.pop()
                tmp_data.append(id_number)
            else:
                while len(tmp_data) > 3:
                    tmp_data.pop()
                if len(tmp_data) < 3:
                    tmp_data.append(id_number)
                if len(tmp_data) >= 2:
                    tmp_data[2] = id_number
            # print("nik done")

        if 'Agama' in tmp_data:
            if len(tmp_data) >= 1:
                for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                    if tmp_word:
                        tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in religion_df[0].values]

                        tmp_sim_np = np.asarray(tmp_sim_list)
                        arg_max = np.argmax(tmp_sim_np)
        
                        if tmp_sim_np[arg_max] >= 0.6:
                           tmp_data[tmp_index + 1] = religion_df[0].values[arg_max]
                # print("agama done")
                                       
        if 'Status' in tmp_data or 'Perkawinan' in tmp_data:
            if len(tmp_data) >= 2:
                for tmp_index, tmp_word in enumerate(tmp_data[2:]): #tadinya index 2
                    if tmp_word:
                        tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in marriage_df[0].values]

                        tmp_sim_np = np.asarray(tmp_sim_list)
                        arg_max = np.argmax(tmp_sim_np)
        
                        if tmp_sim_np[arg_max] >= 0.6:
                            tmp_data[tmp_index + 2] = marriage_df[0].values[arg_max]                 
                # print("stat done")
                
        if 'Alamat' in tmp_data:
            for tmp_index in range(len(tmp_data)):
                if tmp_index:
                    if "!" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("!", "I")
                    if "1" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("1", "I")
                    if "i" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("i", "I")
                    if "RI" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("RI", 'RT')
                    if "Rw" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("Rw", 'RW')
                    if "rw" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("rw", 'RW')
                    if "rt" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("rt", 'RT')
                # print("alamt done")
                
        if 'RT/RW' in tmp_data or 'RT' in tmp_data or 'RW' in tmp_data:
            if len(tmp_data) >= 3:
                tmp_data = [item for item in tmp_data if item != ""]
                for index, elemen in enumerate(tmp_data):
                    if '“' in elemen:
                        tmp_data[index] = tmp_data[index].replace('“', '')
                    if '"' in elemen:
                        tmp_data[index] = tmp_data[index].replace('"', '')
                    if 'f' in elemen:
                        tmp_data[index] = tmp_data[index].replace('f', '/')
                    if elemen.startswith('/') or elemen.endswith('/') and elemen[1:].isdigit():
                        tmp_data[index] = elemen.replace('/', '')
                    if re.match(r'^(\d{3})(\d{3})$', elemen):
                        tmp_data[index] = re.sub(r'^(\d{3})(\d{3})$', r'\1/\2', elemen)
        
                    tmp_data = [re.sub(r'^\d{4,}', lambda x: x.group()[1:], item) for item in tmp_data]
                    if '/' not in tmp_data and len(tmp_data) <= 3:
                        clean_ = [item for item in tmp_data if re.match(r'\d{3}', item)]
                        clean_ = [i.split('/') for i in clean_]
                        clean_[0] = '/'.join(clean_[0])
                        index = tmp_data.index(':')
                        tmp_data = tmp_data[:index+ 1]
                        tmp_data = tmp_data + clean_

                    if '/' not in tmp_data and len(tmp_data) > 3:
                        tmp_data.insert(3, '/')
                        tmp_data = [x for x in tmp_data if x != '']
            # print("rtrw done")
            
        # if 'Kel/Desa' in tmp_data or 'Kel' in tmp_data or 'Desa' in tmp_data:
        #     if len(tmp_data) >= 1:
        #         for tmp_index, tmp_word in enumerate(tmp_data[1:]):
        #             if tmp_word:
        #                 tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in desa_kel_df[0].values]

        #                 tmp_sim_np = np.asarray(tmp_sim_list)
        #                 arg_max = np.argmax(tmp_sim_np)
        #                 if tmp_sim_np[arg_max] >= 0.6:
        #                     tmp_data[tmp_index + 1] = desa_kel_df[0].values[arg_max]
        #         # print("kel des done")  
                
        if 'Jenis' in tmp_data or 'Kelamin' in tmp_data:
            if len(tmp_data) >= 2:
                for tmp_index, tmp_word in enumerate(tmp_data[2:]):
                    tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in jenis_kelamin_df[0].values]

                    tmp_sim_np = np.asarray(tmp_sim_list)
                    arg_max = np.argmax(tmp_sim_np)

                    if tmp_sim_np[arg_max] >= 0.6:
                        tmp_data[tmp_index + 2] = jenis_kelamin_df[0].values[arg_max]
            # print("jenis done")
        
        if 'Gol.' in tmp_data or ' Darah' in tmp_data or 'Darah' in tmp_data:
            if len(tmp_data) >= 4:
                if tmp_data[3] == '0':
                        tmp_data[3] = tmp_data[3].replace('0', 'O')
                if tmp_data[3] == '8':
                        tmp_data[3] = tmp_data[3].replace('8', 'B')
            # print("gol done")

        if 'Tempat' in tmp_data or 'Tgl' in tmp_data or 'Lahir' in tmp_data:
            join_tmp = ' '.join(tmp_data)

            match_tgl1 = re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", join_tmp)
            match_tgl2 = re.search("([0-9]{2}\ [0-9]{2}\ [0-9]{4})", join_tmp)
            match_tgl3 = re.search("([0-9]{2}\-[0-9]{2}\ [0-9]{4})", join_tmp)
            match_tgl4 = re.search("([0-9]{2}\ [0-9]{2}\-[0-9]{4})", join_tmp)

            if match_tgl1:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl1.group(), '%d-%m-%Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            elif match_tgl2:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl2.group(), '%d %m %Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            elif match_tgl3:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl3.group(), '%d-%m %Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            elif match_tgl4:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl4.group(), '%d %m-%Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            else:
                tgl_lahir = ""
            
            if len(tmp_data) >= 2:
                for tmp_index, tmp_word in enumerate(tmp_data[2:]):
                    tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in cities_df[0].values]

                    tmp_sim_np = np.asarray(tmp_sim_list)
                    arg_max = np.argmax(tmp_sim_np)
    
                    if tmp_sim_np[arg_max] >= 0.6:
                        tmp_data[tmp_index + 2] = cities_df[0].values[arg_max]
                        tempat_lahir = tmp_data[tmp_index + 2]
            # print(" ttl done")
                    
        if 'Kecamatan' in tmp_data:
            if len(tmp_data) >= 1:
                for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                    tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in distric_df[0].values]

                    tmp_sim_np = np.asarray(tmp_sim_list)
                    arg_max = np.argmax(tmp_sim_np)
                    
                    if tmp_sim_np[arg_max] >= 0.6:
                        tmp_data[tmp_index + 1] = distric_df[0].values[arg_max] 
                  
        if 'Pekerjaan' in tmp_data:
            if len(tmp_data) >= 1:
                for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                    tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in pekerjaan_df[0].values]

                    tmp_sim_np = np.asarray(tmp_sim_list)
                    arg_max = np.argmax(tmp_sim_np)
                    
                    if tmp_sim_np[arg_max] >= 0.6:
                        tmp_data[tmp_index + 1] = pekerjaan_df[0].values[arg_max]
                                
        data.append(tmp_data)
    clean_data = parsing(data)
    # print("-"*30)
    # print(clean_data)
    return clean_data

if __name__ == '__main__':
    try:
        main(sys.argv[1])
        # main("images/ktp2.jpg")
    except Exception as e:
        print("start error")
        print(f"this error :{e}")

