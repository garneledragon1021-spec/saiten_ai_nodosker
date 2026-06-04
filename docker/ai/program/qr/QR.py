import qrcode                       #QRコード関係
import cv2                          #画像処理関係
import sys                          #外部引数取得
from pyzbar.pyzbar import decode    #デコード

#初期設定
args = sys.argv

if len(args) < 3:
    print("Different")
    sys.exit(1)

file_name = args[1]     #元画像
cutted_name = args[2]   #切り取り画像

#QRコード生成関数
def create_QR():
     #QRコード設定
    qr = qrcode.QRCode(
        version = 1,    #バージョン指定
        error_correction = qrcode.constants.ERROR_CORRECT_L,    #エラー訂正レベル
        box_size = 3,  #1つのブロックサイズ
        border = 3,     #枠線の太さ
    )

    #データ設定
    data = "fujishima startup QRcode"   #データ内容
    qr.add_data(data)   #データ格納
    qr.make(fit=True)   #作成

    #qrコード画像生成
    img = qr.make_image(fill="black", back_color="white")   #QRコードの色合い設定
    img.show()  #QRコード画像の表示


#QRコード読み取り
def read_QR(img_data):
    if img_data is None:
        return "Different"

    decoded_qr = decode(img_data)   #画像ファイルのデコード
    if not decoded_qr:
        return "Different"

    h, w, _ = img_data.shape        #画像の幅(x座標の最大値)(w),高さ(y座標の最大値)(h),チャンネル(c)

    for obj in decoded_qr:
        data = obj.data.decode("utf-8") #QRコード情報
        if data != "fujishima startup QRcode":
            continue

        #切り取り画像座標情報
        xmin = max(obj.rect.left, 0)                 #左上X座標(最小値)
        ymin = max(obj.rect.top + obj.rect.height, 0)  #QR直下を開始位置にする
        xmax = w                                     #右下x座標(最大値)
        ymax = h                                     #右下y座標(最大値)

        cut_img = img_data[ymin:ymax, xmin:xmax]    #画像切り取り
        if cut_img.size == 0:
            return "Different"

        cv2.imwrite(cutted_name, cut_img) #切り取り画像保存
        return data

    return "Different"

#create_QR() #QRコード作成

img_data = cv2.imread(file_name)    #画像読み込み
data = read_QR(img_data)            #読み込み実行

print(data)