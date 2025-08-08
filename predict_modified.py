import hydra
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import easyocr
import cv2
import mysql.connector
import re
import datetime

#  OCR reader
reader = easyocr.Reader(['en'], gpu=True)

#  Regex to validate Indian plates
def is_valid_plate(text):
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$'
    cleaned = re.sub(r'\W+', '', text.upper())  # Remove non-alphanumeric
    return re.match(pattern, cleaned) is not None

#  Formatter for plate like DL7CD5017 → DL 7C D 5017
def format_plate(plate):
    return re.sub(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{3,4})$', r'\1 \2 \3 \4', plate)

#  Insert into MySQL with duplicate prevention
def insert_plate_to_db(plate_text):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="num_plates"
        )
        cursor = conn.cursor()

        # Check if the same plate was recently added (last 10 seconds)
        cursor.execute("""
            SELECT timestamp FROM plates
            WHERE number_plate = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (plate_text,))
        result = cursor.fetchone()

        now = datetime.datetime.now()

        if result:
            last_time = result[0]
            if (now - last_time).total_seconds() < 10:
                cursor.close()
                conn.close()
                return

        # Insert
        cursor.execute("INSERT INTO plates (number_plate) VALUES (%s)", (plate_text,))
        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"[DB ERROR] {e}")

#  Perform OCR + Clean + Validate + Save
def perform_ocr_on_image(img, coordinates):
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray_img)

    for res in results:
        candidate = res[1].strip()
        cleaned = re.sub(r'\W+', '', candidate.upper())
        if len(candidate) > 6 and res[2] > 0.2 and is_valid_plate(cleaned):
            formatted = format_plate(cleaned)
            try:
                insert_plate_to_db(formatted)
            except:
                pass
            return formatted  # Only return first valid one
    return ""  # If no valid plate found

class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:
                c = int(cls)
                text_ocr = perform_ocr_on_image(im0, xyxy)
                label = text_ocr if text_ocr else self.model.names[c]
                self.annotator.box_label(xyxy, label, color=colors(c, True))

            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)

    # ✅ Accept webcam input (if source=0 or '0')
    if str(cfg.source) == "0":
        cfg.source = 0  # OpenCV needs integer not string

    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"

    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()