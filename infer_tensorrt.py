# infer_tensorrt.py
import cv2, numpy as np, time
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

PLAN_PATH = "deepcrack_256_fp16.plan"
IMG_H, IMG_W = 256, 256
THR = 0.5  # change if you tuned a better threshold

def preprocess_bgr(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 256.0   # match training!
    x = np.transpose(x, (2,0,1))[None, ...].copy()  # 1x3xH xW
    return x

def postprocess(mask, orig_shape=None):
    m = (mask[0,0] >= THR).astype(np.uint8) * 255
    if orig_shape is not None:
        m = cv2.resize(m, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    return m

class TRTSession:
    def __init__(self, plan_path):
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        with open(plan_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.in_idx  = self.engine.get_binding_index("input")
        self.out_idx = self.engine.get_binding_index("mask")

        shape = self.engine.get_binding_shape(self.in_idx)
        if -1 in shape:  # dynamic
            self.ctx.set_binding_shape(self.in_idx, (1,3,IMG_H,IMG_W))

        self.in_nbytes  = int(np.prod(self.ctx.get_binding_shape(self.in_idx)) * np.float32().nbytes)
        self.out_nbytes = int(np.prod(self.ctx.get_binding_shape(self.out_idx)) * np.float32().nbytes)

        self.d_in  = cuda.mem_alloc(self.in_nbytes)
        self.d_out = cuda.mem_alloc(self.out_nbytes)
        self.bindings = [int(self.d_in), int(self.d_out)]

    def infer(self, x):
        cuda.memcpy_htod_async(self.d_in, x, self.stream)
        self.ctx.execute_async_v2(self.bindings, self.stream.handle)
        out = np.empty((1,1,IMG_H,IMG_W), dtype=np.float32)
        cuda.memcpy_dtoh_async(out, self.d_out, self.stream)
        self.stream.synchronize()
        return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str)
    ap.add_argument("--video", type=str, help="'cam' or path to video")
    ap.add_argument("--save",  type=str, default="")
    ap.add_argument("--plan",  type=str, default=PLAN_PATH)
    args = ap.parse_args()

    sess = TRTSession(args.plan)

    if args.image:
        img = cv2.imread(args.image, cv2.IMREAD_COLOR)
        x = preprocess_bgr(img)
        t0 = time.time()
        out = sess.infer(x)
        dt = (time.time()-t0)*1000
        print(f"Inference: {dt:.2f} ms")
        mask = postprocess(out, img.shape[:2])
        ov = img.copy(); ov[mask>0] = (0,0,255)
        if args.save: cv2.imwrite(args.save, ov)
        cv2.imshow("overlay", ov); cv2.waitKey(0)

    elif args.video:
        cap = cv2.VideoCapture(0) if args.video == "cam" else cv2.VideoCapture(args.video)
        assert cap.isOpened()
        while True:
            ok, frame = cap.read()
            if not ok: break
            x = preprocess_bgr(frame)
            t0 = time.time(); out = sess.infer(x); dt = (time.time()-t0)*1000
            mask = postprocess(out, frame.shape[:2])
            ov = frame.copy(); ov[mask>0] = (0,0,255)
            cv2.putText(ov, f"{dt:.1f} ms", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Crack (TRT)", ov)
            if cv2.waitKey(1) & 0xFF == 27: break
        cap.release()
    else:
        print("Use --image or --video")
