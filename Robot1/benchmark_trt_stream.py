# benchmark_trt_stream.py
import cv2, time, csv, os, argparse, numpy as np, psutil
import pycuda.autoinit, pycuda.driver as cuda
from infer_tensorrt import TRTSession, preprocess_bgr, postprocess, PLAN_PATH

def gpu_mem_bytes():
    free_b, total_b = cuda.mem_get_info()
    return (total_b - free_b), total_b

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="cam")
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--out_csv", default="bench_stream.csv")
    ap.add_argument("--plan", default=PLAN_PATH)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(0) if args.video=="cam" else cv2.VideoCapture(args.video)
    assert cap.isOpened(), "Cannot open camera/video"

    sess = TRTSession(args.plan)
    start_ev, end_ev = cuda.Event(), cuda.Event()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame","t_pre_ms","t_inf_ms","t_post_ms","t_total_ms",
                     "gpu_mem_used_B","gpu_mem_total_B","cpu_mem_rss_B"])
        n, t_sum = 0, 0.0
        while n < args.frames:
            ok, frame = cap.read()
            if not ok: break
            t0 = time.perf_counter()

            t_pre0 = time.perf_counter()
            x = preprocess_bgr(frame)
            t_pre = (time.perf_counter()-t_pre0)*1000

            start_ev.record()
            out = sess.infer(x)
            end_ev.record(); end_ev.synchronize()
            t_inf = start_ev.time_till(end_ev)  # ms

            t_post0 = time.perf_counter()
            mask = postprocess(out, frame.shape[:2])
            t_post = (time.perf_counter()-t_post0)*1000

            t_tot = (time.perf_counter()-t0)*1000
            t_sum += t_tot; n += 1

            gpu_used, gpu_total = gpu_mem_bytes()
            rss = psutil.Process(os.getpid()).memory_info().rss

            wr.writerow([n, f"{t_pre:.3f}", f"{t_inf:.3f}", f"{t_post:.3f}", f"{t_tot:.3f}",
                         gpu_used, gpu_total, rss])

            if args.display:
                ov = frame.copy(); ov[mask>0]=(0,0,255)
                cv2.putText(ov, f"{t_tot:.1f} ms (inf {t_inf:.1f})",
                            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.imshow("TRT bench", ov)
                if cv2.waitKey(1) & 0xFF == 27: break
    cap.release()
    print(f"Frames: {n} | Avg total latency: {t_sum/max(1,n):.2f} ms")
