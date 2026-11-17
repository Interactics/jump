"""MuJoCo video of one jump (saved plan, same playback as evaluate.py), half speed.

    python scripts/render_video.py 0.1 0.4                    # results/eval/video_h0.1_l0.4.mp4
    python scripts/render_video.py 0.1 0.4 "friction 0.3"     # on a perturbed model (names from evaluate.perturbations)
    python scripts/render_video.py box                        # results/stage3_box.npz onto the box -> results/video_box.mp4
"""
import subprocess
import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid_jump.sim import mujoco_runner as mr  # noqa: E402
from humanoid_jump.sim.robot import build_model  # noqa: E402
from humanoid_jump.stages.stage3_nlp import BOX  # noqa: E402
from scripts.evaluate import OUT, perturbations  # noqa: E402

FPS, SLOW, W, H = 30, 2.0, 640, 480   # ponytail: ffmpeg pipe instead of imageio (not installed)


def _two_views(m, title, sub):
    """1280x720 frame like results/dai2014_box.mp4: title bar, side view | 3/4 view, time stamp."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    font = lambda s: ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", s)  # noqa: E731
    m.vis.global_.offwidth = m.vis.global_.offheight = 640                 # default framebuffer is 640x480
    ren, cams = mujoco.Renderer(m, 640, 640), [mujoco.MjvCamera(), mujoco.MjvCamera()]
    cams[0].azimuth, cams[0].elevation, cams[0].distance = 90, -8, 3.2      # side
    cams[1].azimuth, cams[1].elevation, cams[1].distance = 130, -18, 3.4    # 3/4, front-left
    head = Image.new("RGB", (1280, 80), "white")
    dr = ImageDraw.Draw(head)
    dr.text((20, 10), title, fill="black", font=font(24)); dr.text((20, 44), sub, fill=(90, 90, 90), font=font(18))

    def frame(d):
        views = []
        for c in cams:
            c.lookat[:] = [d.subtree_com[1][0], 0, 0.75]
            ren.update_scene(d, c); views.append(ren.render())
        body = np.hstack([views[0], np.full((640, 2, 3), 200, np.uint8), views[1][:, 2:]])
        img = Image.new("RGB", (1280, 720)); img.paste(head, (0, 0)); img.paste(Image.fromarray(body), (0, 80))
        ImageDraw.Draw(img).text((1050, 684), f"t = {d.time:.2f} s  (0.5x)", fill="white", font=font(18))
        return np.asarray(img)
    return frame, (1280, 720)


def render(tr, m, out, tag, title=None, sub="", **play_kw):
    if title:
        shot, (w, h) = _two_views(m, title, sub)
    else:
        ren, cam = mujoco.Renderer(m, H, W), mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 90, -8, 3.4
        w, h = W, H

        def shot(d):
            cam.lookat[:] = [d.subtree_com[1][0], 0, 0.75]
            ren.update_scene(d, cam)
            return ren.render()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                           "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", str(out)], stdin=subprocess.PIPE)
    nxt = [0.0]

    def grab(d):
        if d.time >= nxt[0]:
            nxt[0] += 1 / (FPS * SLOW)
            ff.stdin.write(shot(d).tobytes())

    res = mr.summarize(tr, mr.play(tr, model=m, hold=3.0, on_step=grab, **play_kw), tag)
    ff.stdin.close(); ff.wait()
    print(out)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "box":     # landing controller: CoM target raised by the box height
        tr = mr.load(mr.ROOT / "results" / "stage3_box.npz"); tr["l"] = 0.5
        render(tr, build_model(box=BOX), mr.ROOT / "results" / "video_box.mp4", f"box {BOX}",
               title=f"T-ASE 3-step: jump onto a {BOX[0]:.2f} m box (0.50 m forward)",
               sub=f"real H1-2 torque limits  |  edge at x {BOX[1]:.2f} m  |  left: side view   right: 3/4 view",
               z_land=0.78 + BOX[0])                                       # 0.78 = LandingController default
        sys.exit(0)
    h, l, err = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else None)
    m = build_model()
    if err:
        perturbations()[err](m); mujoco.mj_setConst(m, mujoco.MjData(m))
    tr = mr.load(OUT / f"traj_h{h}_l{l}.npz"); tr["l"] = float(l)
    out = OUT / f"video_h{h}_l{l}{'_' + err.replace(' ', '_').replace('%', 'pct') if err else ''}.mp4"
    render(tr, m, out, f"h {h} l {l}" + (f", {err}" if err else ""))
