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


def _panel(m, label):
    """640x640 side view with a label and a time stamp, for side-by-side comparisons."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    font = lambda s: ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", s)  # noqa: E731
    m.vis.global_.offwidth = m.vis.global_.offheight = 640
    ren, cam = mujoco.Renderer(m, 640, 640), mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90, -8, 3.2

    def frame(d):
        cam.lookat[:] = [d.subtree_com[1][0], 0, 0.75]
        ren.update_scene(d, cam)
        img = Image.fromarray(ren.render()); dr = ImageDraw.Draw(img)
        for i, line in enumerate(label.split("\n")):
            dr.text((16, 12 + 24 * i), line, fill="black", font=font(20 if i == 0 else 16))
        dr.text((470, 606), f"t = {d.time:.2f} s  (0.5x)", fill="white", font=font(18))
        return np.asarray(img)
    return frame, (640, 640)


def render(tr, m, out, tag, title=None, sub="", label=None, play=None, **play_kw):
    if label:
        shot, (w, h) = _panel(m, label)
    elif title:
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

    play = play or (lambda tr, **kw: mr.summarize(tr, mr.play(tr, hold=3.0, **kw), tag))
    res = play(tr, model=m, on_step=grab, **play_kw)
    ff.stdin.close(); ff.wait()
    print(out)
    return res


def compare_box(same_box: bool = False):
    """T-ASE (left) and Dai 2014 (right) onto a 0.10 m box, l 0.40, h 0.20, one 1280x720 video.
    Default: each method on the box edge where it stands ([T] 0.30; [D] 0.26, it falls at 0.22, 0.24, 0.28, 0.30;
    [T] is infeasible below 0.28). same_box: both at 0.30 ([D] falls) -> compare_box_l0.4_same.mp4.
    Plans: results/stage3_box_l0.4.npz, results/dai2014_box_l0.4{,_e0.26}.npz (each solved with its box)."""
    from humanoid_jump.sim import play_full as pf
    bt, bd = (0.10, 0.30), (0.10, 0.30 if same_box else 0.26)
    dai_npz, dai_txt = ("dai2014_box_l0.4.npz", "174.8 s, 1329 iters") if same_box else ("dai2014_box_l0.4_e0.26.npz", "194.8 s, 1248 iters")
    root, tmp = mr.ROOT / "results", []
    tr = mr.load(root / "stage3_box_l0.4.npz"); tr["l"] = 0.4
    tmp.append(root / "_left.mp4")
    render(tr, build_model(box=bt), tmp[-1], "T-ASE", z_land=0.78 + bt[0],
           label=f"T-ASE 3-step (12.7 s, 215 iters), edge x {bt[1]:.2f} m\nreal H1-2 torque limits, landing controller")
    td = pf.load(root / dai_npz)
    tmp.append(root / "_right.mp4")
    render(td, pf.unlimited_model(box=bd), tmp[-1], "Dai",
           play=lambda tr, **kw: mr.summarize(pf.until_touchdown(tr), pf.play(tr, **kw), "Dai"),
           label=f"Dai 2014 ({dai_txt}), edge x {bd[1]:.2f} m\ntorque limits removed, planned landing")
    edges = f"box edge at x {bt[1]:.2f} m" if same_box else f"box edge: T-ASE x {bt[1]:.2f} m, Dai x {bd[1]:.2f} m (each where it stands)"
    _side_by_side(tmp, f"Jump onto a {bt[0]:.2f} m box, 0.40 m forward (h 0.20 m)",
                  f"{edges}  |  left: T-ASE 3-step   right: Dai et al. 2014  |  side views",
                  root / f"compare_box_l0.4{'_same' if same_box else ''}.mp4", max(tr['t'][-1], td['t'][-1]) + 3.0)


def _side_by_side(tmp, title, sub, out, dur):
    """Two 640x640 panels under an 80 px title bar; deletes the panel files."""
    from PIL import Image, ImageDraw, ImageFont
    root = out.parent
    head = Image.new("RGB", (1280, 80), "white"); dr = ImageDraw.Draw(head)
    font = lambda s: ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", s)  # noqa: E731
    dr.text((20, 10), title, fill="black", font=font(24)); dr.text((20, 44), sub, fill=(90, 90, 90), font=font(18))
    head.save(root / "_head.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp[0]), "-i", str(tmp[1]), "-loop", "1", "-i", str(root / "_head.png"),
                    "-filter_complex", "[0]tpad=stop_mode=clone:stop_duration=4[a];[1]tpad=stop_mode=clone:stop_duration=4[b];"
                    "[a][b]hstack[ab];[2][ab]vstack=shortest=1[v]", "-map", "[v]", "-t", f"{dur * SLOW:.2f}",
                    "-pix_fmt", "yuv420p", "-vcodec", "libx264", str(out)], check=True)
    for p in tmp + [root / "_head.png"]:
        p.unlink()
    print(out)


def compare_flat():
    """Dai 2014, h 0.10 l 0.40: toe push-off (paper schedule, left) vs flat push-off (--flat, NOT in the paper, right).
    Both with motor limits removed and the planned landing, as in the [D] grid -> results/compare_dai_flat.mp4."""
    from humanoid_jump.sim import play_full as pf
    root, tmp, ts = mr.ROOT / "results", [], []
    for f, lab in [("dai2014.npz", "toe push-off (paper schedule), 1051 iters, 131 s\nplan ankle 2.57x limit, 6 joints over"),
                   ("dai2014_flat.npz", "flat push-off (all 8 corners), 1250 iters, 160 s\nplan ankle 1.99x, hip 2.01x, elbow 1.99x, 10 joints over")]:
        td = pf.load(root / f); ts.append(td["t"][-1])
        tmp.append(root / f"_{len(tmp)}.mp4")
        render(td, pf.unlimited_model(), tmp[-1], f,
               play=lambda tr, **kw: mr.summarize(pf.until_touchdown(tr), pf.play(tr, **kw), "Dai"), label=lab)
    _side_by_side(tmp, "Dai et al. 2014 on H1-2, 0.40 m forward, h 0.10 m: toe vs flat push-off",
                  "flat push-off is NOT in the paper (comparison only)  |  torque limits removed, planned landing  |  side views",
                  root / "compare_dai_flat.mp4", max(ts) + 3.0)

if __name__ == "__main__":
    if sys.argv[1] == "compare-box":
        compare_box(same_box="--same" in sys.argv); sys.exit(0)
    if sys.argv[1] == "compare-flat":
        compare_flat(); sys.exit(0)
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
