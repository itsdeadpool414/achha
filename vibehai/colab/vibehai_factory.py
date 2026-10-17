# VibeHai Aftermovie Factory engine (2026-10-17)
# Picks the best crowd moments from an event's approved reels, cleans them up,
# cuts them to the beat in 9:16 and (optionally) publishes the MP4 back to Supabase.
import os, sys, json, math, shutil, subprocess, time, random, re, urllib.request, urllib.parse
from datetime import datetime, timezone
import numpy as np
import cv2

W, H, FPS = 1080, 1920, 30
BUCKET = 'vibehai-media'
MAX_MB = 45  # keep under Supabase free-plan 50 MB upload limit


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def sh(cmd, quiet=True):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, errors='replace')
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd if isinstance(cmd, str) else ' '.join(cmd)}\n{r.stderr[-1500:]}")
    return r.stdout


def probe(path):
    out = sh(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', path])
    j = json.loads(out)
    v = next((s for s in j['streams'] if s['codec_type'] == 'video'), None)
    a = next((s for s in j['streams'] if s['codec_type'] == 'audio'), None)
    dur = float(j['format'].get('duration') or (v or {}).get('duration') or 0)
    rot = 0
    if v:
        for sd in v.get('side_data_list', []) or []:
            if 'rotation' in sd: rot = int(sd['rotation'])
        rot = int((v.get('tags') or {}).get('rotate', rot))
    w, h = (int(v['width']), int(v['height'])) if v else (0, 0)
    if abs(rot) in (90, 270): w, h = h, w
    return {'dur': dur, 'w': w, 'h': h, 'has_audio': a is not None}


# ---------------- encoder ----------------
def pick_encoder():
    try:
        sh('nvidia-smi -L')
        sh('ffmpeg -hide_banner -v error -f lavfi -i color=black:s=256x256:d=0.5 -c:v h264_nvenc -f null -')
        return ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', '21', '-b:v', '10M', '-maxrate', '16M', '-bufsize', '20M']
    except Exception:
        return ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20']


def has_filter(name):
    try:
        return f' {name} ' in sh('ffmpeg -hide_banner -filters')
    except Exception:
        return False


# ---------------- Supabase (REST + storage, service key from Colab Secrets) ----------------
class Supa:
    def __init__(self, url, key):
        self.url, self.key = url.rstrip('/'), key

    def _req(self, method, path, body=None, headers=None, raw=False):
        h = {'apikey': self.key, 'Authorization': 'Bearer ' + self.key}
        if headers: h.update(headers)
        data = body if (raw or body is None) else json.dumps(body).encode()
        if body is not None and not raw: h.setdefault('Content-Type', 'application/json')
        req = urllib.request.Request(self.url + path, data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                t = r.read()
                return json.loads(t) if t and r.headers.get('Content-Type', '').startswith('application/json') else t
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'Supabase {method} {path.split("?")[0]} -> {e.code}: {e.read().decode()[:400]}')

    def event(self, slug):
        q = urllib.parse.quote(slug)
        r = self._req('GET', f'/rest/v1/vh_events?slug=eq.{q}&select=id,slug,name,venue,starts_at,ends_at,vip,hidden')
        return r[0] if r else None

    def ended_events_without_movie(self):
        now = datetime.now(timezone.utc).isoformat()
        return self._req('GET', f'/rest/v1/vh_events?ends_at=lt.{urllib.parse.quote(now)}&aftermovie_path=is.null&hidden=not.is.true&select=slug,name')

    def reels(self, event_id):
        sel = 'id,video_path,status,vip,likes,views,shares,created_at,device:vh_devices(name,instagram)'
        try:
            return self._req('GET', f'/rest/v1/vh_reels?event_id=eq.{event_id}&status=in.(approved,featured)&select={sel}&order=created_at')
        except RuntimeError:  # older schema without vip column
            sel = sel.replace('vip,', '')
            return self._req('GET', f'/rest/v1/vh_reels?event_id=eq.{event_id}&status=in.(approved,featured)&select={sel}&order=created_at')

    def public_url(self, path):
        return f'{self.url}/storage/v1/object/public/{BUCKET}/{urllib.parse.quote(path)}'

    def object_exists(self, path):
        try:
            urllib.request.urlopen(urllib.request.Request(self.public_url(path), method='HEAD'), timeout=30); return True
        except Exception:
            return False

    def upload(self, path, file, ctype='video/mp4'):
        with open(file, 'rb') as f:
            self._req('POST', f'/storage/v1/object/{BUCKET}/{urllib.parse.quote(path)}', f.read(), {'Content-Type': ctype, 'x-upsert': 'true'}, raw=True)

    def mark_ready(self, event_id, path, meta):
        self._req('PATCH', f'/rest/v1/vh_events?id=eq.{event_id}',
                  {'aftermovie_path': path, 'aftermovie_at': datetime.now(timezone.utc).isoformat(), 'aftermovie_meta': meta},
                  {'Prefer': 'return=minimal'})


def download(url, dest):
    req = urllib.request.Request(url, headers={'User-Agent': 'vibehai-factory'})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, 'wb') as f:
        shutil.copyfileobj(r, f)


# ---------------- analysis ----------------
_face = None
def faces(gray):
    global _face
    if _face is None:
        try: _face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception: _face = False  # OpenCV build without Haar cascades: skip face score
    if _face is False: return 0
    return len(_face.detectMultiScale(gray, 1.2, 4, minSize=(18, 18)))


def dhash(gray):
    s = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (s[:, 1:] > s[:, :-1]).flatten()


class Clipper:
    """Optional CLIP 'is this a hype crowd moment' score; used only when torch + open_clip load (GPU runtime)."""
    def __init__(self):
        self.ok = False
        try:
            import torch, open_clip
            self.torch = torch
            self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            if self.dev != 'cuda': return
            self.model, _, self.prep = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.dev)
            tok = open_clip.get_tokenizer('ViT-B-32')
            pos = ['a hyped crowd dancing at a live concert', 'festival stage with lights and a cheering crowd', 'friends celebrating at a college fest']
            neg = ['a blurry dark photo', 'the floor or a pocket', 'an empty room']
            with torch.no_grad():
                t = self.model.encode_text(tok(pos + neg).to(self.dev)); t = t / t.norm(dim=-1, keepdim=True)
            self.t, self.npos, self.ok = t, len(pos), True
        except Exception as e:
            print('   (CLIP scoring off:', str(e)[:80], ')')

    def score(self, frames_bgr):
        if not self.ok or not frames_bgr: return None
        from PIL import Image
        torch = self.torch
        ims = torch.stack([self.prep(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames_bgr]).to(self.dev)
        with torch.no_grad():
            e = self.model.encode_image(ims); e = e / e.norm(dim=-1, keepdim=True)
            p = (100 * e @ self.t.T).softmax(dim=-1)
        return p[:, :self.npos].sum(dim=1).cpu().numpy()  # 0..1 per frame


def audio_energy(path, dur, sr=22050):
    """RMS + onset strength per second (0..1). Crowd noise / drops score high."""
    wav = path + '.wav'
    try:
        sh(['ffmpeg', '-y', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(sr), wav])
        import librosa
        y, _ = librosa.load(wav, sr=sr, mono=True)
        os.remove(wav)
        if len(y) < sr // 2: raise ValueError
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        on = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        n = max(1, int(math.ceil(dur)))
        fps_a = sr / hop
        per = []
        for s in range(n):
            a, b = int(s * fps_a), int((s + 1) * fps_a)
            per.append((float(np.mean(rms[a:b])) if b > a and a < len(rms) else 0.0, float(np.mean(on[a:b])) if b > a and a < len(on) else 0.0))
        per = np.array(per)
        return per[:, 0], per[:, 1]
    except Exception:
        n = max(1, int(math.ceil(dur)))
        return np.zeros(n), np.zeros(n)


def analyse(path, clipper=None, sample_fps=4):
    info = probe(path)
    dur = info['dur']
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(src_fps / sample_fps)))
    prev, idx = None, 0
    rows, hashes, clip_frames = [], [], []
    while True:
        ok = cap.grab()
        if not ok: break
        if idx % step == 0:
            ok, fr = cap.retrieve()
            if not ok: break
            t = idx / src_fps
            hh, ww = fr.shape[:2]
            sc = 320 / max(hh, ww)
            small = cv2.resize(fr, (max(16, int(ww * sc)), max(16, int(hh * sc))))
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            bright = float(g.mean())
            sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
            motion, shake = 0.0, 0.0
            if prev is not None and prev.shape == g.shape:
                motion = float(cv2.absdiff(g, prev).mean())
                (dx, dy), _ = cv2.phaseCorrelate(np.float32(prev), np.float32(g))
                shake = math.hypot(dx, dy) / max(g.shape) * 100  # % of frame per sample
            nf = faces(g) if len(rows) % 2 == 0 else (rows[-1]['faces'] if rows else 0)
            rows.append({'t': t, 'bright': bright, 'sharp': sharp, 'motion': motion, 'shake': shake, 'faces': nf})
            if len(rows) % 8 == 1: hashes.append(dhash(g))
            if clipper is not None and clipper.ok and len(rows) % 4 == 1: clip_frames.append((t, small))
            prev = g
        idx += 1
    cap.release()
    if not rows: return None
    dur = dur or rows[-1]['t']
    rms, onset = audio_energy(path, dur)
    clip_scores = {}
    if clip_frames:
        sc = clipper.score([f for _, f in clip_frames])
        if sc is not None:
            for (t, _), v in zip(clip_frames, sc): clip_scores[int(t)] = float(v)
    n = max(1, int(math.ceil(dur)))
    sec = []
    for s in range(n):
        rs = [r for r in rows if s <= r['t'] < s + 1] or [min(rows, key=lambda r: abs(r['t'] - s))]
        sec.append({
            'bright': np.mean([r['bright'] for r in rs]), 'sharp': np.mean([r['sharp'] for r in rs]),
            'motion': np.mean([r['motion'] for r in rs]), 'shake': np.mean([r['shake'] for r in rs]),
            'faces': np.mean([r['faces'] for r in rs]), 'rms': rms[min(s, len(rms) - 1)], 'onset': onset[min(s, len(onset) - 1)],
            'clip': clip_scores.get(s)})
    return {'path': path, 'dur': dur, 'info': info, 'sec': sec, 'hash': hashes[len(hashes) // 2] if hashes else None}


def norm(vals):
    v = np.array(vals, dtype=float)
    if v.size == 0: return v
    lo, hi = np.percentile(v, 5), np.percentile(v, 95)
    return np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)


def score_all(clips, seg_len=2.0):
    """Adds per-second 'moment' scores using batch-normalised features, and a clip-level quality verdict."""
    keys = ['sharp', 'motion', 'rms', 'onset', 'faces']
    allv = {k: [s[k] for c in clips for s in c['sec']] for k in keys}
    nz = {k: norm(allv[k]) for k in keys}
    i = 0
    for c in clips:
        for s in c['sec']:
            f = {k: nz[k][i] for k in keys}
            motion_good = 1 - abs(f['motion'] - 0.6) / 0.6  # lively, not chaotic
            light = np.clip((s['bright'] - 35) / 60, 0, 1)
            steady = np.clip(1 - s['shake'] / 6, 0, 1)
            sc = 0.26 * (0.6 * f['rms'] + 0.4 * f['onset']) + 0.18 * f['sharp'] + 0.16 * motion_good + 0.12 * min(1, f['faces']) + 0.14 * light + 0.14 * steady
            if s['clip'] is not None: sc = 0.7 * sc + 0.3 * s['clip']
            s['score'] = float(sc)
            i += 1
        sec = c['sec']
        c['mean_bright'] = float(np.mean([s['bright'] for s in sec]))
        c['mean_shake'] = float(np.mean([s['shake'] for s in sec]))
        c['mean_sharp'] = float(np.mean([s['sharp'] for s in sec]))
        reasons = []
        if c['mean_bright'] < 28: reasons.append('too dark')
        if c['mean_shake'] > 7: reasons.append('too shaky')
        if c['mean_sharp'] < 12: reasons.append('blurry')
        if c['dur'] < 1.5: reasons.append('too short')
        c['reject'] = reasons
        # best window
        L = max(1, int(round(seg_len)))
        best, bt = -1, 0
        for st in range(0, max(1, len(sec) - L + 1)):
            v = np.mean([s['score'] for s in sec[st:st + L]])
            if v > best: best, bt = v, st
        c['best_t'], c['best'] = float(bt), float(best)
        c['quality'] = float(np.mean([s['score'] for s in sec]))


def drop_duplicates(clips):
    keep = []
    for c in sorted(clips, key=lambda c: -(c['best'] + c.get('bonus', 0))):
        dup = None
        if c['hash'] is not None:
            for k in keep:
                if k['hash'] is not None and np.count_nonzero(c['hash'] != k['hash']) <= 5:
                    dup = k; break
        if dup: c['reject'] = c['reject'] + ['duplicate of ' + dup['name']]
        else: keep.append(c)
    return keep


# ---------------- music / beats ----------------
def beat_grid(music, total, bpm_fallback=120):
    if music:
        try:
            import librosa
            y, sr = librosa.load(music, sr=22050, mono=True, duration=total + 10)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            bt = librosa.frames_to_time(beats, sr=sr)
            tempo = float(np.atleast_1d(tempo)[0])
            if len(bt) > 8: return list(bt), tempo
        except Exception as e:
            print('   beat detection failed, using 120 BPM grid:', str(e)[:80])
    step = 60 / bpm_fallback
    return [i * step for i in range(int(total / step) + 4)], float(bpm_fallback)


def plan_cuts(beats, start, total, rng):
    """Cut lengths of 4 beats, sometimes 2 (fast) or 8 (hold), snapped to the beat grid."""
    beats = [b for b in beats if b >= start]
    cuts, i = [], 0
    while i < len(beats) - 1 and beats[i] - start < total:
        n = rng.choice([4, 4, 4, 2, 8]) if cuts else 4
        j = min(i + n, len(beats) - 1)
        cuts.append(beats[j] - beats[i]); i = j
    return cuts


# ---------------- rendering ----------------
def font(size, bold=True):
    from PIL import ImageFont
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if os.path.exists(p): return ImageFont.truetype(p, size)
    try: return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()


def make_cards(work, title, subtitle, credits):
    from PIL import Image, ImageDraw, ImageFilter
    def center(d, y, text, f, fill):
        w = d.textbbox((0, 0), text, font=f)[2]
        d.text(((W - w) / 2, y), text, font=f, fill=fill)
    def wrap(text, f, maxw, d):
        words, lines, cur = text.split(), [], ''
        for wd in words:
            t = (cur + ' ' + wd).strip()
            if d.textbbox((0, 0), t, font=f)[2] > maxw and cur: lines.append(cur); cur = wd
            else: cur = t
        return lines + ([cur] if cur else [])
    # title card
    im = Image.new('RGB', (W, H), (8, 6, 16)); d = ImageDraw.Draw(im)
    for r in range(900, 0, -30):
        a = int(70 * (1 - r / 900)); d.ellipse([W / 2 - r, H / 2 - r - 200, W / 2 + r, H / 2 + r - 200], fill=(40 + a, 12, 90 + a))
    im = im.filter(ImageFilter.GaussianBlur(40)); d = ImageDraw.Draw(im)
    center(d, 640, 'VIBEHAI PRESENTS', font(44), (255, 215, 0))
    ft = font(96); y = 760
    for ln in wrap(title.upper(), ft, W - 140, d)[:3]:
        center(d, y, ln, ft, (255, 255, 255)); y += 112
    if subtitle: center(d, y + 30, subtitle, font(40, False), (200, 190, 230))
    im.save(f'{work}/title.png')
    # end card
    im = Image.new('RGB', (W, H), (8, 6, 16)); d = ImageDraw.Draw(im)
    center(d, 720, 'YAAR, VIBE HAI', font(90), (255, 255, 255))
    center(d, 850, 'Filmed by fans who were really there', font(38, False), (200, 190, 230))
    fc = font(36, False); y = 960
    for ln in wrap('  ·  '.join(credits), fc, W - 160, d)[:10]:
        center(d, y, ln, fc, (255, 215, 0)); y += 52
    center(d, 1700, 'vibehai  ·  @originalvibehai', font(40), (180, 140, 255))
    im.save(f'{work}/end.png')
    # watermark
    wm = Image.new('RGBA', (420, 90), (0, 0, 0, 0)); d = ImageDraw.Draw(wm)
    d.rounded_rectangle([0, 0, 419, 89], 45, fill=(0, 0, 0, 110))
    f = font(46); w = d.textbbox((0, 0), 'VIBEHAI', font=f)[2]
    d.text(((420 - w) / 2, 16), 'VIBEHAI', font=f, fill=(255, 255, 255, 230))
    wm.save(f'{work}/wm.png')


def seg_filter(info, stabilise, upscale, vidstab):
    vertical = info['h'] >= info['w']
    stab = 'deshake=rx=24:ry=24,' if stabilise and not vidstab else ''
    grade = 'eq=contrast=1.07:saturation=1.18:brightness=0.015:gamma=0.98,'
    sharp = 'unsharp=5:5:0.9:5:5:0.0,' if upscale else 'unsharp=3:3:0.35,'
    if vertical:
        return f"[0:v]{stab}scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},{grade}{sharp}fps={FPS},setsar=1,format=yuv420p[v]"
    # landscape: blurred fill behind a centred frame
    return (f"[0:v]{stab}split=2[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:3,eq=brightness=-0.12[bg];"
            f"[b]scale={W}:-2:flags=lanczos[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,{grade}{sharp}fps={FPS},setsar=1,format=yuv420p[v]")


def render_segment(c, start, dur, out, enc, stabilise, upscale, vidstab):
    src = c['path']
    pre = []
    if stabilise and vidstab:
        trf = out + '.trf'
        try:
            sh(['ffmpeg', '-y', '-v', 'error', '-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', src, '-vf', f'vidstabdetect=shakiness=6:result={trf}', '-f', 'null', '-'])
            pre = [f'vidstabtransform=input={trf}:smoothing=12:zoom=4']
        except Exception:
            vidstab = False
    vf = seg_filter(c['info'], stabilise, upscale, vidstab)
    if pre: vf = vf.replace('[0:v]', '[0:v]' + ','.join(pre) + ',', 1)
    args = ['ffmpeg', '-y', '-v', 'error', '-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', src]
    if c['info']['has_audio']:
        af = f"[0:a]aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:{dur:.3f},afade=t=in:d=0.04,afade=t=out:st={max(0, dur - 0.06):.3f}:d=0.06[a]"
    else:
        args += ['-f', 'lavfi', '-t', f'{dur:.3f}', '-i', 'anullsrc=r=48000:cl=stereo']; af = '[1:a]atrim=0:%.3f[a]' % dur
    args += ['-filter_complex', vf + ';' + af, '-map', '[v]', '-map', '[a]', '-t', f'{dur:.3f}'] + enc + ['-c:a', 'aac', '-b:a', '192k', out]
    sh(args)


def still_segment(png, dur, out, enc):
    sh(['ffmpeg', '-y', '-v', 'error', '-loop', '1', '-t', f'{dur:.3f}', '-i', png, '-f', 'lavfi', '-t', f'{dur:.3f}', '-i', 'anullsrc=r=48000:cl=stereo',
        '-vf', f'scale={W}:{H},zoompan=z=\'min(zoom+0.0006,1.06)\':d={int(dur * FPS)}:s={W}x{H}:fps={FPS},fade=t=in:d=0.4,fade=t=out:st={dur - 0.4:.2f}:d=0.4,format=yuv420p,setsar=1',
        '-map', '0:v', '-map', '1:a', '-t', f'{dur:.3f}'] + enc + ['-c:a', 'aac', '-b:a', '192k', out])


# ---------------- main per-event pipeline ----------------
def make_aftermovie(slug, supa=None, local_clips=None, work_root='/content/vibehai_work', target=55, music=None,
                    stabilise=True, upscale=False, publish=True, seed=7, event_override=None):
    t0 = time.time()
    rng = random.Random(seed)
    work = os.path.join(work_root, slug); os.makedirs(work, exist_ok=True)
    seg_dir = os.path.join(work, 'segs'); shutil.rmtree(seg_dir, ignore_errors=True); os.makedirs(seg_dir)
    log(f'🎬 [{slug}] starting')

    # 1) clips
    if local_clips is not None:
        ev = event_override or {'id': None, 'name': slug, 'venue': '', 'vip': None}
        reels = local_clips
    else:
        ev = supa.event(slug)
        if not ev: raise RuntimeError(f'Event "{slug}" not found. Check the slug (it is in the event link: #/e/<slug>).')
        reels = supa.reels(ev['id'])
        log(f'   {len(reels)} approved reels found for "{ev["name"]}"')
        for i, r in enumerate(reels):
            dst = os.path.join(work, f'clip_{i:03d}' + os.path.splitext(r['video_path'])[1])
            if not os.path.exists(dst):
                download(supa.public_url(r['video_path']), dst)
            r['local'] = dst
            print(f'   ⬇ {i + 1}/{len(reels)} downloaded', end='\r')
        print()
    if not reels: raise RuntimeError('No approved reels yet. Approve some reels in the organiser portal first.')

    # 2) analyse
    clipper = Clipper() if local_clips is None else None
    clips = []
    for i, r in enumerate(reels):
        try:
            a = analyse(r['local'], clipper)
        except Exception as e:
            log(f'   ⚠ could not read clip {i + 1}: {str(e)[:80]}'); continue
        if not a: continue
        dev = r.get('device') or {}
        a.update({'name': f'clip{i + 1}', 'id': r.get('id'), 'vip': bool(r.get('vip')), 'featured': r.get('status') == 'featured',
                  'creator': ('@' + dev['instagram'].lstrip('@')) if dev.get('instagram') else (dev.get('name') or 'fan'),
                  'created_at': r.get('created_at', ''), 'likes': r.get('likes') or 0})
        clips.append(a)
        print(f'   🔍 analysed {i + 1}/{len(reels)}', end='\r')
    print()
    score_all(clips)
    for c in clips:
        eng = min(0.08, 0.01 * math.log1p((c.get('likes') or 0)))
        c['bonus'] = (0.25 if c['vip'] else 0) + (0.15 if c['featured'] else 0) + eng
    good = [c for c in clips if not c['reject']]
    good = drop_duplicates(good)
    for c in clips:
        if c['reject']: log(f"   ✗ dropped {c['name']} ({', '.join(c['reject'])})")
    if not clips: raise RuntimeError('None of the reels could be read (corrupt or unsupported files).')
    if not good:  # be forgiving on tiny events: use the least-bad clips
        good = sorted(clips, key=lambda c: -c['quality'])[:6]; log('   ⚠ every clip had problems; using the least-bad ones')
    good.sort(key=lambda c: -(c['best'] + c['bonus']))
    log(f'   ✓ {len(good)} usable clips ({sum(c["vip"] for c in good)} VIP-zone)')

    # 3) plan to the beat
    title_d, end_d = 2.4, 3.0
    body = max(10.0, target - title_d - end_d)
    beats, tempo = beat_grid(music, target)
    first_beat = next((b for b in beats if b >= title_d - 0.05), title_d)
    title_d = first_beat  # title ends exactly on a beat
    cuts = plan_cuts(beats, first_beat, body, rng)
    total_media = sum(c['dur'] for c in good)
    # each clip can give several windows, best-first; VIP + top clips first, then by time for story flow
    pool = []
    for c in good:
        order = sorted(range(len(c['sec'])), key=lambda s: -c['sec'][s]['score'])
        used = []
        for s in order:
            if all(abs(s - u) >= 3 for u in used): used.append(s)
            if len(used) >= 3: break
        for rank, s in enumerate(used):
            pool.append({'c': c, 't': float(s), 'score': c['sec'][s]['score'] + c['bonus'] - 0.12 * rank, 'rank': rank})
    pool.sort(key=lambda p: -p['score'])
    picks = pool[:len(cuts)]
    while len(picks) < len(cuts) and pool: picks.append(pool[len(picks) % len(pool)])  # tiny events: reuse
    hook = picks[0] if picks else None
    rest = sorted(picks[1:], key=lambda p: (not p['c']['vip'], p['c']['created_at'], p['t']))
    # spread VIP shots through the first half instead of a block
    vip_r = [p for p in rest if p['c']['vip']]; oth = [p for p in rest if not p['c']['vip']]
    seq = [hook] if hook else []
    while vip_r or oth:
        if vip_r: seq.append(vip_r.pop(0))
        if oth: seq.append(oth.pop(0))
        if oth and len(seq) > len(picks) // 2: seq += oth; oth = []
    # never show the same clip twice in a row
    for k in range(1, len(seq)):
        if seq[k]['c'] is seq[k - 1]['c']:
            for j in range(k + 1, len(seq)):
                if seq[j]['c'] is not seq[k - 1]['c'] and (k + 1 >= len(seq) or seq[j]['c'] is not seq[k + 1]['c']):
                    seq[k], seq[j] = seq[j], seq[k]; break
    log(f'   🥁 {tempo:.0f} BPM · {len(cuts)} cuts · ~{title_d + sum(cuts) + end_d:.0f}s')

    # 4) render
    enc = pick_encoder()
    vidstab = stabilise and has_filter('vidstabdetect')
    log(f"   ⚙ encoder {enc[1]}{' · vidstab' if vidstab else (' · deshake' if stabilise else '')}{' · upscale' if upscale else ''}")
    credits = list(dict.fromkeys(p['c']['creator'] for p in seq))
    sub = ' · '.join(x for x in [ev.get('venue') or '', (ev.get('starts_at') or '')[:10]] if x)
    make_cards(work, ev.get('name') or slug, sub, credits)
    parts = []
    still_segment(f'{work}/title.png', title_d, f'{seg_dir}/000_title.mp4', enc); parts.append(f'{seg_dir}/000_title.mp4')
    for k, (p, d) in enumerate(zip(seq, cuts)):
        c = p['c']
        st = max(0.0, min(p['t'], c['dur'] - d - 0.05)) if c['dur'] > d else 0.0
        d_use = min(d, max(0.5, c['dur'] - st))
        out = f'{seg_dir}/{k + 1:03d}.mp4'
        render_segment(c, st, d_use, out, enc, stabilise, upscale, vidstab)
        parts.append(out)
        print(f'   🎞 cut {k + 1}/{len(cuts)} ({c["name"]}{" VIP" if c["vip"] else ""})', end='\r')
    print()
    still_segment(f'{work}/end.png', end_d, f'{seg_dir}/999_end.mp4', enc); parts.append(f'{seg_dir}/999_end.mp4')
    with open(f'{work}/list.txt', 'w') as f:
        f.write(''.join(f"file '{p}'\n" for p in parts))
    joined = f'{work}/joined.mp4'
    sh(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', f'{work}/list.txt', '-c', 'copy', joined])
    length = probe(joined)['dur']

    # 5) final: watermark + music (crowd audio ducked under the track)
    final = f'{work}/{slug}_aftermovie.mp4'
    wm = f'[0:v][2:v]overlay=W-w-40:60:enable=\'between(t,{title_d:.2f},{length - end_d:.2f})\',format=yuv420p[v]'
    if music:
        af = (f'[1:a]atrim=0:{length:.3f},afade=t=in:d=0.3,afade=t=out:st={length - 2.5:.3f}:d=2.5,volume=0.9[m];'
              f'[0:a]volume=0.35[c];[m][c]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]')
        ins = ['-i', joined, '-i', music, '-i', f'{work}/wm.png']
    else:
        af = f'[0:a]loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=out:st={length - 2:.3f}:d=2[a]'
        ins = ['-i', joined, '-f', 'lavfi', '-i', 'anullsrc', '-i', f'{work}/wm.png']
    def final_pass(vbr):  # bitrate-capped so the file fits Supabase's upload limit (50 MB on the free plan)
        fenc = (['-c:v', 'h264_nvenc', '-preset', 'p6', '-rc', 'vbr', '-cq', '23', '-b:v', vbr, '-maxrate', vbr, '-bufsize', '2M'] if 'nvenc' in enc[1]
                else ['-c:v', 'libx264', '-preset', 'medium', '-crf', '21', '-maxrate', vbr, '-bufsize', '8M'])
        sh(['ffmpeg', '-y', '-v', 'error'] + ins + ['-filter_complex', wm + ';' + af, '-map', '[v]', '-map', '[a]', '-t', f'{length:.3f}'] + fenc +
           ['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', final])
    kbps = int(min(8000, max(1500, (MAX_MB * 8 * 1000 / max(length, 1)) - 200)))
    final_pass(f'{kbps}k')
    while os.path.getsize(final) > MAX_MB * 1e6 and kbps > 1500:
        kbps = int(kbps * 0.75); log(f'   file too big, re-encoding at {kbps} kbps'); final_pass(f'{kbps}k')
    info = probe(final)
    log(f'   ✅ rendered {info["dur"]:.1f}s {info["w"]}x{info["h"]} → {final} ({os.path.getsize(final) / 1e6:.1f} MB)')

    meta = {'duration_s': round(info['dur'], 1), 'cuts': len(cuts), 'clips_used': len({id(p['c']) for p in seq}),
            'vip_clips': len({id(p['c']) for p in seq if p['c']['vip']}), 'bpm': round(tempo), 'creators': credits[:30],
            'music': bool(music), 'encoder': enc[1], 'rendered_at': datetime.now(timezone.utc).isoformat()}
    # 6) publish
    if publish and supa and ev.get('id'):
        path = f'aftermovies/{slug}/{int(time.time())}_{random.randrange(16**6):06x}.mp4'
        log('   ⬆ uploading to vibehai-media …')
        supa.upload(path, final)
        supa.mark_ready(ev['id'], path, meta)
        meta['path'] = path
        log(f'   🚀 live: fans who unlocked this event now get the final cut ({supa.public_url(path)})')
    log(f'🎉 [{slug}] done in {time.time() - t0:.0f}s')
    return final, meta
