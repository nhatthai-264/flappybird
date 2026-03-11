import pygame, random, time, threading, math, os, urllib.request
from pygame.locals import *

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAND_CONTROL_AVAILABLE = True
except ImportError:
    HAND_CONTROL_AVAILABLE = False
    print("[WARNING] mediapipe/opencv not found. pip install mediapipe opencv-python")

# ── MediaPipe model paths ─────────────────────────────────────────────────────
_DIR = os.path.dirname(__file__)

HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_PATH = os.path.join(_DIR, "hand_landmarker.task")

FACE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_MODEL_PATH = os.path.join(_DIR, "face_landmarker.task")

def _ensure_model(url, path):
    if not os.path.exists(path):
        name = os.path.basename(path)
        print(f"[Model] Đang tải {name} (~30 MB)...")
        urllib.request.urlretrieve(url, path)
        print(f"[Model] Tải {name} xong!")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SCREEN_WIDHT  = 400
SCREEN_HEIGHT = 600
SPEED         = 20
GRAVITY       = 1.8
GAME_SPEED    = 15

GROUND_WIDHT  = 2 * SCREEN_WIDHT
GROUND_HEIGHT = 100
PIPE_WIDHT    = 80
PIPE_HEIGHT   = 500
PIPE_GAP      = 150

PINCH_THRESHOLD = 0.06   # khoảng cách chuẩn hóa ngón cái–trỏ

# EAR threshold: nhỏ hơn = đang nhắm mắt
EAR_THRESHOLD   = 0.20
EAR_CONSEC      = 2      # số frame liên tiếp EAR < threshold → xác nhận chớp

wing = 'assets/audio/wing.wav'
hit  = 'assets/audio/hit.wav'

pygame.mixer.init()


# ─────────────────────────────────────────────────────────────────────────────
# HAND GESTURE CONTROLLER  (ngón chạm → bay lên liên tục)
# ─────────────────────────────────────────────────────────────────────────────
class HandGestureController:
    def __init__(self):
        self._running         = False
        self._is_pinching     = False
        self._pinch_triggered = False
        self._prev_pinched    = False
        self._lock            = threading.Lock()
        self._thread          = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False

    def is_pinching(self) -> bool:
        with self._lock:
            return self._is_pinching

    def consume_pinch(self) -> bool:
        with self._lock:
            if self._pinch_triggered:
                self._pinch_triggered = False
                return True
            return False

    # alias để game loop dùng chung interface với Eye
    def consume_trigger(self) -> bool:
        return self.consume_pinch()

    def _run(self):
        try:
            _ensure_model(HAND_MODEL_URL, HAND_MODEL_PATH)
        except Exception as e:
            print(f"[Hand] Không tải được model: {e}"); return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Hand] Không mở được webcam."); return

        base_opts = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts, num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6)

        with mp_vision.HandLandmarker.create_from_options(opts) as lmk:
            while self._running:
                ok, frame = cap.read()
                if not ok: continue
                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]

                img    = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = lmk.detect(img)

                pinched = False
                dist    = 1.0

                if result.hand_landmarks:
                    lms   = result.hand_landmarks[0]
                    thumb = lms[4]; index = lms[8]
                    dist    = math.hypot(thumb.x - index.x, thumb.y - index.y)
                    pinched = dist < PINCH_THRESHOLD

                    for lm in lms:
                        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (180,180,180), -1)

                    tx, ty = int(thumb.x*w), int(thumb.y*h)
                    ix, iy = int(index.x*w), int(index.y*h)
                    col    = (0,255,0) if pinched else (0,120,255)
                    cv2.circle(frame, (tx,ty), 14, col, -1)
                    cv2.circle(frame, (ix,iy), 14, col, -1)
                    cv2.line(frame, (tx,ty), (ix,iy), col, 3)
                    cv2.putText(frame, "CHAP! Bay len ^" if pinched else "Mo tay Roi xuong v",
                                (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                    cv2.putText(frame, f"dist:{dist:.3f} thr:{PINCH_THRESHOLD}",
                                (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                else:
                    cv2.putText(frame, "Khong thay ban tay...",
                                (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

                with self._lock:
                    self._is_pinching = pinched
                    if pinched and not self._prev_pinched:
                        self._pinch_triggered = True
                self._prev_pinched = pinched

                # distance bar
                bx, by = 10, h-20
                bl = int((1-min(dist,1.0))*(w-20))
                cv2.rectangle(frame,(bx,by-14),(bx+w-20,by),(50,50,50),-1)
                if bl > 0:
                    cv2.rectangle(frame,(bx,by-14),(bx+bl,by),(0,255,0) if pinched else (0,120,255),-1)
                cv2.putText(frame,"Khoang cach ngon tay",(bx,by-17),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)

                cv2.imshow("TAY - Flappy Bird", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._running = False; break

        cap.release(); cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# EYE BLINK CONTROLLER  (chớp mắt → trigger bay lên 1 lần)
# ─────────────────────────────────────────────────────────────────────────────
class EyeBlinkController:
    # Landmark indices (MediaPipe Face Landmarker 478-point model)
    _LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    _RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self):
        self._running         = False
        self._blink_triggered = False
        self._ear_counter     = 0    # frames liên tiếp EAR < threshold
        self._blinked         = False
        self._lock            = threading.Lock()
        self._thread          = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False

    def consume_trigger(self) -> bool:
        """True một lần ngay sau mỗi cái chớp mắt."""
        with self._lock:
            if self._blink_triggered:
                self._blink_triggered = False
                return True
            return False

    # alias compatibility
    def consume_pinch(self) -> bool:
        return self.consume_trigger()

    def is_pinching(self) -> bool:
        return False   # eye mode không có continuous lift

    @staticmethod
    def _ear(lms, indices, w, h):
        """Eye Aspect Ratio từ 6 landmark (normalized)."""
        pts = [(lms[i].x, lms[i].y) for i in indices]
        # vertical distances
        A = math.hypot(pts[1][0]-pts[5][0], pts[1][1]-pts[5][1])
        B = math.hypot(pts[2][0]-pts[4][0], pts[2][1]-pts[4][1])
        # horizontal distance
        C = math.hypot(pts[0][0]-pts[3][0], pts[0][1]-pts[3][1])
        return (A + B) / (2.0 * C) if C > 0 else 1.0

    def _run(self):
        try:
            _ensure_model(FACE_MODEL_URL, FACE_MODEL_PATH)
        except Exception as e:
            print(f"[Eye] Không tải được model: {e}"); return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Eye] Không mở được webcam."); return

        base_opts = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5)

        with mp_vision.FaceLandmarker.create_from_options(opts) as lmk:
            while self._running:
                ok, frame = cap.read()
                if not ok: continue
                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]

                img    = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = lmk.detect(img)

                ear_val = 1.0
                blinking = False

                if result.face_landmarks:
                    lms = result.face_landmarks[0]

                    ear_l = self._ear(lms, self._LEFT_EYE,  w, h)
                    ear_r = self._ear(lms, self._RIGHT_EYE, w, h)
                    ear_val = (ear_l + ear_r) / 2.0

                    blinking = ear_val < EAR_THRESHOLD

                    # Vẽ điểm mắt
                    for idx in self._LEFT_EYE + self._RIGHT_EYE:
                        px = int(lms[idx].x * w)
                        py = int(lms[idx].y * h)
                        cv2.circle(frame, (px,py), 3, (0,220,255), -1)

                    color = (0,80,255) if blinking else (0,220,255)
                    label = f"EAR: {ear_val:.3f}  {'NHAP MAT! Bay len ^' if blinking else 'Mo mat  Roi xuong v'}"
                    cv2.putText(frame, label, (10,35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"Threshold: {EAR_THRESHOLD}",
                                (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                else:
                    cv2.putText(frame, "Khong thay khuon mat...",
                                (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

                # Đếm frame nhắm mắt liên tiếp → xác nhận chớp
                if blinking:
                    self._ear_counter += 1
                else:
                    if self._ear_counter >= EAR_CONSEC:
                        with self._lock:
                            self._blink_triggered = True
                    self._ear_counter = 0

                # EAR bar
                bx, by = 10, h-20
                bl = int(min(ear_val / 0.5, 1.0) * (w-20))
                cv2.rectangle(frame,(bx,by-14),(bx+w-20,by),(50,50,50),-1)
                bar_col = (0,80,255) if blinking else (0,220,255)
                if bl > 0:
                    cv2.rectangle(frame,(bx,by-14),(bx+bl,by),bar_col,-1)
                cv2.putText(frame,"EAR (cao=mo mat)",(bx,by-17),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)

                cv2.imshow("MAT - Flappy Bird", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._running = False; break

        cap.release(); cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# GAME CLASSES
# ─────────────────────────────────────────────────────────────────────────────
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.images = [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha(),
        ]
        self.speed = SPEED; self.current_image = 0
        self.image = self.images[0]
        self.mask  = pygame.mask.from_surface(self.image)
        self.rect  = self.image.get_rect()
        self.rect[0] = SCREEN_WIDHT / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY
        self.rect[1] += self.speed

    def bump(self):
        self.speed = -SPEED

    def lift(self):
        self.speed = min(self.speed, 6)
        self.speed = max(self.speed - 8, -SPEED * 0.7)

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]


class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDHT, PIPE_HEIGHT))
        self.rect  = self.image.get_rect()
        self.rect[0] = xpos
        if inverted:
            self.image   = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize
        self.mask = pygame.mask.from_surface(self.image)
        self.scored = False   # đã tính điểm chưa

    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDHT, GROUND_HEIGHT))
        self.mask  = pygame.mask.from_surface(self.image)
        self.rect  = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

def get_random_pipes(xpos):
    size = random.randint(100, 300)
    return Pipe(False, xpos, size), Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)

def do_bump(bird):
    bird.bump()
    pygame.mixer.music.load(wing)
    pygame.mixer.music.play()

def reset_game():
    bird_group   = pygame.sprite.Group()
    bird         = Bird()
    bird_group.add(bird)

    ground_group = pygame.sprite.Group()
    for i in range(2):
        ground_group.add(Ground(GROUND_WIDHT * i))

    pipe_group = pygame.sprite.Group()
    for i in range(2):
        p, pi = get_random_pipes(SCREEN_WIDHT * i + 800)
        pipe_group.add(p, pi)

    return bird, bird_group, ground_group, pipe_group


# ─────────────────────────────────────────────────────────────────────────────
# PYGAME INIT
# ─────────────────────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird - Gesture Control')

BACKGROUND  = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND  = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))
BEGIN_IMAGE = pygame.image.load('assets/sprites/message.png').convert_alpha()

clock = pygame.time.Clock()
pygame.font.init()

font_score    = pygame.font.SysFont("Arial", 46, bold=True)
font_score_sm = pygame.font.SysFont("Arial", 28, bold=True)
font_hud      = pygame.font.SysFont("Arial", 17, bold=True)
font_big      = pygame.font.SysFont("Arial", 42, bold=True)
font_med      = pygame.font.SysFont("Arial", 24)
font_mode     = pygame.font.SysFont("Arial", 28, bold=True)

# ─────────────────────────────────────────────────────────────────────────────
# DRAW HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def draw_score(score: int):
    surf = font_score.render(str(score), True, (255, 255, 255))
    shadow = font_score.render(str(score), True, (0, 0, 0))
    x = SCREEN_WIDHT // 2 - surf.get_width() // 2
    screen.blit(shadow, (x+2, 52))
    screen.blit(surf,   (x,   50))

def draw_hud(mode: str, pinching: bool):
    if mode == "hand":
        if pinching:
            color, label = (60,220,60),  "CHAP NGON  Bay len ^"
        else:
            color, label = (255,120,60), "MO TAY  Roi xuong v"
    else:
        color, label = (0,220,255), "CHOP MAT → Bay len"

    surf = font_hud.render(label, True, color)
    bg   = pygame.Surface((surf.get_width()+12, surf.get_height()+6), pygame.SRCALPHA)
    bg.fill((0,0,0,130))
    screen.blit(bg,   (4, 4))
    screen.blit(surf, (10, 7))

def draw_gameover(score: int):
    overlay = pygame.Surface((SCREEN_WIDHT, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    screen.blit(overlay, (0, 0))

    go   = font_big.render("GAME OVER", True, (255, 80, 80))
    sc   = font_med.render(f"Score: {score}", True, (255, 220, 80))
    hint = font_med.render("Chop / SPACE de choi lai", True, (200, 200, 200))

    screen.blit(go,   (SCREEN_WIDHT//2 - go.get_width()//2,   180))
    screen.blit(sc,   (SCREEN_WIDHT//2 - sc.get_width()//2,   240))
    screen.blit(hint, (SCREEN_WIDHT//2 - hint.get_width()//2, 280))


# ─────────────────────────────────────────────────────────────────────────────
# MODE SELECTION SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def draw_mode_select(hover: str):
    screen.blit(BACKGROUND, (0, 0))

    title = font_big.render("CHON CHE DO CHOI", True, (255, 255, 255))
    shadow = font_big.render("CHON CHE DO CHOI", True, (0,0,0))
    tx = SCREEN_WIDHT//2 - title.get_width()//2
    screen.blit(shadow, (tx+2, 102)); screen.blit(title, (tx, 100))

    sub = font_med.render("Nhan phim de chon:", True, (220, 220, 220))
    screen.blit(sub, (SCREEN_WIDHT//2 - sub.get_width()//2, 160))

    # Hand box
    hc = (80, 255, 120) if hover == "hand" else (200, 200, 200)
    hbox = pygame.Rect(40, 210, 140, 120)
    pygame.draw.rect(screen, (0,0,0,0), hbox, border_radius=14)
    pygame.draw.rect(screen, hc, hbox, 3, border_radius=14)
    hl1 = font_mode.render("[H]", True, hc)
    hl2 = font_med.render("Ngon tay", True, hc)
    hl3 = font_hud.render("Cham = Bay len", True, (180,180,180))
    hl4 = font_hud.render("Mo tay = Roi xuong", True, (180,180,180))
    screen.blit(hl1, (hbox.centerx - hl1.get_width()//2, hbox.y + 12))
    screen.blit(hl2, (hbox.centerx - hl2.get_width()//2, hbox.y + 46))
    screen.blit(hl3, (hbox.centerx - hl3.get_width()//2, hbox.y + 76))
    screen.blit(hl4, (hbox.centerx - hl4.get_width()//2, hbox.y + 96))

    # Eye box
    ec = (0, 200, 255) if hover == "eye" else (200, 200, 200)
    ebox = pygame.Rect(220, 210, 140, 120)
    pygame.draw.rect(screen, ec, ebox, 3, border_radius=14)
    el1 = font_mode.render("[E]", True, ec)
    el2 = font_med.render("Nhay mat", True, ec)
    el3 = font_hud.render("Chop = Bay len", True, (180,180,180))
    el4 = font_hud.render("Mo mat = Roi xuong", True, (180,180,180))
    screen.blit(el1, (ebox.centerx - el1.get_width()//2, ebox.y + 12))
    screen.blit(el2, (ebox.centerx - el2.get_width()//2, ebox.y + 46))
    screen.blit(el3, (ebox.centerx - el3.get_width()//2, ebox.y + 76))
    screen.blit(el4, (ebox.centerx - el4.get_width()//2, ebox.y + 96))

    hint = font_hud.render("H = Ngon tay    E = Nhay mat    SPACE = Ngon tay", True, (160,160,160))
    screen.blit(hint, (SCREEN_WIDHT//2 - hint.get_width()//2, 350))

    pygame.display.update()


def run_mode_select():
    """Hiện màn hình chọn chế độ, trả về 'hand' hoặc 'eye'."""
    hover = "hand"
    clock.tick(30)
    while True:
        clock.tick(30)
        mx, my = pygame.mouse.get_pos()
        hover = "hand" if mx < SCREEN_WIDHT//2 else "eye"

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); raise SystemExit
            if event.type == KEYDOWN:
                if event.key == K_h or event.key == K_SPACE:
                    return "hand"
                if event.key == K_e:
                    return "eye"
            if event.type == MOUSEBUTTONDOWN:
                if mx < SCREEN_WIDHT//2:
                    return "hand"
                else:
                    return "eye"

        draw_mode_select(hover)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROGRAM LOOP
# ─────────────────────────────────────────────────────────────────────────────
if not HAND_CONTROL_AVAILABLE:
    print("[WARNING] Chỉ dùng phím SPACE/UP.")

first_run = True

while True:  # ── outer restart loop ──────────────────────────────────────────

    # ── Chọn chế độ (chỉ hiện lần đầu hoặc sau game over nếu muốn đổi)
    game_mode = run_mode_select()
    print(f"[Mode] Đã chọn: {game_mode}")

    # ── Khởi tạo controller
    controller = None
    if HAND_CONTROL_AVAILABLE:
        if game_mode == "hand":
            controller = HandGestureController()
            print("[Hand] Chạm ngón cái + ngón trỏ → bay lên | Mở tay → rơi")
        else:
            controller = EyeBlinkController()
            print("[Eye]  Chớp mắt → bay lên (1 lần) | Mở mắt → rơi")
        controller.start()

    # ── Reset game entities
    bird, bird_group, ground_group, pipe_group = reset_game()
    score         = 0
    wing_cooldown = 0
    WING_CD       = 8

    # ── BEGIN SCREEN ─────────────────────────────────────────────────────────
    begin = True
    while begin:
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == QUIT:
                if controller: controller.stop()
                pygame.quit(); raise SystemExit
            if event.type == KEYDOWN:
                if event.key in (K_SPACE, K_UP):
                    do_bump(bird); begin = False

        if controller and controller.consume_trigger():
            do_bump(bird); begin = False

        screen.blit(BACKGROUND, (0,0))
        screen.blit(BEGIN_IMAGE, (120, 150))
        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            ground_group.add(Ground(GROUND_WIDHT - 20))

        bird.begin()
        ground_group.update()
        bird_group.draw(screen)
        ground_group.draw(screen)
        pinching = controller.is_pinching() if controller else False
        draw_hud(game_mode, pinching)
        pygame.display.update()

    # ── MAIN GAME LOOP ────────────────────────────────────────────────────────
    game_over = False
    while not game_over:
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == QUIT:
                if controller: controller.stop()
                pygame.quit(); raise SystemExit
            if event.type == KEYDOWN:
                if event.key in (K_SPACE, K_UP):
                    do_bump(bird)

        # ── Điều khiển theo chế độ
        pinching = False
        if controller:
            if game_mode == "hand":
                pinching = controller.is_pinching()
                if pinching:
                    bird.lift()
                    if wing_cooldown <= 0:
                        pygame.mixer.music.load(wing)
                        pygame.mixer.music.play()
                        wing_cooldown = WING_CD
                else:
                    wing_cooldown = max(wing_cooldown - 1, 0)
            else:  # eye mode
                if controller.consume_trigger():
                    do_bump(bird)

        screen.blit(BACKGROUND, (0,0))

        # ── Ground scroll
        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            ground_group.add(Ground(GROUND_WIDHT - 20))

        # ── Pipe scroll + score
        pipes = pipe_group.sprites()
        if pipes and is_off_screen(pipes[0]):
            pipe_group.remove(pipes[0])
            pipe_group.remove(pipes[1])
            p, pi = get_random_pipes(SCREEN_WIDHT * 2)
            pipe_group.add(p, pi)

        # ── Score: tính khi chim vượt qua mép phải của ống (chỉ pipe dưới)
        for pipe in pipe_group.sprites():
            if not pipe.scored:
                # chỉ đếm pipe dưới (không inverted) để không đếm 2 lần
                if pipe.rect[1] > 0 and pipe.rect[0] + PIPE_WIDHT < bird.rect[0]:
                    pipe.scored = True
                    score += 1

        bird_group.update()
        ground_group.update()
        pipe_group.update()

        bird_group.draw(screen)
        pipe_group.draw(screen)
        ground_group.draw(screen)

        draw_score(score)
        draw_hud(game_mode, pinching)
        pygame.display.update()

        # ── Collision
        if (pygame.sprite.groupcollide(bird_group, ground_group, False, False,
                                       pygame.sprite.collide_mask) or
            pygame.sprite.groupcollide(bird_group, pipe_group,   False, False,
                                       pygame.sprite.collide_mask)):
            game_over = True

    # ── GAME OVER SCREEN ──────────────────────────────────────────────────────
    pygame.mixer.music.load(hit)
    pygame.mixer.music.play()

    waiting = True
    while waiting:
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == QUIT:
                if controller: controller.stop()
                pygame.quit(); raise SystemExit
            if event.type == KEYDOWN:
                if event.key in (K_SPACE, K_UP):
                    waiting = False
        if controller and controller.consume_trigger():
            waiting = False

        draw_gameover(score)
        pygame.display.update()

    if controller:
        controller.stop()
        time.sleep(0.3)  # chờ thread dừng hẳn
