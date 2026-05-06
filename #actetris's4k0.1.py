import pygame
import random
import sys
import numpy as np

# ====================== CONFIG ======================
WIDTH, HEIGHT = 640, 480
BLOCK_SIZE = 24
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BOARD_X = (WIDTH - BOARD_WIDTH * BLOCK_SIZE) // 2
BOARD_Y = (HEIGHT - BOARD_HEIGHT * BLOCK_SIZE) // 2 - 20

# One game tick per display frame. NES NTSC is ~60.0988 Hz; gravity is stored as
# whole frames per cell for Nintendo Tetris NTSC — running at 60 Hz matches that curve.
FPS = 60
TITLE = "AC'S Tetris 0.1 by ac. holding [c] 1999-2026 [c] 1999-2026 Nintendo"

# Colors (NES/Famicom style)
BLACK = (0, 0, 0)
GRAY = (40, 40, 40)
WHITE = (255, 255, 255)
COLORS = [
    (0, 0, 0),      # 0 empty
    (0, 255, 255),  # I cyan
    (255, 165, 0),  # O orange? Wait classic:
    (0, 0, 255),    # J blue
    (255, 165, 0),  # L orange
    (0, 255, 0),    # S green
    (255, 0, 0),    # Z red
    (255, 255, 0),  # T yellow
]

# Tetromino shapes
SHAPES = [
    [[1, 1, 1, 1]],           # I
    [[1, 1], [1, 1]],         # O
    [[0, 1, 0], [1, 1, 1]],   # T
    [[1, 0, 0], [1, 1, 1]],   # J
    [[0, 0, 1], [1, 1, 1]],   # L
    [[0, 1, 1], [1, 1, 0]],   # S
    [[1, 1, 0], [0, 1, 1]]    # Z
]

# NES Tetris NTSC: frames per grid cell (levels 0–28), then 29+ = 1 (TetrisWiki / $898E).
GRAVITY = [48, 43, 38, 33, 28, 23, 18, 13, 8, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]

# ====================== SOUND ======================
SAMPLE_RATE = 44100


def generate_square_wave(freq, duration, volume=0.22):
    """Short SFX: band-limited-ish square (50% duty), mono int16 for pygame-ce."""
    n = max(1, int(SAMPLE_RATE * duration))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    phase = (t * freq) % 1.0
    sq = np.where(phase < 0.5, 1.0, -1.0)
    wave = np.clip(sq * float(volume) * 12000.0, -32768, 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(wave.ravel()))


def _note_square(freq_hz, duration_sec, duty=0.5, peak=0.16):
    """One square-wave note with tiny fades to avoid clicks (Korobeiniki chiptune)."""
    n = max(1, int(SAMPLE_RATE * duration_sec))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    if freq_hz <= 0:
        return np.zeros(n, dtype=np.float64)
    phase = (t * freq_hz) % 1.0
    sq = np.where(phase < duty, 1.0, -1.0).astype(np.float64)
    fade = min(64, max(1, n // 8))
    env = np.ones(n, dtype=np.float64)
    env[:fade] *= np.linspace(0.0, 1.0, fade, endpoint=False)
    env[-fade:] *= np.linspace(1.0, 0.0, fade, endpoint=False)
    return sq * env * peak


def build_tetris_korobeiniki_loop():
    """
    Type A (Korobeiniki) — long in-game loop: section A (verified 40-note opening),
    bass echo, section B (second strain, same rhythm), recap A, then rest.
    """
    # Section A — first ~8 bars (Hz + ms per step, 40 notes; published Processing sketch).
    k_hz = [
        659.25511, 493.8833, 523.25113, 587.32954, 523.25113, 493.8833, 440.0, 440.0, 523.25113,
        659.25511, 587.32954, 523.25113, 493.8833, 523.25113, 587.32954, 659.25511, 523.25113,
        440.0, 440.0, 440.0, 493.8833, 523.25113, 587.32954, 698.45646, 880.0, 783.99087,
        698.45646, 659.25511, 523.25113, 659.25511, 587.32954, 523.25113, 493.8833, 493.8833,
        523.25113, 587.32954, 659.25511, 523.25113, 440.0, 440.0,
    ]
    k_ms = [
        406.250, 203.125, 203.125, 406.250, 203.125, 203.125, 406.250, 203.125, 203.125, 406.250,
        203.125, 203.125, 609.375, 203.125, 406.250, 406.250, 406.250, 406.250, 203.125, 203.125,
        203.125, 203.125, 609.375, 203.125, 406.250, 203.125, 203.125, 609.375, 203.125, 406.250,
        203.125, 203.125, 406.250, 203.125, 203.125, 406.250, 406.250, 406.250, 406.250, 406.250,
    ]
    # Section B — Korobeiniki continuation (E minor, monophonic; 40 notes, reuse A rhythm).
    b_hz = [
        392.0, 392.0, 523.25113, 659.25511, 587.32954, 523.25113, 493.8833, 440.0,
        440.0, 493.8833, 523.25113, 587.32954, 523.25113, 493.8833, 440.0, 392.0,
        369.99343, 392.0, 440.0, 493.8833, 523.25113, 659.25511, 587.32954, 587.32954,
        659.25511, 739.98885, 659.25511, 587.32954, 493.8833, 523.25113, 523.25113, 659.25511,
        587.32954, 523.25113, 493.8833, 440.0, 523.25113, 493.8833, 440.0, 392.0,
    ]

    def _silence(sec):
        n = max(1, int(SAMPLE_RATE * sec))
        return np.zeros(n, dtype=np.float64)

    def _phrase(freqs, mss, octave_shift=0):
        fac = 2.0 ** octave_shift
        out = []
        for hz, ms in zip(freqs, mss):
            f = 0.0 if hz <= 0 else float(hz) * fac
            out.append(_note_square(f, float(ms) / 1000.0))
        return out

    gap = 0.22
    parts = (
        _phrase(k_hz, k_ms, 0)
        + [_silence(gap)]
        + _phrase(k_hz, k_ms, -1)
        + [_silence(gap)]
        + _phrase(b_hz, k_ms, 0)
        + [_silence(gap)]
        + _phrase(k_hz, k_ms, 0)
        + [_silence(0.45)]
    )
    full = np.concatenate(parts)
    wave = np.clip(full * 32767.0, -32768, 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(wave.ravel()))


pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, 512)
pygame.init()
pygame.mixer.set_num_channels(12)
THEME_CHANNEL = pygame.mixer.Channel(0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(TITLE)
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)
bigfont = pygame.font.SysFont("consolas", 36)

# Sounds
snd_rotate = generate_square_wave(800, 0.05)
snd_lock = generate_square_wave(200, 0.1)
snd_clear = generate_square_wave(1200, 0.15)
snd_levelup = generate_square_wave(600, 0.4)
snd_gameover = generate_square_wave(150, 1.0)

snd_tetris_theme = build_tetris_korobeiniki_loop()
music_playing = False

# Menu / settings (sound screen toggles these)
settings_music = True
settings_sfx = True

MENU_ITEMS = (
    ("Play Game", "play"),
    ("Credits", "credits"),
    ("About", "about"),
    ("Sound Settings", "sound"),
    ("Help", "help"),
    ("Exit", "exit"),
)


def sfx(sound):
    if settings_sfx:
        sound.play()


def update_theme():
    """Looping Korobeiniki-style theme on a dedicated channel (no per-frame beep spam)."""
    if not music_playing:
        if THEME_CHANNEL.get_busy():
            THEME_CHANNEL.stop()
        return
    if not THEME_CHANNEL.get_busy():
        THEME_CHANNEL.play(snd_tetris_theme, loops=-1)

# ====================== GAME ======================
class Tetris:
    def __init__(self):
        self.board = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.reset_game()

    def reset_game(self):
        self.board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self.score = 0
        self.lines = 0
        self.level = 0
        self.next_piece = random.randint(0, 6)
        self.new_piece()
        self.game_over = False
        self.paused = False

    def new_piece(self):
        self.piece = self.next_piece
        self.next_piece = random.randint(0, 6)
        self.rotation = 0
        self.x = BOARD_WIDTH // 2 - len(SHAPES[self.piece][0]) // 2
        self.y = 0
        if self.check_collision():
            self.game_over = True

    def check_collision(self, dx=0, dy=0, rot=None):
        if rot is None:
            rot = self.rotation
        shape = self.get_shape(rot)
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    bx = self.x + x + dx
                    by = self.y + y + dy
                    if bx < 0 or bx >= BOARD_WIDTH or by >= BOARD_HEIGHT:
                        return True
                    if by >= 0 and self.board[by][bx]:
                        return True
        return False

    def get_shape(self, rot=None):
        if rot is None:
            rot = self.rotation
        shape = [list(row) for row in SHAPES[self.piece]]
        for _ in range(rot % 4):
            shape = [list(reversed(col)) for col in zip(*shape)]
        return shape

    def lock_piece(self):
        shape = self.get_shape()
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self.board[self.y + y][self.x + x] = self.piece + 1
        sfx(snd_lock)
        self.clear_lines()
        self.new_piece()

    def clear_lines(self):
        lines_cleared = 0
        y = BOARD_HEIGHT - 1
        while y >= 0:
            if all(self.board[y]):
                del self.board[y]
                self.board.insert(0, [0] * BOARD_WIDTH)
                lines_cleared += 1
            else:
                y -= 1
        if lines_cleared:
            sfx(snd_clear)
            self.lines += lines_cleared
            idx = min(lines_cleared, 4)
            self.score += [0, 100, 300, 500, 800][idx] * (self.level + 1)
            self.level = self.lines // 10

    def rotate(self):
        if not self.check_collision(rot=(self.rotation + 1) % 4):
            self.rotation = (self.rotation + 1) % 4
            sfx(snd_rotate)

    def move(self, dx):
        if not self.check_collision(dx=dx):
            self.x += dx

    def drop(self):
        if not self.check_collision(dy=1):
            self.y += 1
            return True
        return False

    def hard_drop(self):
        while self.drop():
            self.score += 2
        self.lock_piece()

    def draw_board(self, surface):
        # Background
        pygame.draw.rect(surface, GRAY, (BOARD_X - 4, BOARD_Y - 4, BOARD_WIDTH * BLOCK_SIZE + 8, BOARD_HEIGHT * BLOCK_SIZE + 8), 0)
        pygame.draw.rect(surface, BLACK, (BOARD_X, BOARD_Y, BOARD_WIDTH * BLOCK_SIZE, BOARD_HEIGHT * BLOCK_SIZE), 0)

        # Board blocks
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.board[y][x]:
                    color = COLORS[self.board[y][x]]
                    rect = pygame.Rect(BOARD_X + x * BLOCK_SIZE, BOARD_Y + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                    pygame.draw.rect(surface, color, rect)
                    pygame.draw.rect(surface, WHITE, rect, 1)

        # Ghost: max k steps down from (self.x, self.y) with same rotation
        ghost_drop = 0
        while not self.check_collision(dy=ghost_drop + 1):
            ghost_drop += 1
        ghost_y = self.y + ghost_drop
        shape = self.get_shape()
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    gx = BOARD_X + (self.x + x) * BLOCK_SIZE
                    gy = BOARD_Y + (ghost_y + y) * BLOCK_SIZE
                    pygame.draw.rect(surface, (60, 60, 60), (gx, gy, BLOCK_SIZE, BLOCK_SIZE), 2)

        # Current piece
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    color = COLORS[self.piece + 1]
                    rect = pygame.Rect(BOARD_X + (self.x + x) * BLOCK_SIZE, BOARD_Y + (self.y + y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                    pygame.draw.rect(surface, color, rect)
                    pygame.draw.rect(surface, WHITE, rect, 2)

    def draw_ui(self, surface):
        # Next piece
        next_shape = SHAPES[self.next_piece]
        nx = WIDTH - 120
        ny = 80
        for y, row in enumerate(next_shape):
            for x, cell in enumerate(row):
                if cell:
                    color = COLORS[self.next_piece + 1]
                    pygame.draw.rect(surface, color, (nx + x * BLOCK_SIZE, ny + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(surface, WHITE, (nx + x * BLOCK_SIZE, ny + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

        # Stats
        texts = [
            f"SCORE {self.score:07d}",
            f"LINES {self.lines:04d}",
            f"LEVEL {self.level:02d}"
        ]
        for i, txt in enumerate(texts):
            surf = font.render(txt, True, WHITE)
            surface.blit(surf, (20, 60 + i * 30))

        # Title
        title = bigfont.render("AC'S TETRIS", True, (255, 215, 0))
        surface.blit(title, (WIDTH//2 - title.get_width()//2, 15))


def sync_music(state):
    """Korobeiniki theme only during active gameplay (never on main menu or sub-screens)."""
    global music_playing
    if not settings_music:
        music_playing = False
    elif state == "PLAYING":
        music_playing = True
    else:
        music_playing = False


def draw_text_screen(surface, title, lines, footer=None):
    surface.fill(BLACK)
    t = bigfont.render(title, True, (255, 215, 0))
    surface.blit(t, (WIDTH // 2 - t.get_width() // 2, 40))
    y = 100
    for line in lines:
        s = font.render(line, True, WHITE)
        surface.blit(s, (WIDTH // 2 - s.get_width() // 2, y))
        y += 26
    if footer:
        f = font.render(footer, True, GRAY)
        surface.blit(f, (WIDTH // 2 - f.get_width() // 2, HEIGHT - 50))


def draw_main_menu(surface, menu_sel):
    surface.fill(BLACK)
    t = bigfont.render("AC's Tetris", True, (255, 215, 0))
    surface.blit(t, (WIDTH // 2 - t.get_width() // 2, 36))
    st = font.render("by ac holding", True, GRAY)
    surface.blit(st, (WIDTH // 2 - st.get_width() // 2, 78))
    y0 = 130
    for i, (label, _) in enumerate(MENU_ITEMS):
        col = (255, 215, 0) if i == menu_sel else WHITE
        prefix = "> " if i == menu_sel else "  "
        row = font.render(prefix + label, True, col)
        surface.blit(row, (WIDTH // 2 - 110, y0 + i * 32))
    hint = font.render("Up/Down  Enter or Space  Esc: Quit", True, GRAY)
    surface.blit(hint, (WIDTH // 2 - hint.get_width() // 2, HEIGHT - 44))


def draw_sound_menu(surface, sound_sel):
    surface.fill(BLACK)
    t = bigfont.render("Sound Settings", True, (255, 215, 0))
    surface.blit(t, (WIDTH // 2 - t.get_width() // 2, 50))
    opts = [
        ("Background music", "ON" if settings_music else "OFF"),
        ("Sound effects", "ON" if settings_sfx else "OFF"),
    ]
    y = 140
    for i, (lab, val) in enumerate(opts):
        col = (255, 215, 0) if i == sound_sel else WHITE
        line = font.render(("> " if i == sound_sel else "  ") + f"{lab}:  {val}", True, col)
        surface.blit(line, (WIDTH // 2 - 180, y + i * 40))
    foot = font.render("Left/Right or Enter: toggle   Esc: back", True, GRAY)
    surface.blit(foot, (WIDTH // 2 - foot.get_width() // 2, HEIGHT - 50))


# ====================== MAIN LOOP ======================
def main():
    global music_playing, settings_music, settings_sfx
    game = Tetris()
    drop_timer = 0
    running = True
    # MENU, CREDITS, ABOUT, SOUND_SETTINGS, HELP, PLAYING, PAUSED, GAMEOVER
    state = "MENU"
    menu_sel = 0
    sound_sel = 0

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type != pygame.KEYDOWN:
                continue

            if state == "MENU":
                if event.key in (pygame.K_UP, pygame.K_w):
                    menu_sel = (menu_sel - 1) % len(MENU_ITEMS)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    menu_sel = (menu_sel + 1) % len(MENU_ITEMS)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    act = MENU_ITEMS[menu_sel][1]
                    if act == "play":
                        game.reset_game()
                        state = "PLAYING"
                        drop_timer = 0
                    elif act == "credits":
                        state = "CREDITS"
                    elif act == "about":
                        state = "ABOUT"
                    elif act == "sound":
                        state = "SOUND_SETTINGS"
                        sound_sel = 0
                    elif act == "help":
                        state = "HELP"
                    elif act == "exit":
                        running = False
                elif event.key == pygame.K_ESCAPE:
                    running = False

            elif state == "CREDITS":
                if event.key == pygame.K_ESCAPE:
                    state = "MENU"

            elif state == "ABOUT":
                if event.key == pygame.K_ESCAPE:
                    state = "MENU"

            elif state == "HELP":
                if event.key == pygame.K_ESCAPE:
                    state = "MENU"

            elif state == "SOUND_SETTINGS":
                if event.key == pygame.K_ESCAPE:
                    state = "MENU"
                elif event.key in (pygame.K_UP, pygame.K_w):
                    sound_sel = (sound_sel - 1) % 2
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    sound_sel = (sound_sel + 1) % 2
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    if sound_sel == 0:
                        settings_music = not settings_music
                    else:
                        settings_sfx = not settings_sfx

            elif state == "PLAYING":
                if event.key == pygame.K_ESCAPE:
                    state = "PAUSED"
                elif event.key == pygame.K_LEFT:
                    game.move(-1)
                elif event.key == pygame.K_RIGHT:
                    game.move(1)
                elif event.key in (pygame.K_UP, pygame.K_z, pygame.K_x):
                    game.rotate()
                elif event.key == pygame.K_DOWN:
                    if game.drop():
                        game.score += 1
                elif event.key == pygame.K_SPACE:
                    game.hard_drop()
                elif event.key == pygame.K_p:
                    state = "PAUSED"
                elif event.key == pygame.K_r:
                    game.reset_game()
                    drop_timer = 0

            elif state == "PAUSED":
                if event.key == pygame.K_ESCAPE:
                    game.reset_game()
                    state = "MENU"
                elif event.key == pygame.K_p:
                    state = "PLAYING"

            elif state == "GAMEOVER":
                if event.key == pygame.K_r:
                    game.reset_game()
                    state = "PLAYING"
                    drop_timer = 0
                elif event.key == pygame.K_ESCAPE:
                    game.reset_game()
                    state = "MENU"

        sync_music(state)

        if state == "PLAYING":
            drop_speed = GRAVITY[min(game.level, len(GRAVITY) - 1)]
            drop_timer += 1
            if drop_timer >= drop_speed:
                if not game.drop():
                    game.lock_piece()
                drop_timer = 0

            if game.game_over:
                state = "GAMEOVER"
                sfx(snd_gameover)

        update_theme()

        if state in ("PLAYING", "PAUSED", "GAMEOVER"):
            screen.fill(BLACK)
            game.draw_board(screen)
            game.draw_ui(screen)
            if state == "PAUSED":
                txt = bigfont.render("PAUSED", True, WHITE)
                screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, HEIGHT // 2 - 20))
                h = font.render("P: resume   Esc: main menu", True, GRAY)
                screen.blit(h, (WIDTH // 2 - h.get_width() // 2, HEIGHT // 2 + 28))
            elif state == "GAMEOVER":
                txt = bigfont.render("GAME OVER", True, (255, 50, 50))
                screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, HEIGHT // 2 - 50))
                txt2 = font.render("R: play again   Esc: main menu", True, WHITE)
                screen.blit(txt2, (WIDTH // 2 - txt2.get_width() // 2, HEIGHT // 2 + 10))
        elif state == "MENU":
            draw_main_menu(screen, menu_sel)
        elif state == "CREDITS":
            draw_text_screen(
                screen,
                "Credits",
                [
                    "AC's Tetris — Famicom Edition",
                    "",
                    "Programming & chiptune-style audio",
                    "built with Python and pygame.",
                    "",
                    "Tetris is a trademark of Tetris Holding.",
                    "This is an unofficial fan-style clone.",
                ],
                "Esc: back to main menu",
            )
        elif state == "ABOUT":
            draw_text_screen(
                screen,
                "About",
                [
                    "A compact falling-block puzzle inspired",
                    "by classic NES / Famicom Tetris.",
                    "",
                    "Gravity table and colors nod to the",
                    "8-bit console feel; gameplay is standard",
                    "SRS-free vanilla lock and line clears.",
                ],
                "Esc: back to main menu",
            )
        elif state == "HELP":
            draw_text_screen(
                screen,
                "Help",
                [
                    "Left / Right — move piece",
                    "Up, Z, or X — rotate",
                    "Down — soft drop (+1 score per step)",
                    "Space — hard drop (locks, +2 per cell)",
                    "P — pause   Esc — pause (then Esc: main menu)",
                    "",
                    "Game over: R new game, Esc main menu.",
                ],
                "Esc: back to main menu",
            )
        elif state == "SOUND_SETTINGS":
            draw_sound_menu(screen, sound_sel)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()