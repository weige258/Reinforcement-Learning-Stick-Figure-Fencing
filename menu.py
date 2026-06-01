"""
主菜单界面 - 支持鼠标和键盘
"""
import os
import sys
import pygame
import game_config as cfg

CHINESE_FONTS = [
    "SourceHanSansSC-Regular.otf",     # 项目自带字体(优先)
    "C:/Windows/Fonts/msyh.ttc",       # Windows
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
    "/System/Library/Fonts/PingFang.ttc",            # macOS
]

def _get_font(size):
    for p in CHINESE_FONTS:
        if os.path.exists(p):
            try: return pygame.font.Font(p, size)
            except: continue
    return pygame.font.Font(None, size)


class Menu:
    def __init__(self, screen):
        self.screen = screen
        self.font_title = _get_font(52)
        self.font_item = _get_font(36)
        self.font_desc = _get_font(22)
        self.font_footer = _get_font(18)
        self.menu_items = [
            ("play",  "人机对战", "WASD移动 | 鼠标左键挥剑 | 右键举盾"),
            ("train", "训练AI",   "自我对抗强化学习训练"),
            ("exit",  "退出",     "退出游戏"),
        ]
        self.selected = 0
        self.clock = pygame.time.Clock()
        self._bg_cache = None

    def _build_bg(self):
        if self._bg_cache:
            return
        self._bg_cache = pygame.Surface((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
        for i in range(cfg.GROUND_Y):
            r = cfg.SKY_COLOR[0] + i * 10 // cfg.GROUND_Y
            g = cfg.SKY_COLOR[1] + i * 5 // cfg.GROUND_Y
            b = cfg.SKY_COLOR[2] + i * 8 // cfg.GROUND_Y
            pygame.draw.line(self._bg_cache, (r, g, b), (0, i), (cfg.WINDOW_WIDTH, i))
        pygame.draw.rect(self._bg_cache, cfg.GROUND_COLOR,
                        (0, cfg.GROUND_Y, cfg.WINDOW_WIDTH, cfg.GROUND_THICKNESS))
        pygame.draw.line(self._bg_cache, (100, 90, 80),
                        (0, cfg.GROUND_Y), (cfg.WINDOW_WIDTH, cfg.GROUND_Y), 2)

    def _draw_title(self):
        t = self.font_title.render("火柴人击剑格斗", True, (255, 255, 255))
        self.screen.blit(t, t.get_rect(center=(cfg.WINDOW_WIDTH//2, 70)))
        sub = self.font_footer.render("Bloody Bastards 风格 | PyMunk物理 | PyTorch DQN", True, (170, 170, 180))
        self.screen.blit(sub, sub.get_rect(center=(cfg.WINDOW_WIDTH//2, 105)))
        pygame.draw.line(self.screen, (100, 100, 110),
                        (cfg.WINDOW_WIDTH//2 - 180, 120),
                        (cfg.WINDOW_WIDTH//2 + 180, 120), 1)

    def _draw_menu(self):
        start_y, item_h, cx = 190, 70, cfg.WINDOW_WIDTH // 2
        for i, (key, label, desc) in enumerate(self.menu_items):
            y = start_y + i * item_h
            sel = (i == self.selected)
            rect = pygame.Rect(cx - 160, y - 6, 320, item_h - 8)
            if sel:
                pygame.draw.rect(self.screen, (80, 120, 180), rect, border_radius=6)
                pygame.draw.rect(self.screen, (100, 150, 210), rect, 2, border_radius=6)
                tc, dc = (255, 255, 255), (200, 220, 255)
            else:
                tc, dc = (170, 170, 180), (120, 120, 130)
            ls = self.font_item.render(label, True, tc)
            self.screen.blit(ls, ls.get_rect(center=(cx, y + 2)))
            ds = self.font_desc.render(desc, True, dc)
            self.screen.blit(ds, ds.get_rect(center=(cx, y + 28)))

    def _draw_footer(self):
        txt = self.font_footer.render("鼠标点击选择 | 方向键 + Enter 确认 | ESC 退出", True, (120, 120, 130))
        self.screen.blit(txt, txt.get_rect(center=(cfg.WINDOW_WIDTH//2, cfg.WINDOW_HEIGHT - 35)))

    def _item_rect(self, idx):
        y = 190 + idx * 70
        return pygame.Rect(cfg.WINDOW_WIDTH//2 - 160, y - 6, 320, 62)

    def run(self):
        self._build_bg()
        selected_mode = None
        while selected_mode is None:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "exit"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.selected = (self.selected - 1) % len(self.menu_items)
                    elif event.key == pygame.K_DOWN:
                        self.selected = (self.selected + 1) % len(self.menu_items)
                    elif event.key == pygame.K_RETURN:
                        selected_mode = self.menu_items[self.selected][0]
                    elif event.key == pygame.K_ESCAPE:
                        return "exit"
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for i in range(len(self.menu_items)):
                        if self._item_rect(i).collidepoint(mouse_pos):
                            selected_mode = self.menu_items[i][0]
                            break
                elif event.type == pygame.MOUSEMOTION:
                    for i in range(len(self.menu_items)):
                        if self._item_rect(i).collidepoint(mouse_pos):
                            self.selected = i
                            break
            self.screen.blit(self._bg_cache, (0, 0))
            self._draw_title()
            self._draw_menu()
            self._draw_footer()
            pygame.display.flip()
            self.clock.tick(cfg.FPS)
        return selected_mode
