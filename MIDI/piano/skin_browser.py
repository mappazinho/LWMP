import pygame
import numpy as np
import os
from OpenGL.GL import *

from piano.pianoroll import PianoRoll
from piano.note_utils import RENDER_NOTE_DTYPE
from piano.skin_utils import _resolve_skin_dir, _list_available_skins, _SKIN_DIR


class SkinBrowser(PianoRoll):
    """A pygame+OpenGL window that lets users browse and preview skins
    using the exact same rendering pipeline as the live piano roll."""

    SIDEBAR_WIDTH = 220
    PREVIEW_SECONDS = 4.0

    def __init__(self, config, skin_root, current_skin_name):
        self._skin_root = skin_root
        self._current_skin_name = current_skin_name
        self._selected_skin_name = current_skin_name
        self._skin_list = _list_available_skins(skin_root)
        self._selected_index = 0
        if current_skin_name in self._skin_list:
            self._selected_index = self._skin_list.index(current_skin_name)
        self._sidebar_scroll = 0
        self._hover_index = -1
        self._applied = False

        preview_w = 1280
        preview_h = 720
        super().__init__(preview_w + self.SIDEBAR_WIDTH, preview_h, config)
        self.show_keyboard = True
        self.show_bloom = False
        self.show_glow = False
        self.export_mode = False

    def _active_skin_dir(self):
        return _resolve_skin_dir(self._selected_skin_name, self._skin_root)

    def init_pygame_and_gl(self, hidden=False):
        import piano.skin_utils as _su
        saved = _su._SKIN_DIR
        _su._SKIN_DIR = self._active_skin_dir() or saved
        try:
            super().init_pygame_and_gl(hidden=hidden)
        finally:
            _su._SKIN_DIR = saved
        pygame.display.set_caption("Skin Browser")
        self._sidebar_font = pygame.font.Font(None, 20)
        self._sidebar_font_bold = pygame.font.Font(None, 22)
        self._title_font = pygame.font.Font(None, 26)
        self._generate_preview_notes()

    def _generate_preview_notes(self):
        dt = RENDER_NOTE_DTYPE
        notes = []
        all_pitches = list(range(60, 84))
        note_idx = 0
        for pitch in all_pitches:
            if note_idx % 2 != 0:
                note_idx += 1
                continue
            n = np.zeros(1, dtype=dt)
            n['on_time'] = 0.0
            n['off_time'] = 2.0
            n['pitch'] = pitch
            n['velocity'] = 100
            n['track'] = note_idx % 16
            notes.append(n)
            note_idx += 1
        if notes:
            self.all_notes_gpu = np.concatenate(notes)
        else:
            self.all_notes_gpu = np.zeros(0, dtype=dt)
        self.notes_to_draw = len(self.all_notes_gpu)
        self.render_notes_array = self.all_notes_gpu
        self.render_on_times = self.all_notes_gpu['on_time'].astype(np.float64)
        durations = self.all_notes_gpu['off_time'] - self.all_notes_gpu['on_time']
        if len(durations) > 0:
            self.max_note_duration = float(np.max(durations))
        self._preview_time = 0.0
        self.get_current_time = lambda: self._preview_time

    def _reload_skin_textures(self):
        skin_dir = self._active_skin_dir()
        if skin_dir is None:
            return
        for tex_id in self.keyboard_textures.values():
            try:
                glDeleteTextures([tex_id])
            except Exception:
                pass
        self.keyboard_textures.clear()
        self.keyboard_texture_info.clear()
        if self.note_texture:
            try: glDeleteTextures([self.note_texture])
            except Exception: pass
        if self.note_edge_texture:
            try: glDeleteTextures([self.note_edge_texture])
            except Exception: pass
        self.note_texture = 0
        self.note_edge_texture = 0
        import piano.skin_utils as _su
        saved = _su._SKIN_DIR
        _su._SKIN_DIR = skin_dir
        try:
            try:
                self.note_texture, _, _ = self._load_texture(os.path.join(skin_dir, "note.png"))
                self.note_edge_texture, _, _ = self._load_texture(os.path.join(skin_dir, "noteEdge.png"))
                if self.note_texture is None: self.note_texture = 0
                if self.note_edge_texture is None: self.note_edge_texture = 0
            except Exception:
                self.note_texture = 0
                self.note_edge_texture = 0
            self.show_keyboard = True
            self._load_keyboard_assets()
        finally:
            _su._SKIN_DIR = saved

    def _blit_2d(self, text_surf, x, y):
        pixel_data = pygame.image.tostring(text_surf, "RGBA", True)
        w, h = text_surf.get_size()
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glWindowPos2i(int(x), int(self.height - y - h))
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixel_data)

    def _draw_sidebar(self):
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        sb_w = self.SIDEBAR_WIDTH
        bg = self.config.get('gui', {}).get('pianoroll_bg', [13, 13, 20])
        bg_f = tuple(c / 255.0 for c in bg)
        glColor3f(*bg_f)
        glBegin(GL_QUADS)
        glVertex2f(0, 0); glVertex2f(sb_w, 0)
        glVertex2f(sb_w, self.height); glVertex2f(0, self.height)
        glEnd()

        glColor3f(0.2, 0.2, 0.25)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glVertex2f(sb_w, 0); glVertex2f(sb_w, self.height)
        glEnd()

        title_surf = self._title_font.render("Skins", True, (223, 177, 103))
        self._blit_2d(title_surf, (sb_w - title_surf.get_width()) // 2, 12)

        y = 48
        item_h = 28
        visible_items = (self.height - 100) // item_h
        start_idx = max(0, self._selected_index - visible_items // 2)
        end_idx = min(len(self._skin_list), start_idx + visible_items)

        for i in range(start_idx, end_idx):
            name = self._skin_list[i]
            is_selected = (i == self._selected_index)
            is_hover = (i == self._hover_index)
            is_current = (name == self._current_skin_name)

            if is_selected:
                bg_col = (50, 55, 70)
            elif is_hover:
                bg_col = (35, 38, 50)
            else:
                bg_col = None

            if bg_col:
                glColor3f(bg_col[0]/255.0, bg_col[1]/255.0, bg_col[2]/255.0)
                glBegin(GL_QUADS)
                glVertex2f(4, y); glVertex2f(sb_w - 4, y)
                glVertex2f(sb_w - 4, y + item_h); glVertex2f(4, y + item_h)
                glEnd()

            label = name + ("  [active]" if is_current else "")
            text_col = (220, 220, 230) if is_selected else (170, 175, 190)
            text_surf = self._sidebar_font.render(label, True, text_col)
            self._blit_2d(text_surf, 14, y + (item_h - text_surf.get_height()) // 2)
            y += item_h

        btn_y = self.height - 44
        btn_w, btn_h = 100, 30
        btn_x = (sb_w - btn_w) // 2
        btn_bg = (50, 120, 80) if self._hover_index == -2 else (40, 100, 65)
        glColor3f(btn_bg[0]/255.0, btn_bg[1]/255.0, btn_bg[2]/255.0)
        glBegin(GL_QUADS)
        glVertex2f(btn_x, btn_y); glVertex2f(btn_x + btn_w, btn_y)
        glVertex2f(btn_x + btn_w, btn_y + btn_h); glVertex2f(btn_x, btn_y + btn_h)
        glEnd()
        btn_text = self._sidebar_font_bold.render("Apply Skin", True, (230, 230, 230))
        self._blit_2d(btn_text, btn_x + (btn_w - btn_text.get_width()) // 2,
                       btn_y + (btn_h - btn_text.get_height()) // 2)
        self._apply_btn_rect = (btn_x, btn_y, btn_w, btn_h)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

    def draw(self, current_time, present=True):
        sb_w = self.SIDEBAR_WIDTH
        preview_w = self.width - sb_w
        glViewport(sb_w, 0, preview_w, self.height)
        old_w = self.width
        self.width = preview_w

        self.last_frame_time = current_time
        self._init_slider_geometry()
        if self.show_glow and (self.show_key_press_glow or self.show_key_light_fade):
            self._update_glow_trails(current_time)
            self._update_glow_cull()
        else:
            self.glow_trails.clear()
            self._split_fade_trails.clear()

        if self.all_notes_gpu is not None and len(self.all_notes_gpu) > 0:
            self.notes_to_draw = len(self.all_notes_gpu)
            self.last_visible_notes = self.all_notes_gpu
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_stream_data)
            glBufferSubData(GL_ARRAY_BUFFER, 0, self.all_notes_gpu.nbytes, self.all_notes_gpu)

        self._render_scene_content(current_time)
        self.width = old_w
        glViewport(0, 0, self.width, self.height)
        self._draw_sidebar()
        if present:
            pygame.display.flip()

    def _handle_click(self, pos):
        x, y = pos
        sb_w = self.SIDEBAR_WIDTH
        if x >= sb_w:
            return
        item_h = 28
        start_y = 48
        visible_items = (self.height - 100) // item_h
        start_idx = max(0, self._selected_index - visible_items // 2)
        end_idx = min(len(self._skin_list), start_idx + visible_items)
        for i in range(start_idx, end_idx):
            item_y = start_y + (i - start_idx) * item_h
            if item_y <= y <= item_y + item_h:
                self._selected_index = i
                self._selected_skin_name = self._skin_list[i]
                self._reload_skin_textures()
                return
        if hasattr(self, '_apply_btn_rect'):
            bx, by, bw, bh = self._apply_btn_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._applied = True
                self.app_running.clear()

    def _handle_hover(self, pos):
        x, y = pos
        if x >= self.SIDEBAR_WIDTH:
            self._hover_index = -1
            return
        item_h = 28
        start_y = 48
        visible_items = (self.height - 100) // item_h
        start_idx = max(0, self._selected_index - visible_items // 2)
        end_idx = min(len(self._skin_list), start_idx + visible_items)
        self._hover_index = -1
        for i in range(start_idx, end_idx):
            item_y = start_y + (i - start_idx) * item_h
            if item_y <= y <= item_y + item_h:
                self._hover_index = i
                return
        if hasattr(self, '_apply_btn_rect'):
            bx, by, bw, bh = self._apply_btn_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._hover_index = -2

    def run_browser(self):
        """Run the skin browser modally. Returns the applied skin name or None."""
        self.init_pygame_and_gl()
        clock = pygame.time.Clock()
        self._apply_btn_rect = (0, 0, 0, 0)

        while self.app_running.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.app_running.clear()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_hover(event.pos)
                elif event.type == pygame.MOUSEWHEEL:
                    if len(self._skin_list) > 0:
                        self._selected_index = max(0, min(len(self._skin_list) - 1,
                                                          self._selected_index - event.y))
                        self._selected_skin_name = self._skin_list[self._selected_index]
                        self._reload_skin_textures()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and self._selected_index > 0:
                        self._selected_index -= 1
                        self._selected_skin_name = self._skin_list[self._selected_index]
                        self._reload_skin_textures()
                    elif event.key == pygame.K_DOWN and self._selected_index < len(self._skin_list) - 1:
                        self._selected_index += 1
                        self._selected_skin_name = self._skin_list[self._selected_index]
                        self._reload_skin_textures()
                    elif event.key == pygame.K_RETURN:
                        self._applied = True
                        self.app_running.clear()
                    elif event.key == pygame.K_ESCAPE:
                        self.app_running.clear()

            self.draw(self._preview_time)
            caption = f"Skin Browser - {self._selected_skin_name}"
            if self._selected_skin_name == self._current_skin_name:
                caption += " [active]"
            pygame.display.set_caption(caption)
            clock.tick(60)

        applied_name = self._selected_skin_name if self._applied else None
        self.cleanup()
        return applied_name
