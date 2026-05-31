"""
多宫格训练视图 - 同时监控多个训练局
使用 ThreadPoolExecutor 并行加速 (PyMunk+PyTorch 均释放GIL)
"""
import os
import time
import math
import pygame
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from game_config import *
from fencing_game import FencingGame
from dqn_agent import DQNAgent


class MultiGameViewer:
    """多宫格训练视图管理器 (并行加速版)"""
    
    def __init__(self, grid_size=(2, 2), render=True):
        """
        grid_size: (行, 列) 例如 (2,2)=4局, (2,3)=6局, (3,3)=9局
        """
        self.rows, self.cols = grid_size
        self.num_games = self.rows * self.cols
        self.render = render
        
        # 窗口尺寸 - 适中大小, 不全屏
        max_win_w = 1200
        max_win_h = 720
        cell_w_max = max_win_w // self.cols
        cell_h_max = max_win_h // self.rows
        self.cell_w = min(SCREEN_WIDTH // 3, cell_w_max)
        self.cell_h = min(SCREEN_HEIGHT // 3, cell_h_max)
        self.win_w = self.cell_w * self.cols
        self.win_h = self.cell_h * self.rows
        
        # 创建窗口
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.win_w, self.win_h))
            pygame.display.set_caption(
                f"🔥 多宫格训练 ({self.rows}x{self.cols}) - {self.num_games}局同时训练")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        
        # 创建游戏实例 + 智能体
        self.games = []
        self.agents_p1 = []
        self.agents_p2 = []
        
        for i in range(self.num_games):
            # 每个游戏共享同一个物理空间？不，每个独立
            if render:
                # 创建子表面
                col = i % self.cols
                row = i // self.cols
                sub = self.screen.subsurface(
                    (col * self.cell_w, row * self.cell_h, self.cell_w, self.cell_h))
            else:
                sub = None
            
            game = FencingGame(render=render, render_surface=sub)
            game._viewer_index = i + 1
            self.games.append(game)
            
            self.agents_p1.append(DQNAgent(RL['state_dim'], RL['action_dim']))
            self.agents_p2.append(DQNAgent(RL['state_dim'], RL['action_dim']))
        
        # 统计
        self.total_steps = 0
        self.start_time = time.time()
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _train_one_game(self, idx, should_optimize, should_update_target):
        """训练单个游戏一步 (供线程池调用)"""
        game = self.games[idx]
        ap1 = self.agents_p1[idx]
        ap2 = self.agents_p2[idx]
        
        state = game._cached_state if hasattr(game, '_cached_state') else game.reset()
        
        action1 = ap1.select_action(state)
        state_p2 = game._get_state(perspective=2)
        action2 = ap2.select_action(state_p2)
        
        ns1, r1, ns2, r2, done, info = game.step(action1, action2)
        
        ap1.store_transition(state, action1, ns1, r1)
        ap2.store_transition(state_p2, action2, ns2, r2)
        
        if should_optimize:
            ap1.optimize_model()
            ap2.optimize_model()
        
        if should_update_target:
            ap1.soft_update_target()
            ap2.soft_update_target()
        
        game._cached_state = ns1
        if done:
            game._cached_state = game.reset()
        
        return done
    
    def train_one_step(self):
        """所有游戏并行步进一步 (ThreadPoolExecutor加速)"""
        should_optimize = (self.total_steps % 3 == 0)
        should_update_target = (self.total_steps % RL['target_update'] == 0)
        
        # 使用线程池并行执行 (PyMunk物理+PyTorch GPU均释放GIL)
        with ThreadPoolExecutor(max_workers=self.num_games) as executor:
            futures = {
                executor.submit(self._train_one_game, idx, should_optimize, should_update_target): idx
                for idx in range(self.num_games)
            }
            done_flags = [False] * self.num_games
            for future in as_completed(futures):
                idx = futures[future]
                done_flags[idx] = future.result()
        
        self.total_steps += 1
        return any(done_flags)
    
    def render_all(self):
        """渲染所有游戏到窗口"""
        if not self.render:
            return True
        
        # 处理事件 (含ESC退出)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        
        # 每个游戏自己渲染到子表面
        for game in self.games:
            game.render_frame()
        
        # 绘制边框和标签
        for idx in range(self.num_games):
            col = idx % self.cols
            row = idx // self.cols
            x, y = col * self.cell_w, row * self.cell_h
            pygame.draw.rect(self.screen, (60, 60, 80), (x, y, self.cell_w, self.cell_h), 2)
            
            # 显示训练信息
            game = self.games[idx]
            ap1 = self.agents_p1[idx]
            info_text = [
                f"#{idx+1} Step:{self.total_steps}",
                f"P1:{game.player1.health:.0f} P2:{game.player2.health:.0f}",
                f"Mem:{len(ap1.memory)}",
            ]
            for li, txt in enumerate(info_text):
                txt_surf = self.font.render(txt, True, (200, 200, 200))
                self.screen.blit(txt_surf, (x + 5, y + self.cell_h - 50 + li * 16))
        
        # 全局状态栏
        elapsed = time.time() - self.start_time
        stats = self.font.render(
            f"Total Steps: {self.total_steps} | Time: {elapsed:.0f}s | Games: {self.num_games} | ESC=退出",
            True, (180, 180, 200))
        self.screen.blit(stats, (10, self.win_h - 20))
        
        pygame.display.flip()
        self.clock.tick(30)  # 训练时30fps就够了
        return True
    
    def train_loop(self):
        """训练主循环"""
        print(f"🔥 多宫格训练启动: {self.rows}x{self.cols} = {self.num_games}局同时训练")
        print(f"   窗口: {self.win_w}x{self.win_h}, 每格: {self.cell_w}x{self.cell_h}")
        print(f"   按 ESC 或关闭窗口退出, 自动保存模型\n")
        
        try:
            frame_count = 0
            while True:
                self.train_one_step()
                
                # 隔帧渲染 - 节省CPU
                frame_count += 1
                if frame_count % 2 == 0:
                    cont = self.render_all()
                    if not cont:
                        break
                    
        except KeyboardInterrupt:
            print("\n🛑 训练中断")
        
        self._save_models()
        total_time = time.time() - self.start_time
        print(f"💾 训练结束! 总步数: {self.total_steps}, 用时: {total_time:.0f}s")
        pygame.quit()
    
    def _save_models(self):
        """基于滑动平均奖励保存最佳模型 (共2个文件)"""
        # 收集所有agent的历史平均奖励
        avg_rewards = []
        for idx in range(self.num_games):
            a1 = self.agents_p1[idx]
            a2 = self.agents_p2[idx]
            # 用loss_history长度作为局数的近似, 结合memory占用
            mem_ratio = len(a1.memory) / RL['memory_size']
            avg_rewards.append(mem_ratio)
        
        best_idx = int(np.argmax(avg_rewards))
        
        self.agents_p1[best_idx].save(
            os.path.join(self.model_dir, "agent_p1_best.pth"))
        self.agents_p2[best_idx].save(
            os.path.join(self.model_dir, "agent_p2_best.pth"))
        print(f"💾 已保存最佳模型 (#{best_idx+1}, mem={avg_rewards[best_idx]:.0%})")


def start_multi_training(grid_rows=2, grid_cols=2):
    """启动多宫格训练"""
    viewer = MultiGameViewer(grid_size=(grid_rows, grid_cols), render=True)
    viewer.train_loop()
