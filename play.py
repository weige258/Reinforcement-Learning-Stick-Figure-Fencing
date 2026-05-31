"""
人机对战模式 - 人类玩家 VS AI智能体
"""
import sys
import pygame
import torch

from game_config import *
from fencing_game import FencingGame
from dqn_agent import DQNAgent


def human_vs_ai(model_path=None):
    """人类玩家 VS AI - WASD移动 + 鼠标拖拽挥剑"""
    print("=" * 60)
    print(" 火柴人击剑格斗 - 人机对战")
    print(" 🖱️ 鼠标左键拖拽 = 挥剑 (方向+速度决定攻击)")
    print(" ⌨️  WASD = 移动  |  W = 跳跃")
    print("    Space = 重置  |  ESC = 退出")
    print("=" * 60)
    
    game = FencingGame(render=True)
    agent = DQNAgent(RL['state_dim'], RL['action_dim'])
    
    if model_path:
        agent.load(model_path)
    else:
        import os
        default_path = os.path.join(os.path.dirname(__file__), 'models', 'agent_p2_final.pth')
        if os.path.exists(default_path):
            agent.load(default_path)
            print(f"已加载模型: {default_path}")
        else:
            print("未找到模型，AI将使用随机动作")
    
    agent.policy_net.eval()
    
    running = True
    state = game.reset()
    clock = pygame.time.Clock()
    
    try:
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        state = game.reset()
                        print("游戏重置!")
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # AI选择动作
            ai_action = agent.select_action(state, eval_mode=True)
            
            # 执行人机对战一步 (内部处理鼠标+WASD输入)
            next_state, reward, done, info = game.human_step(ai_action)
            
            # 渲染
            cont = game.render_frame()
            if not cont:
                running = False
            
            state = next_state
            clock.tick(FPS)
            
            if done:
                if info['winner'] == game.player1:
                    print(f"🎉 你赢了! 剩余血量: {info['health1']:.0f}")
                elif info['winner'] == game.player2:
                    print(f"💀 AI 赢了! 你的血量: {info['health1']:.0f}")
                else:
                    print(f"⚖️ 平局! P1: {info['health1']:.0f} vs P2: {info['health2']:.0f}")
                
                pygame.time.wait(1000)
                state = game.reset()
    
    finally:
        game.close()


def ai_vs_ai(model1_path=None, model2_path=None):
    """AI VS AI (观看两个智能体对战)"""
    print("观看 AI VS AI 对战...")
    
    game = FencingGame(render=True)
    agent1 = DQNAgent(RL['state_dim'], RL['action_dim'])
    agent2 = DQNAgent(RL['state_dim'], RL['action_dim'])
    
    if model1_path:
        agent1.load(model1_path)
    if model2_path:
        agent2.load(model2_path)
    
    agent1.policy_net.eval()
    agent2.policy_net.eval()
    
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    for episode in range(10):
        state = game.reset()
        total_steps = 0
        
        while total_steps < RL['max_steps_per_episode']:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.close()
                    return
            
            action1 = agent1.select_action(state, eval_mode=True)
            action2 = agent2.select_action(state, eval_mode=True)
            
            next_state, r1, r2, done, info = game.step(action1, action2)
            
            cont = game.render_frame()
            if not cont:
                game.close()
                return
            
            state = next_state
            total_steps += 1
            pygame.time.wait(16)  # ~60fps
            
            if done:
                break
        
        if info['winner'] == game.player1:
            p1_wins += 1
            result = "P1 胜"
        elif info['winner'] == game.player2:
            p2_wins += 1
            result = "P2 胜"
        else:
            draws += 1
            result = "平局"
        
        print(f"局 {episode+1}: {result} | P1血量={info['health1']:.0f} P2血量={info['health2']:.0f}")
        pygame.time.wait(500)
    
    print(f"\n总计: P1胜 {p1_wins}, P2胜 {p2_wins}, 平局 {draws}")
    game.close()


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='火柴人击剑格斗 - 对战模式')
    parser.add_argument('--mode', type=str, default='human_vs_ai',
                       choices=['human_vs_ai', 'ai_vs_ai'],
                       help='对战模式')
    parser.add_argument('--model1', type=str, default='',
                       help='智能体1模型路径 (P1)')
    parser.add_argument('--model2', type=str, default='',
                       help='智能体2模型路径 (P2)')
    
    args = parser.parse_args()
    
    if args.mode == 'human_vs_ai':
        human_vs_ai(model_path=args.model2 if args.model2 else '')
    elif args.mode == 'ai_vs_ai':
        ai_vs_ai(model1_path=args.model1, model2_path=args.model2)


if __name__ == '__main__':
    main()
