#!/usr/bin/python
# coding=utf-8
import os
import time
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None
from flappybird_wrapper import FlappyBirdWrapper
from dqn_flappybird import Agent, Model, device

def main():
    # 配置信息（不再使用命令行参数，方便直接运行）
    MODEL_PATH = 'dqn_flappybird_best.pt' # 默认调用最佳模型
    SAVE_VIDEO = False
    SAVE_GIF = True
    VIDEO_OUTPUT = 'gameplay.mp4'
    GIF_OUTPUT = 'gameplay-no_double.gif'
    FPS = 30
    MAX_STEPS = 200 # 限制步数防止文件过大

    # 环境初始化
    env = FlappyBirdWrapper(display_screen=False) 
    agent = Agent()
    
    # 手动加载指定的 best 模型
    if os.path.exists(MODEL_PATH):
        agent.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"成功加载最佳模型: {MODEL_PATH}")
    else:
        print(f"未找到 {MODEL_PATH}，尝试加载默认模型...")
        agent.load()

    obs = env.reset()
    total_reward = 0
    pipe_count = 0  # 记录通过的管道数
    
    frames = []
    video_writer = None

    print("开始运行测试...")
    try:
        for step in range(MAX_STEPS):
            act = agent.predict(obs)
            obs, reward, done, _ = env.step(act)
            total_reward += reward
            
            # 在 FlappyBird 中，通常通过一根管子奖励是 +1.0
            if reward > 0.5: 
                pipe_count += 1
                print(f"成功通过第 {pipe_count} 根管子！")

            # 渲染画面
            frame = env.render()
            
            # 视频处理
            if SAVE_VIDEO and cv2:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if video_writer is None:
                    h, w = bgr_frame.shape[:2]
                    video_writer = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
                video_writer.write(bgr_frame)
            
            # GIF处理：为了减小体积，可以每隔 2 帧取一帧（即跳帧）
            if SAVE_GIF and step % 2 == 0:
                frames.append(Image.fromarray(frame))

            if done:
                break
            
            # 限制 GIF 帧数，防止内存溢出或文件过大（最多保留 300 帧）
            if len(frames) > 300 and SAVE_GIF:
                # 如果游戏太长，我们只取前 300 帧或均匀抽样
                pass 

        print(f"\n测试结束！")
        print(f"最终得分 (通过管子数): {pipe_count}")
        print(f"累计奖励值: {total_reward:.2f}")

    finally:
        if video_writer:
            video_writer.release()
            print(f"视频已保存至: {VIDEO_OUTPUT}")
        
        if SAVE_GIF and frames:
            print(f"正在压缩并保存 GIF ({len(frames)} 帧)...")
            # 适当缩小尺寸可以大幅减小 GIF 体积
            first_frame = frames[0]
            w, h = first_frame.size # 显式获取图片宽高
            # 重新调整大小以减小体积
            small_frames = [f.resize((w//2, h//2), Image.Resampling.LANCZOS) for f in frames]
            small_frames[0].save(
                GIF_OUTPUT,
                save_all=True,
                append_images=small_frames[1:],
                duration=int(2000 / FPS), # 因为每2帧取1帧，所以时间翻倍
                loop=0
            )
            print(f"GIF 已保存至: {GIF_OUTPUT}")

if __name__ == '__main__':
    import torch # 确保导入了 torch 用于加载
    main()


if __name__ == '__main__':
    main()