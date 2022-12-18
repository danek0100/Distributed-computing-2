### LB #2
### Соболев Данил, Ольга Фролова, Диана Шарибжинова 19ПМИ-1

import matplotlib.pyplot as plt
from matplotlib.patches import Arc

import pandas as pd
import numpy as np
import celluloid as cld


def make_movie(df: pd.DataFrame, filename: str, isOneThread=True):
    camera = cld.Camera(plt.figure())
    if isOneThread:
        plt.title(f'Animation from 0 to 2.5 seconds with step 0.01 sequential')
    else:
        plt.title(f'Animation from 0 to 2.5 seconds with step 0.01 parallel')

    for step in range(250):
        ax = plt.gca().set_aspect('equal')
        plt.scatter(x=[df.x1[step], df.x2[step]], y=[df.y[step], df.y[step]], c='b', s=80)
        radius = np.sqrt((df.x1[step] - df.Ax[step]) ** 2 + (df.y[step] - df.Ay[step]) ** 2)

        arc_angles = np.linspace(-np.pi / 2 - df.f1[step], -np.pi / 2, 20)
        arc_xs = radius * np.cos(arc_angles) + df.x1[step]
        arc_ys = radius * np.sin(arc_angles) + df.y[step]
        plt.plot(arc_xs, arc_ys, color='red', lw=3)

        arc_angles = np.linspace(-np.pi / 2, -np.pi / 2 + df.f2[step], 20)
        arc_xs = radius * np.cos(arc_angles) + df.x2[step]
        arc_ys = radius * np.sin(arc_angles) + df.y[step]
        plt.plot(arc_xs, arc_ys, color='red', lw=3)
        camera.snap()
    anim = camera.animate(blit=True)
    anim.save(filename)


single_thread_data = pd.read_csv('single_tread.csv', sep=',')
parallel_data = pd.read_csv('multi_treads.csv', sep=',')

print(single_thread_data.keys())
plt.plot(single_thread_data.step, single_thread_data.time)
plt.xlabel('step')
plt.ylabel('calculation time')
plt.title('Execution time graph (one thread)')
plt.savefig("Calc time and Step single thread.png")

plt.close()
plt.plot(parallel_data.step, parallel_data.time)
plt.xlabel('step')
plt.ylabel('calculation time')
plt.title('Execution time graph (multi threads)')
plt.savefig("Calc time and Step multi thread.png")


speed_down = parallel_data.time / single_thread_data.time

print(speed_down)

plt.close()
plt.plot(parallel_data.step, speed_down)
plt.xlabel('step')
plt.ylabel('calculation time')
plt.title('Speed down')
plt.savefig("Calc time and speed down.png")


### Эффективность E=S/p, где S - время паралельных вычислений, p - число потоков
mean_time = parallel_data.time.mean()
E = mean_time / 5
print(E)

# Clean Time with cuda: 35000 nanos
# Clean Time with cuda: 11000 nanos
# Clean Time with cuda: 11500 nanos
# Clean Time without Cuda: 14300 nanos
# Clean Time without Cuda: 1000 nanos
# Clean Time without Cuda: 700 nanos

make_movie(parallel_data, 'animation_multi.gif', False)
make_movie(single_thread_data, 'animation_single.gif', True)
