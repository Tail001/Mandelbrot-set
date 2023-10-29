import numpy as np
import gc
import cv2
from tqdm import tqdm
import colorsys

# s -0.75 + 0.1i,
def gradient_color(_iteration):
    if 0 <= _iteration < 20:

        hue = 0.75 - (_iteration / 20) * 0.25
    elif 20 <= _iteration < 40:
        hue = 0.5 + ((_iteration - 20) / 20) * 0.25
    else:

        hue = 0.75 - ((_iteration - 40) % 20) / 20 * 0.25

    r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1, 1)]
    return (b, g, r)

# resolution -> how many point in this interval
def mandelbrot_set(iteration: int, resolution: tuple, _resize: tuple, _center: tuple) -> np.ndarray:
    
    def build_grid() -> np.ndarray:
        xline = np.linspace(-2 / _resize[0] + _center[0], 2 / _resize[0] + _center[0], resolution[0])  # real axis
        yline = np.linspace(-3 / _resize[1] + _center[1], 3 / _resize[1] + _center[1], resolution[1])  # imaginary axis
        x, y = np.meshgrid(yline, xline)
        print(f"xvalue: {-2 / _resize[0] + _center[0]}, {_resize[0] + _center[0], resolution[0]}")
        return x + 1j*y  # j: sqrt(-1) image unit

    c = build_grid()
    # print(c.shape)
    z = c *np.ones((resolution[0], resolution[1]))
    background = np.zeros((resolution[0], resolution[1], 3))
    lastmask = np.ones((resolution[0], resolution[1], 3))

    color = np.array([255, 0, 0])

    for i in tqdm(range(iteration)):
        
        # judge whether beyond the range
        mask = (abs(z) <= 2).astype(np.int32) # numpy->ndarray
        # discard escaped value
        z = z*mask
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # union along all previous iteration by using the concept of bit-wised AND
        mask = lastmask*mask

        color = gradient_color(_iteration=i)
        framei = (lastmask - mask)*color
        background += framei
        lastmask = mask

        z = z ** 2 + c # according to formula

        del mask, framei
        gc.collect()
    return background.astype(np.uint8)



def main():
    
    video_name = "Mandelbrot_new.mp4"
    
    videowriter = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('m','p','4','v'), 10.0, (1280, 720))
    
    for i in range(300):
        resize = np.array([2 * (i + 1), 3 * (i+1)]) 
        center = np.array([-0.127129856, -0.739637352 ])
        # center = np.array([-0.136640848, -0.776592847])
        # iteration from 100 -> 1000
        iteration_value = 100 + (3*i)
        itera_100_frame = mandelbrot_set(iteration= iteration_value, resolution=(720, 1280), _resize=resize, _center=center)
        videowriter.write(itera_100_frame)
        cv2.imwrite(f"./image/m_n{i}.png", itera_100_frame)
    

if __name__ == "__main__":
    main()
