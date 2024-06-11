import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.avi', fourcc, 1, (376, 376))
img = cv2.imread("img_aligned.png",cv2.IMREAD_COLOR)
video.write(img)

video.release()