import cv2
from detection import Detect
from PostProcessing import PostProcess



if __name__ == '__main__':
    cap = cv2.VideoCapture('/home/zhanggong/disk/Elements/data/paoyuan/source/data3.avi')
    detector = Detect('/home/zhanggong/disk/Extern/workspace/seg/Bisnet_best.pth')
    post_process = PostProcess()
    while True:
        import time
        start = time.time()
        res, img = cap.read()
        if res:
            pre,img_src = detector.detect(img)##0.03362393379211426  0.03487873077392578
            post_process.get_info(pre)
            img_show = post_process.draw_show(img_src)
            add_image = cv2.addWeighted(img_show, 0.7, pre, 0.3, 0)
            add_image = cv2.hconcat([add_image,pre])
            post_process.empty_info()
            cv2.imshow('image',add_image)
            k = cv2.waitKey(20)
            end = time.time()
            print(end - start)
            if k==27:
                break
        else:
            cap.release()
    cv2.destroyAllWindows()
