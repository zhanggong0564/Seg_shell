import cv2
import numpy as np

class PostProcess(object):
    def __init__(self):
        super(PostProcess, self).__init__()
        self.new_contour = []
        self.x1y1 = []
        self.x2y2 = []
        self.box = []
        self.cxcy = []
        self.shift_x = None
    def get_info(self,mask):
        _,w,_= mask.shape
        binary = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        binary[binary > 0] = 255
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, )
        if self.filter_feature(contours):
            for cnt in self.new_contour:
                M = cv2.moments(cnt)
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                self.cxcy.append((cx,cy))
            self.cxcy.sort(key=lambda x:x[1])
            self.shift_x = (self.cxcy[-1][0] - w//2)/(w/2)
            return self.shift_x
        else:
            return 0
    def empty_info(self):
        self.new_contour = []
        self.x1y1 = []
        self.x2y2 = []
        self.box = []
        self.cxcy = []
    def filter_feature(self,contours):
        for cnt in contours:
            contours_area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            length = cv2.arcLength(cnt, True)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            scale_w_h = w / h
            if 0.3 < scale_w_h < 4.9 and 40 < contours_area < 29556 and 26 < length < 1344:
                self.new_contour.append(cnt)
                self.x1y1.append((x,y))
                self.x2y2.append((x+w,y+h))
                self.box.append(box)
        if self.new_contour:
            return True
        else:
            return False

    def draw_show(self,mask):
        if self.new_contour:
            for i,(x,y) in enumerate(zip(self.x1y1,self.x2y2)):
                # cv2.rectangle(mask, x,y,(0, 255, 0), 2)
                cv2.drawContours(mask, [self.box[i]], 0, (0, 0, 255), 1, cv2.LINE_AA)
        return mask.copy()


if __name__ == '__main__':
    mask = cv2.imread('./test/596.png')
    pproce = PostProcess()
    pproce.get_info(mask)
    pproce.draw_show()
    print(pproce.cxcy)
    print(pproce.shift_x)
    pproce.empty_info()
    cv2.waitKey()
    cv2.destroyAllWindows()
