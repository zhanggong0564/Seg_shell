import numpy as np

class AccScores(object):
    def __init__(self,n_classes,ignore_index=None):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes,n_classes))

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def _fast_hist(self,label_true, label_pred):
        mask = (label_true>=0)&(label_true<self.n_classes)
        hist = np.bincount(
            self.n_classes*label_true[mask].astype(int) +label_pred[mask],minlength=self.n_classes**2
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, label_trues, label_preds):
        for l,p in zip(label_trues,label_preds):
            self.confusion_matrix+=self._fast_hist(l.flatten(),p.flatten())

    def get_scores(self):
        hist = self.confusion_matrix
        # ignore unlabel
        if self.ignore_index is not None:
            for index in self.ignore_index:
                hist = np.delete(hist, index, axis=0)
                hist = np.delete(hist, index, axis=1)
        acc = np.diag(hist).sum()/hist.sum()
        acc_cls = np.diag(hist)/hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist)/(hist.sum(axis=1)+hist.sum(axis=0)-np.diag(hist)
        )
        mean_iou = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()

        # set unlabel as nan
        if self.ignore_index is not None:
            for index in self.ignore_index:
                iu = np.insert(iu, index, np.nan)

        cls_iu = dict(zip(range(self.n_classes), iu))#{0: 0.5, 1: 0.5, 2: 0.5} 每个类别对应iou
        score_dict = {"pixel_acc": acc,
                "class_acc": acc_cls,
                "mIou": mean_iou,
                "fwIou": fw_iou,}
        return score_dict,cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self,val,n=1):
        self.val = val
        self.sum+=val
        self.count+=n
        self.avg = self.sum/self.count
